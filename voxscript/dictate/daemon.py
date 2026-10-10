"""VOXSCRIPT dictation daemon.

Press the hotkey (or run `voxscript-ctl toggle`) to start listening.
Press it again to stop. The audio is transcribed locally with faster-whisper,
then the text is copied to the clipboard and pasted into the focused window.

Trigger sources:
  * global hotkey via pynput (X11, and XWayland windows on Wayland)
  * control socket via `voxscript-ctl`, bound to a compositor shortcut (works on native Wayland)
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from voxscript.audio import AudioError, load_sounddevice, resolve_device
from voxscript.config import Settings
from voxscript.dictate.injector import InjectionError, Injector, PasteError, check_tools, detect_backend
from voxscript.dictate.ipc import ControlServer, default_socket_path
from voxscript.transcriber import Transcriber, TranscriptionError

log = logging.getLogger("voxscript.dictate")

SAMPLE_RATE = 16000
MODIFIER_NAMES = {
    "ctrl": "<ctrl>", "control": "<ctrl>", "alt": "<alt>", "shift": "<shift>",
    "super": "<cmd>", "meta": "<cmd>", "cmd": "<cmd>", "win": "<cmd>",
}


def normalize_combo(combo: str) -> str:
    """Convert 'ctrl+space' into pynput's '<ctrl>+<space>' format. Already-correct strings pass through."""
    out: list[str] = []
    for raw in (p.strip() for p in combo.split("+")):
        if not raw:
            continue
        low = raw.lower()
        if low in MODIFIER_NAMES:
            out.append(MODIFIER_NAMES[low])
        elif raw.startswith("<") and raw.endswith(">"):
            out.append(low)
        elif len(raw) == 1:
            out.append(low)
        else:
            out.append(f"<{low}>")
    return "+".join(out)


@dataclass
class DictateConfig:
    hotkey_label: str = "Ctrl+Space"
    socket_path: Path = field(default_factory=default_socket_path)
    language: str | None = None
    max_seconds: float = 120.0      # safety cap so a forgotten recording cannot run forever
    min_seconds: float = 0.4        # shorter taps are ignored
    toggle_debounce_s: float = 0.5  # key auto-repeat must not start and stop the recording
    notify: bool = True


class MicCapture:
    """Opens the microphone only while recording, so the mic is off the rest of the time."""

    def __init__(self, device: int | None) -> None:
        self._device = device
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._stream = None

    def start(self) -> None:
        sd = load_sounddevice()
        self._chunks = []
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                device=self._device, callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise AudioError(f"Could not open microphone: {exc}") from exc

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        with self._lock:
            self._chunks.append(indata[:, 0].copy())

    def stop(self) -> np.ndarray:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        with self._lock:
            chunks, self._chunks = self._chunks, []
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


class DictationController:
    """State machine: IDLE -> RECORDING -> BUSY (transcribe + inject) -> IDLE."""

    IDLE, RECORDING, BUSY = "idle", "recording", "busy"

    def __init__(self, cfg: DictateConfig, transcriber: Transcriber, injector: Injector, capture: MicCapture) -> None:
        self.cfg = cfg
        self._transcriber = transcriber
        self._injector = injector
        self._capture = capture
        self._lock = threading.Lock()
        self._state = self.IDLE
        self._timer: threading.Timer | None = None
        self._last_toggle = 0.0

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def toggle(self) -> str:
        now = time.monotonic()
        with self._lock:
            if now - self._last_toggle < self.cfg.toggle_debounce_s:
                return "ignored (debounce)"
            self._last_toggle = now
            recording = self._state == self.RECORDING
        return self.stop() if recording else self.start()

    def start(self) -> str:
        with self._lock:
            if self._state == self.BUSY:
                log.warning("Still processing the previous dictation. Wait for the 'Done' line first.")
                return "busy"
            if self._state == self.RECORDING:
                return "already recording"
            try:
                self._capture.start()
            except AudioError as exc:
                log.error("%s", exc)
                return f"error: {exc}"
            self._state = self.RECORDING
            self._timer = threading.Timer(self.cfg.max_seconds, self._auto_stop)
            self._timer.daemon = True
            self._timer.start()
        log.info("●  LISTENING    speak now, press %s again to stop", self.cfg.hotkey_label)
        self._notify("Listening...", "Press the hotkey again to stop")
        return "recording"

    def stop(self) -> str:
        with self._lock:
            if self._state != self.RECORDING:
                return "not recording"
            audio = self._capture.stop()
            self._state = self.BUSY
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        log.info("■  STOPPED      captured %.1f s of audio", len(audio) / SAMPLE_RATE)
        threading.Thread(target=self._process, args=(audio,), name="dictation", daemon=True).start()
        return "stopped"

    def _auto_stop(self) -> None:
        log.warning("Maximum recording length (%.0f s) reached; stopping.", self.cfg.max_seconds)
        self.stop()

    def _process(self, audio: np.ndarray) -> None:
        try:
            seconds = len(audio) / SAMPLE_RATE
            if seconds < self.cfg.min_seconds:
                log.info("…  Too short (%.2f s), ignored.", seconds)
                return

            log.info("…  TRANSCRIBING %.1f s of audio with faster-whisper", seconds)
            started = time.monotonic()
            segments, language, _ = self._transcriber.transcribe_samples(
                audio, language=self.cfg.language, beam_size=5
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            took = time.monotonic() - started
            if not text:
                log.info("   No speech detected (%.2f s).", took)
                return
            log.info("   Transcribed in %.2f s [%s]: %s", took, language or "?", text[:200])

            log.info("⌨  INJECTING   %d characters via %s", len(text), self._injector.backend)
            outcome = self._injector.inject(text)
            log.info("✓  Done: text %s into the focused window.", outcome)
            self._notify("Dictation done", text[:120])
        except PasteError as exc:
            log.warning("Paste failed (%s). The text is on the clipboard; press Ctrl+V to insert it.", exc)
        except (TranscriptionError, InjectionError) as exc:
            log.error("%s", exc)
        except Exception:
            log.exception("Unexpected error while processing dictation")
        finally:
            with self._lock:
                self._state = self.IDLE

    def _notify(self, title: str, body: str) -> None:
        if self.cfg.notify and shutil.which("notify-send"):
            subprocess.Popen(
                ["notify-send", "-a", "VOXSCRIPT", "-t", "1500", title, body],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )


def start_global_hotkey(combo: str, callback: Callable[[], object]):  # noqa: ANN201
    """Register a system-wide hotkey with pynput. Returns the listener, or None if unavailable."""
    if not os.environ.get("DISPLAY"):
        log.warning("No X display (DISPLAY unset): pynput hotkey disabled. Use `voxscript-ctl toggle` instead.")
        return None
    try:
        from pynput import keyboard

        listener = keyboard.GlobalHotKeys({combo: callback})
        listener.start()
    except Exception as exc:  # pynput raises ImportError when no X server is reachable
        log.warning("Global hotkey unavailable (%s). Use `voxscript-ctl toggle` from a compositor binding.", exc)
        return None
    log.info("Global hotkey registered: %s (X11 and XWayland windows)", combo)
    return listener


def handle_command(line: str, controller: DictationController, stop_event: threading.Event) -> str:
    """Dispatch one control-socket command. Returns the reply sent to voxscript-ctl."""
    word = (line.split() or [""])[0].lower()
    if word == "toggle":
        return f"ok: {controller.toggle()}"
    if word == "start":
        return f"ok: {controller.start()}"
    if word == "stop":
        return f"ok: {controller.stop()}"
    if word == "status":
        return f"ok: {controller.state}"
    if word == "ping":
        return "ok: pong"
    if word == "quit":
        stop_event.set()
        return "ok: shutting down"
    return f"error: unknown command {word!r} (toggle|start|stop|status|ping|quit)"


def build_parser() -> argparse.ArgumentParser:
    settings = Settings()
    p = argparse.ArgumentParser(
        prog="voxscript-daemon",
        description="System-wide dictation: press a hotkey, speak, and the text is pasted into the focused window.",
    )
    p.add_argument("--hotkey", default="ctrl+space",
                   help="global toggle shortcut, e.g. 'ctrl+space' or 'ctrl+w' (default: %(default)s)")
    p.add_argument("--no-hotkey", action="store_true",
                   help="do not register a global hotkey (use voxscript-ctl from a compositor binding)")
    p.add_argument("--socket", type=Path, default=default_socket_path(), help="control socket path")
    p.add_argument("--whisper-model", default=settings.whisper_model,
                   help="faster-whisper model (default: %(default)s; tiny.en or base.en are fastest)")
    p.add_argument("--whisper-device", default=settings.whisper_device, choices=["cpu", "cuda", "auto"])
    p.add_argument("--compute-type", default=settings.whisper_compute_type)
    p.add_argument("--language", default=None, help="spoken language, e.g. en (default: auto; en for *.en models)")
    p.add_argument("--mic", default=None, help="input device index or name substring")
    p.add_argument("--inject", choices=["auto", "x11", "wayland"], default="auto",
                   help="force the output backend (default: auto-detect from the session)")
    p.add_argument("--inject-mode", choices=["paste", "type"], default="paste",
                   help="paste via clipboard (default) or type characters directly")
    p.add_argument("--paste-combo", default="ctrl+v",
                   help="paste keystroke, e.g. ctrl+v, or ctrl+shift+v for terminal emulators")
    p.add_argument("--max-seconds", type=float, default=120.0, help="auto-stop a recording after N seconds")
    p.add_argument("--no-notify", action="store_true", help="disable desktop notifications")
    p.add_argument("--check", action="store_true", help="check dependencies and exit")
    p.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, or ERROR")
    return p


def run_check(backend: str) -> int:
    print(f"Display backend : {backend}")
    print(f"DISPLAY         : {os.environ.get('DISPLAY', '(unset)')}")
    print(f"WAYLAND_DISPLAY : {os.environ.get('WAYLAND_DISPLAY', '(unset)')}")
    missing = []
    for tool, ok in check_tools(backend) + [("notify-send", shutil.which("notify-send") is not None)]:
        print(f"  {'OK     ' if ok else 'MISSING'}  {tool}")
        if not ok:
            missing.append(tool)
    if missing:
        print("\nInstall on Arch:")
        print("  X11     : sudo pacman -S --needed xdotool xclip libnotify")
        print("  Wayland : sudo pacman -S --needed wl-clipboard ydotool libnotify")
        print("  (ydotool also needs the ydotoold service running; see README)")
    return 0 if not missing else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    if args.check:
        backend = detect_backend() if args.inject == "auto" else args.inject
        return run_check(backend)

    cfg = DictateConfig(
        hotkey_label=args.hotkey.replace("<", "").replace(">", "").title(),
        socket_path=args.socket,
        language=args.language,
        max_seconds=args.max_seconds,
        notify=not args.no_notify,
    )
    settings = Settings()
    settings.whisper_model = args.whisper_model
    settings.whisper_device = args.whisper_device
    settings.whisper_compute_type = args.compute_type

    try:
        injector = Injector(args.inject, args.paste_combo, args.inject_mode)
        missing = [tool for tool, ok in check_tools(injector.backend) if not ok]
        if missing:
            log.warning("Missing tools: %s. Run `voxscript-daemon --check` for install commands.", ", ".join(missing))
        log.info("Session backend: %s. Paste via %s.", injector.backend, args.paste_combo)
        if injector.backend == "wayland":
            log.info("On native Wayland windows the pynput hotkey cannot see keys. "
                     "Bind your compositor shortcut to `voxscript-ctl toggle`.")

        log.info("Loading Whisper model %s (first run may download it)...", settings.whisper_model)
        transcriber = Transcriber(settings)
        transcriber.load()
        capture = MicCapture(resolve_device(args.mic))
    except (AudioError, InjectionError, TranscriptionError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    stop_event = threading.Event()
    controller = DictationController(cfg, transcriber, injector, capture)
    server = ControlServer(cfg.socket_path, lambda line: handle_command(line, controller, stop_event))
    try:
        server.start()
    except (RuntimeError, OSError) as exc:
        log.error("%s", exc)
        return 1

    hotkey = None
    if not args.no_hotkey:
        hotkey = start_global_hotkey(normalize_combo(args.hotkey), controller.toggle)

    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    log.info("Ready. Toggle with %s or `voxscript-ctl toggle`. Ctrl+C quits.", cfg.hotkey_label)
    try:
        while not stop_event.wait(0.5):
            pass
    finally:
        log.info("Shutting down...")
        if hotkey is not None:
            hotkey.stop()
        if controller.state == DictationController.RECORDING:
            controller.stop()  # keep what was said so far
        deadline = time.monotonic() + 30
        while controller.state != DictationController.IDLE and time.monotonic() < deadline:
            time.sleep(0.2)
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
