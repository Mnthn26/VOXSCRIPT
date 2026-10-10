"""Put transcribed text into the focused window: clipboard + paste, or direct typing.

Backends:
  x11      -> xclip (clipboard) + xdotool (keystrokes)
  wayland  -> wl-clipboard (wl-copy) + ydotool (uinput keystrokes; needs the ydotoold daemon)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

# Linux input-event keycodes (linux/input-event-codes.h), used by ydotool.
KEYCODES = {"ctrl": 29, "shift": 42, "alt": 56, "space": 57, "v": 47, "insert": 110}


class InjectionError(RuntimeError):
    """Text could not be placed into the focused window."""


class PasteError(InjectionError):
    """The text reached the clipboard, but the paste keystroke failed."""


def detect_backend() -> str:
    """Wayland if a Wayland display is present, otherwise X11 if DISPLAY is set."""
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "none"


def check_tools(backend: str) -> list[tuple[str, bool]]:
    """Return (tool, installed) pairs for a backend, for `--check`."""
    names = ["wl-copy", "ydotool"] if backend == "wayland" else ["xclip", "xdotool"]
    return [(name, shutil.which(name) is not None) for name in names]


def _ydotool_key_args(combo: str) -> list[str]:
    """Convert 'ctrl+shift+v' into ydotool raw key events, releasing stray modifiers first."""
    names = [part.strip().lower() for part in combo.split("+") if part.strip()]
    unknown = [n for n in names if n not in KEYCODES]
    if unknown:
        raise ValueError(f"Unsupported key(s) in paste combo {combo!r}: {', '.join(unknown)}")
    codes = [KEYCODES[n] for n in names]
    release_stray = [f"{KEYCODES[k]}:0" for k in ("space", "shift", "ctrl")]  # the hotkey may still be held
    press = [f"{c}:1" for c in codes]
    release = [f"{c}:0" for c in reversed(codes)]
    return release_stray + press + release


class Injector:
    def __init__(self, backend: str = "auto", paste_combo: str = "ctrl+v", mode: str = "paste",
                 settle_s: float = 0.12) -> None:
        self.backend = detect_backend() if backend == "auto" else backend
        if self.backend not in ("x11", "wayland"):
            raise InjectionError("No display found (DISPLAY and WAYLAND_DISPLAY are both unset).")
        if mode not in ("paste", "type"):
            raise ValueError("mode must be 'paste' or 'type'")
        self.paste_combo = paste_combo.lower()
        self.mode = mode
        self.settle_s = settle_s
        if self.backend == "wayland":
            _ydotool_key_args(self.paste_combo)  # validate early

    def inject(self, text: str) -> str:
        """Insert text into the focused window. Returns 'pasted' or 'typed'."""
        if self.mode == "type":
            self._run(self._type_command(text))
            return "typed"
        self._copy(text)
        time.sleep(self.settle_s)  # let the clipboard owner settle before pasting
        try:
            self._run(self._paste_command())
        except InjectionError as exc:
            raise PasteError(str(exc)) from exc
        return "pasted"

    def _copy(self, text: str) -> None:
        cmd = ["wl-copy"] if self.backend == "wayland" else ["xclip", "-selection", "clipboard"]
        try:
            # xclip and wl-copy fork into the background to serve the clipboard. Their stdout and stderr
            # must not be pipes, or subprocess would wait for them forever.
            subprocess.run(cmd, input=text.encode("utf-8"), stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=5, check=True)
        except FileNotFoundError as exc:
            raise InjectionError(f"'{cmd[0]}' is not installed") from exc
        except subprocess.CalledProcessError as exc:
            raise InjectionError(f"{cmd[0]} exited with status {exc.returncode}") from exc
        except subprocess.TimeoutExpired as exc:
            raise InjectionError(f"{cmd[0]} timed out") from exc

    def _paste_command(self) -> list[str]:
        if self.backend == "wayland":
            return ["ydotool", "key", *_ydotool_key_args(self.paste_combo)]
        return ["xdotool", "key", "--clearmodifiers", self.paste_combo]

    def _type_command(self, text: str) -> list[str]:
        if self.backend == "wayland":
            return ["ydotool", "type", "--", text]
        return ["xdotool", "type", "--clearmodifiers", "--delay", "8", "--", text]

    @staticmethod
    def _run(cmd: list[str]) -> None:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except FileNotFoundError as exc:
            raise InjectionError(f"'{cmd[0]}' is not installed") from exc
        except subprocess.TimeoutExpired as exc:
            raise InjectionError(f"{cmd[0]} timed out") from exc
        if proc.returncode != 0:
            detail = proc.stderr.strip() or "no error output"
            hint = " (is ydotoold running?)" if cmd[0] == "ydotool" else ""
            raise InjectionError(f"{cmd[0]} failed with status {proc.returncode}: {detail}{hint}")
