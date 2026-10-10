"""Offline tests for the dictation daemon. No microphone, display, clipboard, or Whisper model needed."""

from __future__ import annotations

import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from voxscript.dictate import injector as inj
from voxscript.dictate.daemon import DictateConfig, DictationController, handle_command, normalize_combo
from voxscript.dictate.ipc import ControlServer, send_command
from voxscript.transcriber import Segment


def test_normalize_combo():
    assert normalize_combo("ctrl+space") == "<ctrl>+<space>"
    assert normalize_combo("<ctrl>+<space>") == "<ctrl>+<space>"
    assert normalize_combo("ctrl+alt+d") == "<ctrl>+<alt>+d"
    assert normalize_combo("super+w") == "<cmd>+w"


def test_ydotool_paste_sequence_releases_stray_modifiers_first():
    assert inj._ydotool_key_args("ctrl+v") == [
        "57:0", "42:0", "29:0",          # release space, shift, ctrl the user may still hold
        "29:1", "47:1", "47:0", "29:0",  # press ctrl, press v, release v, release ctrl
    ]


def test_ydotool_rejects_unknown_key():
    with pytest.raises(ValueError):
        inj._ydotool_key_args("ctrl+banana")


class Recorder:
    """Replaces subprocess.run so commands can be inspected without running them."""

    def __init__(self, fail_on: str | None = None):
        self.calls: list[list[str]] = []
        self.fail_on = fail_on

    def __call__(self, cmd, **kwargs):  # noqa: ANN001
        self.calls.append(cmd)
        if self.fail_on and cmd[0] == self.fail_on:
            return subprocess.CompletedProcess(cmd, 1, "", "boom")
        return subprocess.CompletedProcess(cmd, 0, "", "")


def test_wayland_injection_uses_wl_copy_then_ydotool(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(inj.subprocess, "run", rec)
    injector = inj.Injector(backend="wayland", paste_combo="ctrl+v", settle_s=0)
    assert injector.inject("hello world") == "pasted"
    assert rec.calls[0] == ["wl-copy"]
    assert rec.calls[1][:3] == ["ydotool", "key", "57:0"]


def test_x11_type_mode_uses_xdotool(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(inj.subprocess, "run", rec)
    injector = inj.Injector(backend="x11", mode="type", settle_s=0)
    assert injector.inject("hi") == "typed"
    assert rec.calls == [["xdotool", "type", "--clearmodifiers", "--delay", "8", "--", "hi"]]


def test_paste_failure_is_reported_but_text_stays_on_clipboard(monkeypatch):
    rec = Recorder(fail_on="ydotool")
    monkeypatch.setattr(inj.subprocess, "run", rec)
    injector = inj.Injector(backend="wayland", settle_s=0)
    with pytest.raises(inj.PasteError):
        injector.inject("text")
    assert rec.calls[0] == ["wl-copy"]  # the clipboard was still written


def test_detect_backend(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    assert inj.detect_backend() == "wayland"
    monkeypatch.delenv("WAYLAND_DISPLAY")
    monkeypatch.setenv("DISPLAY", ":0")
    assert inj.detect_backend() == "x11"
    monkeypatch.delenv("DISPLAY")
    assert inj.detect_backend() == "none"


class FakeCapture:
    def __init__(self, seconds: float = 1.0):
        self.seconds = seconds
        self.started = 0

    def start(self):
        self.started += 1

    def stop(self) -> np.ndarray:
        return np.full(int(self.seconds * 16000), 0.1, dtype=np.float32)


class FakeTranscriber:
    def __init__(self, text: str = "hello there"):
        self.text = text
        self.calls = 0

    def transcribe_samples(self, samples, language=None, beam_size=5, vad=True):  # noqa: ANN001
        self.calls += 1
        return [Segment(0.0, 1.0, self.text)], "en", 0.9


class FakeInjector:
    backend = "x11"

    def __init__(self):
        self.injected: list[str] = []

    def inject(self, text: str) -> str:
        self.injected.append(text)
        return "pasted"


def _wait_idle(controller: DictationController, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while controller.state != DictationController.IDLE and time.monotonic() < deadline:
        time.sleep(0.01)


def _controller(**overrides):
    cfg = DictateConfig(toggle_debounce_s=0.0, notify=False, **overrides)
    transcriber, injector, capture = FakeTranscriber(), FakeInjector(), FakeCapture()
    return DictationController(cfg, transcriber, injector, capture), transcriber, injector, capture


def test_toggle_records_then_transcribes_and_injects():
    controller, transcriber, injector, capture = _controller()
    assert controller.toggle() == "recording"
    assert controller.state == DictationController.RECORDING
    assert controller.toggle() == "stopped"
    _wait_idle(controller)
    assert transcriber.calls == 1
    assert injector.injected == ["hello there"]
    assert capture.started == 1


def test_short_tap_is_ignored():
    controller, transcriber, injector, _ = _controller(min_seconds=0.5)
    controller._capture = FakeCapture(seconds=0.1)
    controller.toggle()
    controller.toggle()
    _wait_idle(controller)
    assert transcriber.calls == 0
    assert injector.injected == []


def test_no_speech_does_not_inject():
    controller, _, injector, _ = _controller()
    controller._transcriber = FakeTranscriber(text="   ")
    controller.toggle()
    controller.toggle()
    _wait_idle(controller)
    assert injector.injected == []


def test_debounce_blocks_key_repeat():
    cfg = DictateConfig(toggle_debounce_s=5.0, notify=False)
    controller = DictationController(cfg, FakeTranscriber(), FakeInjector(), FakeCapture())
    assert controller.toggle() == "recording"
    assert controller.toggle() == "ignored (debounce)"
    assert controller.state == DictationController.RECORDING


def test_max_seconds_auto_stops():
    controller, _, injector, _ = _controller(max_seconds=0.2)
    controller.toggle()
    _wait_idle(controller)
    assert controller.state == DictationController.IDLE
    assert injector.injected == ["hello there"]


def test_socket_roundtrip_and_command_dispatch():
    controller, _, injector, _ = _controller()
    stop_event = threading.Event()
    # Use a short path under /tmp, since AF_UNIX paths are limited to ~108 characters.
    sock_path = Path(tempfile.mkdtemp(dir="/tmp")) / "vs.sock"
    server = ControlServer(sock_path, lambda line: handle_command(line, controller, stop_event))
    server.start()
    try:
        assert send_command(sock_path, "ping") == "ok: pong"
        assert send_command(sock_path, "status") == "ok: idle"
        assert send_command(sock_path, "toggle") == "ok: recording"
        assert send_command(sock_path, "toggle").startswith("ok: ")  # stop request
        _wait_idle(controller)
        assert injector.injected == ["hello there"]
        assert send_command(sock_path, "bogus").startswith("error")
        assert not stop_event.is_set()
        assert send_command(sock_path, "quit") == "ok: shutting down"
        assert stop_event.is_set()
    finally:
        server.stop()
    assert not sock_path.exists()


def test_second_daemon_refuses_to_start_over_a_live_socket():
    sock_path = Path(tempfile.mkdtemp(dir="/tmp")) / "vs.sock"
    first = ControlServer(sock_path, lambda line: "ok")
    first.start()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            ControlServer(sock_path, lambda line: "ok").start()
    finally:
        first.stop()
