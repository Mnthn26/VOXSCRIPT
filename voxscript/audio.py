"""Microphone capture: interactive start/stop recording and WAV output."""

from __future__ import annotations

import queue
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from voxscript.config import Settings


class AudioError(RuntimeError):
    """Raised when audio hardware or capture fails."""


def list_input_devices() -> list[tuple[int, str]]:
    """Return (index, name) for every device that can record audio."""
    sd = _sounddevice()
    try:
        devices = sd.query_devices()
    except Exception as exc:  # PortAudio missing or no audio backend
        raise AudioError(f"Cannot query audio devices: {exc}") from exc
    return [(i, d["name"]) for i, d in enumerate(devices) if d["max_input_channels"] > 0]


def resolve_device(device: str | None) -> int | None:
    """Map a device spec (index or name substring) to a PortAudio device index."""
    if device is None:
        return None
    if device.isdigit():
        return int(device)
    matches = [i for i, name in list_input_devices() if device.lower() in name.lower()]
    if not matches:
        raise AudioError(f"No input device matching {device!r}. Run --list-devices.")
    return matches[0]


def _sounddevice():  # noqa: ANN202
    """Import sounddevice lazily so the rest of the app works without PortAudio."""
    try:
        import sounddevice as sd
    except OSError as exc:
        raise AudioError(
            "PortAudio library not found. Install it with 'sudo pacman -S portaudio' "
            "(Arch) or 'sudo apt install libportaudio2' (Debian/Ubuntu)."
        ) from exc
    return sd


class Recorder:
    """Captures microphone audio into memory from a background PortAudio stream."""

    def __init__(self, settings: Settings, device: int | None = None) -> None:
        self._settings = settings
        self._device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None
        self.started_at: float | None = None

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        self._queue.put(indata.copy())

    def start(self) -> None:
        sd = _sounddevice()
        try:
            self._stream = sd.InputStream(
                samplerate=self._settings.sample_rate,
                channels=1,
                dtype="int16",
                device=self._device,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            raise AudioError(
                f"Could not open microphone: {exc}. Check --device and that your user is in the 'audio' group."
            ) from exc
        self.started_at = time.monotonic()

    def elapsed(self) -> float:
        return 0.0 if self.started_at is None else time.monotonic() - self.started_at

    def stop(self, output_path: Path | None = None) -> Path:
        """Stop the stream and write captured audio to a 16-bit mono WAV file."""
        if self._stream is None:
            raise AudioError("Recorder was not started.")
        self._stream.stop()
        self._stream.close()
        self._stream = None

        chunks: list[np.ndarray] = []
        while not self._queue.empty():
            chunks.append(self._queue.get_nowait())
        if not chunks:
            raise AudioError("No audio was captured.")

        samples = np.concatenate(chunks).reshape(-1)
        if output_path is None:
            output_path = Path(f"recording-{datetime.now():%Y%m%d-%H%M%S}.wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # int16
            wav.setframerate(self._settings.sample_rate)
            wav.writeframes(samples.tobytes())
        return output_path


class EnterKeyWatcher:
    """Waits for the ENTER key on stdin in a daemon thread so the UI can keep refreshing."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._thread = threading.Thread(target=self._wait, daemon=True)

    def _wait(self) -> None:
        try:
            input()
        except EOFError:
            pass
        self._event.set()

    def start(self) -> None:
        self._thread.start()

    def pressed(self) -> bool:
        return self._event.is_set()
