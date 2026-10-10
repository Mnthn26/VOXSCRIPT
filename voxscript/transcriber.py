"""Offline speech-to-text using faster-whisper (CTranslate2 backend)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from voxscript.config import Settings


class TranscriptionError(RuntimeError):
    """Raised when the audio file cannot be transcribed."""


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)
    language: str = ""
    language_probability: float = 0.0
    duration: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()

    def timestamped(self) -> str:
        """Return the transcript with [mm:ss] prefixes per segment."""
        return "\n".join(f"[{fmt_time(s.start)}] {s.text.strip()}" for s in self.segments)


def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Transcriber:
    """Lazily loads a faster-whisper model and transcribes audio files."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None

    def _load(self):  # noqa: ANN202
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover
                raise TranscriptionError("faster-whisper is not installed. Run: pip install -r requirements.txt") from exc
            try:
                self._model = WhisperModel(
                    self._settings.whisper_model,
                    device=self._settings.whisper_device,
                    compute_type=self._settings.whisper_compute_type,
                    download_root=self._settings.whisper_cache_dir,
                )
            except Exception as exc:
                raise TranscriptionError(
                    f"Failed to load Whisper model {self._settings.whisper_model!r}: {exc}. "
                    "Models download once from Hugging Face; after that VOXSCRIPT runs offline."
                ) from exc
        return self._model

    def transcribe(self, audio_path: Path, language: str | None = None, vad: bool = True) -> Transcript:
        if not audio_path.is_file():
            raise TranscriptionError(f"Audio file not found: {audio_path}")
        model = self._load()
        try:
            segments, info = model.transcribe(
                str(audio_path),
                language=language,
                beam_size=5,
                vad_filter=vad,  # skip silence; reduces hallucinations on quiet audio
            )
            result = Transcript(
                segments=[Segment(s.start, s.end, s.text) for s in segments],  # generator is consumed here
                language=info.language,
                language_probability=info.language_probability,
                duration=info.duration,
            )
        except Exception as exc:
            raise TranscriptionError(f"Transcription failed for {audio_path}: {exc}") from exc
        return result
