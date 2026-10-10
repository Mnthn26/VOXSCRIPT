"""Runtime configuration with environment-variable defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


@dataclass
class Settings:
    """All tunable options. Defaults can be overridden via environment variables."""

    whisper_model: str = field(default_factory=lambda: _env("VOXSCRIPT_WHISPER_MODEL", "base.en"))
    whisper_device: str = field(default_factory=lambda: _env("VOXSCRIPT_WHISPER_DEVICE", "cpu"))
    whisper_compute_type: str = field(default_factory=lambda: _env("VOXSCRIPT_COMPUTE_TYPE", "int8"))
    whisper_cache_dir: str | None = field(default_factory=lambda: os.environ.get("VOXSCRIPT_MODEL_DIR") or None)
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("VOXSCRIPT_OLLAMA_MODEL", "llama3.2"))
    ollama_timeout: float = 300.0
    sample_rate: int = 16000  # Whisper expects 16 kHz mono audio.
    # Transcripts longer than this (characters) are summarized in chunks, then merged.
    summary_chunk_chars: int = 12000
