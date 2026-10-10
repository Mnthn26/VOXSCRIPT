"""Real-time streaming transcription.

Pipeline (all audio stays on this machine):

    microphone (sounddevice callback, 30 ms blocks)
        -> energy-based voice activity detection   (UtteranceSegmenter, main thread)
        -> faster-whisper on a background thread   (TranscriptionWorker)
        -> rich.live.Live terminal view            (LiveSession, main thread)

While someone is speaking, the current phrase is re-transcribed about every
`partial_interval_s` seconds and shown in italics. When a pause closes the phrase,
a final transcription replaces the italic text and is committed to the transcript.
"""

from __future__ import annotations

import queue
import textwrap
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from voxscript import ui
from voxscript.audio import AudioError, EnterKeyWatcher, load_sounddevice, write_wav
from voxscript.config import Settings
from voxscript.transcriber import Segment, Transcriber, Transcript, TranscriptionError


@dataclass
class StreamConfig:
    sample_rate: int = 16000
    block_ms: int = 30               # audio block size delivered by the callback
    vad_threshold: float = 0.015     # RMS energy (0..1) that counts as speech; raised automatically above room noise
    silence_ms: int = 600            # pause length that closes a phrase
    min_speech_ms: int = 250         # shorter bursts (clicks, coughs) are discarded
    max_utterance_s: float = 15.0    # force a phrase boundary during long monologues
    partial_interval_s: float = 1.0  # how often the live (italic) hypothesis is refreshed
    pre_roll_ms: int = 240           # audio kept before speech onset so first words are not clipped

    @property
    def block_size(self) -> int:
        return self.sample_rate * self.block_ms // 1000

    @property
    def block_s(self) -> float:
        return self.block_ms / 1000


@dataclass
class Utterance:
    seq: int               # increases by one for every committed phrase
    start_time: float      # seconds since the session started
    samples: np.ndarray    # float32 mono in [-1, 1]


@dataclass
class VadEvent:
    kind: str              # "partial" (provisional) or "final" (committed)
    utterance: Utterance


class UtteranceSegmenter:
    """Energy-based voice activity detector that groups audio blocks into phrases.

    Pure logic with no I/O, so it can be unit-tested with synthetic audio.
    """

    def __init__(self, cfg: StreamConfig) -> None:
        self.cfg = cfg
        self._pre_roll: deque[np.ndarray] = deque(maxlen=max(1, cfg.pre_roll_ms // cfg.block_ms))
        self._frames: list[np.ndarray] = []
        self._in_speech = False
        self._speech_blocks = 0
        self._silence_blocks = 0
        self._last_partial_frames = 0
        self._block_index = 0
        self._seq = 0
        self._start_time = 0.0
        self._noise_floor = 0.001
        self.level = 0.0  # RMS of the most recent block, for the level meter

    def process(self, block: np.ndarray) -> list[VadEvent]:
        """Feed one block of float32 samples and return any events it produced."""
        cfg = self.cfg
        rms = float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0
        self.level = rms
        threshold = max(cfg.vad_threshold, self._noise_floor * 3.0)
        is_speech = rms > threshold
        index = self._block_index
        self._block_index += 1
        events: list[VadEvent] = []

        if not self._in_speech:
            if is_speech:
                self._in_speech = True
                self._start_time = (index - len(self._pre_roll)) * cfg.block_s
                self._frames = [*self._pre_roll, block]
                self._pre_roll.clear()
                self._speech_blocks = 1
                self._silence_blocks = 0
                self._last_partial_frames = 0
            else:
                self._noise_floor = 0.98 * self._noise_floor + 0.02 * rms
                self._pre_roll.append(block)
            return events

        self._frames.append(block)
        if is_speech:
            self._speech_blocks += 1
            self._silence_blocks = 0
        else:
            self._silence_blocks += 1

        duration = len(self._frames) * cfg.block_s
        ended_by_silence = self._silence_blocks * cfg.block_ms >= cfg.silence_ms
        ended_by_length = duration >= cfg.max_utterance_s
        if ended_by_silence or ended_by_length:
            events += self._finish_utterance()
        elif (
            self._speech_blocks * cfg.block_ms >= cfg.min_speech_ms
            and (len(self._frames) - self._last_partial_frames) * cfg.block_s >= cfg.partial_interval_s
        ):
            self._last_partial_frames = len(self._frames)
            events.append(VadEvent("partial", self._make_utterance()))
        return events

    def flush(self) -> list[VadEvent]:
        """Close any phrase still in progress (call when the stream stops)."""
        return self._finish_utterance() if self._in_speech else []

    def _make_utterance(self) -> Utterance:
        samples = np.concatenate(self._frames).astype(np.float32, copy=False).reshape(-1)
        return Utterance(self._seq, self._start_time, samples)

    def _finish_utterance(self) -> list[VadEvent]:
        events: list[VadEvent] = []
        if self._speech_blocks * self.cfg.block_ms >= self.cfg.min_speech_ms:
            events.append(VadEvent("final", self._make_utterance()))
            self._seq += 1
        self._frames = []
        self._in_speech = False
        self._speech_blocks = 0
        self._silence_blocks = 0
        self._last_partial_frames = 0
        return events


class TranscriptionWorker:
    """Runs faster-whisper on a background thread so audio capture never waits for it.

    Finals are processed in order. Partials are coalesced: only the newest one is kept.
    """

    def __init__(self, transcriber: Transcriber, language: str | None) -> None:
        self._transcriber = transcriber
        self._language = language
        self._cond = threading.Condition()
        self._finals: deque[Utterance] = deque()
        self._partial: Utterance | None = None
        self._closed = False
        self._lock = threading.Lock()
        self._committed: list[Segment] = []
        self._partial_text = ""
        self._last_final_seq = -1
        self._language_found = ""
        self._language_probability = 0.0
        self.errors: list[str] = []
        self._thread = threading.Thread(target=self._run, name="whisper-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, event: VadEvent) -> None:
        with self._cond:
            if event.kind == "partial":
                self._partial = event.utterance
            else:
                self._finals.append(event.utterance)
            self._cond.notify()

    def close(self) -> None:
        """No more input; the worker exits once its queue is empty."""
        with self._cond:
            self._closed = True
            self._cond.notify()

    def join(self) -> None:
        self._thread.join()

    def snapshot(self) -> tuple[str, str]:
        """Return (committed text, provisional text) for the UI."""
        with self._lock:
            committed = " ".join(s.text.strip() for s in self._committed).strip()
            return committed, self._partial_text

    def result(self) -> tuple[list[Segment], str, float]:
        with self._lock:
            return sorted(self._committed, key=lambda s: s.start), self._language_found, self._language_probability

    def _next_job(self) -> tuple[str, Utterance] | None:
        with self._cond:
            while not self._finals and self._partial is None and not self._closed:
                self._cond.wait(timeout=0.2)
            if self._finals:
                return "final", self._finals.popleft()
            if self._partial is not None:
                job = self._partial
                self._partial = None
                return "partial", job
            return None  # closed and drained

    def _run(self) -> None:
        while (job := self._next_job()) is not None:
            kind, utt = job
            is_partial = kind == "partial"
            try:
                segments, language, probability = self._transcriber.transcribe_samples(
                    utt.samples,
                    language=self._language,
                    beam_size=1 if is_partial else 5,  # greedy decoding keeps the live line responsive
                )
            except TranscriptionError as exc:
                with self._lock:
                    self.errors.append(str(exc))
                continue

            shifted = [Segment(s.start + utt.start_time, s.end + utt.start_time, s.text) for s in segments]
            text = " ".join(s.text.strip() for s in shifted).strip()
            with self._lock:
                if is_partial:
                    # Ignore partials for phrases that were already committed.
                    if utt.seq > self._last_final_seq:
                        self._partial_text = text
                else:
                    self._last_final_seq = utt.seq
                    self._partial_text = ""
                    self._committed.extend(s for s in shifted if s.text.strip())
                    self._language_found = language
                    self._language_probability = probability


class LiveSession:
    """Runs one streaming session: capture, segment, transcribe, and display until stopped."""

    STOP_HINT = "Press ENTER or Ctrl+C to stop"

    def __init__(
        self,
        settings: Settings,
        cfg: StreamConfig,
        transcriber: Transcriber,
        device: int | None,
        language: str | None,
        audio_out: Path,
        max_seconds: float = 0.0,
    ) -> None:
        self._settings = settings
        self._cfg = cfg
        self._transcriber = transcriber
        self._device = device
        self._language = language
        self._audio_out = audio_out
        self._max_seconds = max_seconds
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._blocks: list[np.ndarray] = []  # raw int16 audio, kept so the full session can be saved
        self._started = 0.0

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        # Runs on PortAudio's thread: copy and hand off, nothing heavier.
        self._queue.put_nowait(indata[:, 0].copy())

    def _drain(self, segmenter: UtteranceSegmenter, worker: TranscriptionWorker) -> None:
        while True:
            try:
                raw = self._queue.get_nowait()
            except queue.Empty:
                return
            self._blocks.append(raw)
            for event in segmenter.process(raw.astype(np.float32) / 32768.0):
                worker.submit(event)

    def _render(self, worker: TranscriptionWorker, segmenter: UtteranceSegmenter, status: str) -> Panel:
        committed, partial = worker.snapshot()
        width = max(20, ui.console.width - 8)
        max_lines = max(4, ui.console.height - 10)

        partial_lines = textwrap.wrap(partial, width) if partial else []
        committed_lines = textwrap.wrap(committed, width) if committed else []
        room = max_lines - len(partial_lines)
        committed_lines = committed_lines[-room:] if room > 0 else []  # keep the newest lines visible

        body = Text()
        body.append("\n".join(committed_lines), style="bold white")
        if committed_lines and partial_lines:
            body.append("\n")
        body.append("\n".join(partial_lines), style="italic cyan")
        if not body.plain:
            body = Text("Listening... start speaking.", style="dim")

        filled = int(min(1.0, segmenter.level * 12) * 20)
        meter = Text("█" * filled + "░" * (20 - filled), style="green" if filled < 16 else "yellow")
        minutes, seconds = divmod(time.monotonic() - self._started, 60)
        title = Text.assemble(
            ("● LIVE ", "bold red"),
            (f"{int(minutes):02d}:{seconds:04.1f}", "bold"),
            ("   mic ", "dim"),
            meter,
        )
        return Panel(body, title=title, subtitle=Text(status, style="dim"), border_style="red", padding=(1, 2))

    def _shutdown(self, stream, segmenter: UtteranceSegmenter, worker: TranscriptionWorker) -> None:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        self._drain(segmenter, worker)          # audio captured before the stop
        for event in segmenter.flush():         # phrase still being spoken
            worker.submit(event)
        worker.close()
        worker.join()                           # wait for the last transcription to finish

    def run(self) -> Transcript:
        """Blocks until the user stops the session. Returns the full transcript."""
        sd = load_sounddevice()
        language = self._language or ("en" if self._settings.whisper_model.endswith(".en") else None)
        with ui.console.status("[cyan]Loading Whisper model...", spinner="dots"):
            self._transcriber.load()  # load before the mic opens so no audio is lost

        worker = TranscriptionWorker(self._transcriber, language)
        segmenter = UtteranceSegmenter(self._cfg)
        try:
            stream = sd.InputStream(
                samplerate=self._cfg.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self._cfg.block_size,
                device=self._device,
                callback=self._callback,
            )
        except Exception as exc:
            raise AudioError(f"Could not open microphone: {exc}. Check --mic and audio permissions.") from exc

        stop_key = EnterKeyWatcher()
        worker.start()
        self._started = time.monotonic()
        try:
            stream.start()
        except Exception as exc:
            raise AudioError(f"Could not start microphone stream: {exc}") from exc
        stop_key.start()

        with Live(self._render(worker, segmenter, self.STOP_HINT), console=ui.console,
                  refresh_per_second=8, transient=False) as live:
            try:
                while not stop_key.pressed():
                    self._drain(segmenter, worker)
                    live.update(self._render(worker, segmenter, self.STOP_HINT))
                    if self._max_seconds and time.monotonic() - self._started >= self._max_seconds:
                        break
                    time.sleep(0.05)
            except KeyboardInterrupt:
                pass  # Ctrl+C is a normal way to stop a live session
            finally:
                live.update(self._render(worker, segmenter, "Finishing last phrase..."))
                self._shutdown(stream, segmenter, worker)
            live.update(self._render(worker, segmenter, "Stopped"))

        duration = time.monotonic() - self._started
        for message in worker.errors:
            ui.warn(message)

        segments, found_language, probability = worker.result()
        if self._blocks:
            samples = np.concatenate(self._blocks)
            write_wav(self._audio_out, samples, self._cfg.sample_rate)
            ui.success(f"Audio saved to [bold]{self._audio_out}[/]")
        return Transcript(
            segments=segments,
            language=found_language,
            language_probability=probability,
            duration=duration,
        )
