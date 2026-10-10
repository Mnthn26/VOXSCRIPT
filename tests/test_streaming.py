"""Offline tests for live streaming: VAD segmentation, the worker, and an end-to-end session.

No microphone, Whisper model, or Ollama server is needed. A fake transcriber stands in for faster-whisper.
"""

from __future__ import annotations

import numpy as np
import wave

from voxscript.cli import build_parser
from voxscript.config import Settings
from voxscript.streaming import (
    LiveSession,
    StreamConfig,
    TranscriptionWorker,
    Utterance,
    UtteranceSegmenter,
    VadEvent,
)
from voxscript.transcriber import Segment

SR = 16000
BLOCK = 480  # 30 ms at 16 kHz


def _tone(seconds: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _blocks(signal: np.ndarray) -> list[np.ndarray]:
    return [signal[i:i + BLOCK] for i in range(0, len(signal) - BLOCK + 1, BLOCK)]


class FakeTranscriber:
    """Stands in for faster-whisper: reports the audio length as a single segment."""

    def load(self):  # noqa: ANN201
        return None

    def transcribe_samples(self, samples, language=None, beam_size=5, vad=True):  # noqa: ANN001
        return [Segment(0.0, len(samples) / SR, "hello world")], "en", 0.95


def test_silence_produces_no_events():
    seg = UtteranceSegmenter(StreamConfig())
    rng = np.random.default_rng(0)
    events = []
    for b in _blocks(rng.normal(0, 0.001, SR * 3).astype(np.float32)):
        events += seg.process(b)
    assert events == []


def test_single_phrase_becomes_one_final_with_partials():
    cfg = StreamConfig(partial_interval_s=0.5)
    seg = UtteranceSegmenter(cfg)
    # 0.48 s silence, 1.5 s tone, 1.2 s silence
    signal = np.concatenate([np.zeros(int(0.48 * SR), np.float32), _tone(1.5), np.zeros(int(1.2 * SR), np.float32)])
    events: list[VadEvent] = []
    for b in _blocks(signal):
        events += seg.process(b)

    partials = [e for e in events if e.kind == "partial"]
    finals = [e for e in events if e.kind == "final"]
    assert len(partials) >= 2
    assert len(finals) == 1
    final = finals[0].utterance
    # Pre-roll (8 blocks = 0.24 s) is included, so the phrase starts slightly before the speech.
    assert abs(final.start_time - 0.24) < 0.031
    # 8 pre-roll + 50 speech + 20 trailing-silence blocks (600 ms) = 78 blocks.
    assert len(final.samples) == 78 * BLOCK


def test_two_phrases_separated_by_pause():
    seg = UtteranceSegmenter(StreamConfig())
    signal = np.concatenate([_tone(1.0), np.zeros(int(1.0 * SR), np.float32), _tone(1.0), np.zeros(int(1.0 * SR), np.float32)])
    finals = []
    for b in _blocks(signal):
        finals += [e.utterance for e in seg.process(b) if e.kind == "final"]
    assert [f.seq for f in finals] == [0, 1]
    assert finals[1].start_time > finals[0].start_time + 1.5


def test_short_click_is_discarded():
    seg = UtteranceSegmenter(StreamConfig())
    signal = np.concatenate([_tone(0.09), np.zeros(int(1.0 * SR), np.float32)])  # 90 ms < 250 ms minimum
    events = []
    for b in _blocks(signal):
        events += seg.process(b)
    assert events == []


def test_long_monologue_is_split():
    cfg = StreamConfig(max_utterance_s=2.0, partial_interval_s=100.0)
    seg = UtteranceSegmenter(cfg)
    finals = []
    for b in _blocks(_tone(5.0)):
        finals += [e.utterance for e in seg.process(b) if e.kind == "final"]
    assert len(finals) >= 2


def test_flush_closes_phrase_still_in_progress():
    seg = UtteranceSegmenter(StreamConfig())
    for b in _blocks(_tone(1.0)):
        seg.process(b)
    events = seg.flush()
    assert len(events) == 1 and events[0].kind == "final"


def test_worker_commits_finals_and_ignores_stale_partials():
    worker = TranscriptionWorker(FakeTranscriber(), language="en")
    worker.start()
    u0 = Utterance(seq=0, start_time=1.0, samples=np.zeros(SR, np.float32))
    worker.submit(VadEvent("partial", u0))
    worker.submit(VadEvent("final", u0))
    # A late partial for an already committed phrase must not reappear.
    worker.submit(VadEvent("partial", u0))
    worker.close()
    worker.join()

    committed, partial = worker.snapshot()
    assert committed == "hello world"
    assert partial == ""
    segments, language, prob = worker.result()
    assert segments[0].start == 1.0  # offset by the utterance start time
    assert language == "en" and prob == 0.95


def test_live_session_end_to_end_with_fake_microphone(tmp_path, monkeypatch):
    """Feeds a synthetic 3-second recording through LiveSession as if it came from a mic."""

    class FakeInputStream:
        def __init__(self, samplerate, channels, dtype, blocksize, device, callback):  # noqa: ANN001
            self.blocksize = blocksize
            self.callback = callback

        def start(self):
            audio = np.zeros(SR * 3, dtype=np.float32)
            audio[SR:2 * SR] = _tone(1.0)
            pcm = (audio * 32767).astype(np.int16)
            for i in range(0, len(pcm), self.blocksize):
                block = pcm[i:i + self.blocksize].reshape(-1, 1)
                self.callback(block, len(block), None, None)

        def stop(self):
            pass

        def close(self):
            pass

    class FakeSoundDevice:
        InputStream = FakeInputStream

    class NeverPressed:
        def start(self):
            pass

        def pressed(self):
            return False

    monkeypatch.setattr("voxscript.streaming.load_sounddevice", lambda: FakeSoundDevice)
    monkeypatch.setattr("voxscript.streaming.EnterKeyWatcher", NeverPressed)

    out = tmp_path / "session.wav"
    session = LiveSession(
        settings=Settings(whisper_model="base.en"),
        cfg=StreamConfig(),
        transcriber=FakeTranscriber(),  # type: ignore[arg-type]
        device=None,
        language=None,
        audio_out=out,
        max_seconds=0.3,  # the fake device delivers all audio at once, so this just ends the loop
    )
    transcript = session.run()

    assert transcript.text == "hello world"
    assert transcript.language == "en"
    with wave.open(str(out)) as wav:
        assert wav.getframerate() == SR
        assert wav.getnframes() == SR * 3  # the full session is saved, not only the speech


def test_cli_live_flags_parse():
    args = build_parser(Settings()).parse_args(["--live", "--silence-ms", "500", "--vad-threshold", "0.03"])
    assert args.live is True
    assert args.silence_ms == 500
    assert args.vad_threshold == 0.03


def test_live_and_file_are_mutually_exclusive():
    import pytest

    with pytest.raises(SystemExit):
        build_parser(Settings()).parse_args(["--live", "--file", "x.wav"])
