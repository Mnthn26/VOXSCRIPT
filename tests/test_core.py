"""Offline tests: no microphone, no Whisper download, no real Ollama required."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from voxscript.config import Settings
from voxscript.report import build_report, write_output
from voxscript.summarizer import OllamaClient, SummarizerError, chunk_text
from voxscript.transcriber import Segment, Transcript, fmt_time


class FakeOllama(BaseHTTPRequestHandler):
    models = ["llama3.2:latest"]
    calls: list[dict] = []

    def log_message(self, *args):  # silence test output
        pass

    def do_GET(self):  # noqa: N802
        body = json.dumps({"models": [{"name": m} for m in self.models]}).encode()
        self._send(200, body)

    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        FakeOllama.calls.append(payload)
        content = "## Summary\n- Point one\n\n## Action Items\n- [ ] Ship it (Owner: Alex, Due: Friday)"
        self._send(200, json.dumps({"message": {"role": "assistant", "content": content}}).encode())

    def _send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def ollama_server():
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    FakeOllama.calls = []
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_fmt_time():
    assert fmt_time(65) == "01:05"
    assert fmt_time(3725) == "01:02:05"


def test_chunk_text_short_and_long():
    assert chunk_text("Hello there.", 100) == ["Hello there."]
    text = " ".join(["This is a sentence."] * 200)
    chunks = chunk_text(text, 300)
    assert len(chunks) > 1
    assert all(len(c) <= 320 for c in chunks)


def test_summarize_and_check(ollama_server):
    settings = Settings(ollama_host=ollama_server, ollama_model="llama3.2")
    client = OllamaClient(settings)
    client.check()  # model installed as 'llama3.2:latest'
    result = client.summarize("We agreed to ship on Friday.")
    assert "## Action Items" in result
    assert FakeOllama.calls[-1]["model"] == "llama3.2"
    assert FakeOllama.calls[-1]["stream"] is False


def test_long_transcript_is_chunked_then_merged(ollama_server):
    settings = Settings(ollama_host=ollama_server, ollama_model="llama3.2", summary_chunk_chars=500)
    OllamaClient(settings).summarize(" ".join(["Budget review content."] * 300))
    assert len(FakeOllama.calls) >= 3  # >=2 partial summaries + 1 merge


def test_missing_model_is_reported(ollama_server):
    settings = Settings(ollama_host=ollama_server, ollama_model="mistral")
    with pytest.raises(SummarizerError, match="ollama pull mistral"):
        OllamaClient(settings).check()


def test_unreachable_ollama():
    settings = Settings(ollama_host="http://127.0.0.1:1", ollama_model="llama3.2", ollama_timeout=2)
    with pytest.raises(SummarizerError, match="Cannot reach Ollama"):
        OllamaClient(settings).summarize("hi")


def test_write_markdown_and_json(tmp_path):
    t = Transcript(segments=[Segment(0.0, 2.0, " Hello team."), Segment(65.0, 67.0, "Ship it.")],
                   language="en", language_probability=0.99, duration=67.0)
    summary = "## Summary\n- Hello"
    report = build_report(t, summary, "meeting.wav", {"whisper": "base.en", "ollama": "llama3.2"})

    md = tmp_path / "out.md"
    write_output(md, report, t)
    text = md.read_text()
    assert "## Summary" in text and "[01:05] Ship it." in text

    js = tmp_path / "out.json"
    write_output(js, report, t)
    data = json.loads(js.read_text())
    assert data["transcript"]["text"] == "Hello team. Ship it."
    assert data["summary"] == summary
