"""Summaries and action items via the Ollama HTTP API (http://localhost:11434 by default)."""

from __future__ import annotations

import requests

from voxscript.config import Settings

SYSTEM_PROMPT = """You are an expert meeting and voice-note analyst.
Read the transcript and respond in Markdown using exactly these two sections:

## Summary
- 3 to 8 concise bullet points covering the main topics, facts and conclusions.

## Action Items
- [ ] <task> (Owner: <name or "Unassigned">, Due: <date or "Not specified">)

If there are no action items, write "- None identified." under Action Items.
Do not invent facts that are not in the transcript."""

MERGE_PROMPT = """The following are summaries of consecutive parts of one long transcript.
Merge them into one final response with exactly the same two Markdown sections
("## Summary" with concise bullets, and "## Action Items" with checkbox bullets).
Remove duplicates and do not invent facts."""


class SummarizerError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error."""


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text on paragraph/sentence-ish boundaries into chunks of at most max_chars."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current = ""
    for piece in text.replace("\n", " ").split(". "):
        piece = piece if piece.endswith(".") else piece + "."
        if len(current) + len(piece) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = ""
        current += piece + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base = settings.ollama_host.rstrip("/")
        self._session = requests.Session()

    def _post_chat(self, system: str, user: str) -> str:
        payload = {
            "model": self._settings.ollama_model,
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 8192},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            resp = self._session.post(
                f"{self._base}/api/chat", json=payload, timeout=self._settings.ollama_timeout
            )
        except requests.ConnectionError as exc:
            raise SummarizerError(
                f"Cannot reach Ollama at {self._base}. Start it with 'ollama serve' "
                "or pass --no-summary."
            ) from exc
        except requests.RequestException as exc:
            raise SummarizerError(f"Ollama request failed: {exc}") from exc
        if resp.status_code != 200:
            raise SummarizerError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()["message"]["content"].strip()
        except (ValueError, KeyError) as exc:
            raise SummarizerError("Unexpected response format from Ollama.") from exc

    def check(self) -> None:
        """Verify the server is up and the configured model is installed."""
        try:
            resp = self._session.get(f"{self._base}/api/tags", timeout=5)
            resp.raise_for_status()
            names = {m.get("name", "") for m in resp.json().get("models", [])}
        except requests.RequestException as exc:
            raise SummarizerError(f"Cannot reach Ollama at {self._base}: {exc}") from exc
        model = self._settings.ollama_model
        wanted = {model, f"{model}:latest"} if ":" not in model else {model}
        if not names & wanted:
            raise SummarizerError(
                f"Model {model!r} is not installed. Run: ollama pull {model}"
            )

    def summarize(self, transcript: str) -> str:
        """Summarize a transcript, chunking long inputs and merging partial summaries."""
        chunks = chunk_text(transcript, self._settings.summary_chunk_chars)
        if len(chunks) == 1:
            return self._post_chat(SYSTEM_PROMPT, f"Transcript:\n\n{transcript}")
        partials = [
            self._post_chat(SYSTEM_PROMPT, f"Transcript part {i}/{len(chunks)}:\n\n{c}")
            for i, c in enumerate(chunks, start=1)
        ]
        merged_input = "\n\n---\n\n".join(partials)
        return self._post_chat(MERGE_PROMPT, merged_input)
