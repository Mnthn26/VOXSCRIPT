"""Build and write the final report (Markdown or JSON, chosen by file extension)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from voxscript.transcriber import Transcript


def build_report(
    transcript: Transcript,
    summary: str | None,
    source: str,
    model_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": source,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "language": transcript.language,
        "duration_seconds": round(transcript.duration, 2),
        "models": model_info,
        "summary": summary,
        "transcript": {
            "text": transcript.text,
            "segments": [
                {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
                for s in transcript.segments
            ],
        },
    }


def render_markdown(report: dict[str, Any], transcript: Transcript) -> str:
    lines = [
        "# VOXSCRIPT Report",
        "",
        f"- **Source:** `{report['source']}`",
        f"- **Created:** {report['created_at']}",
        f"- **Language:** {report['language']}",
        f"- **Duration:** {report['duration_seconds']} s",
        "",
    ]
    if report["summary"]:
        lines += [report["summary"].strip(), ""]
    lines += ["## Transcript", "", "```text", transcript.timestamped(), "```", ""]
    return "\n".join(lines)


def write_output(path: Path, report: dict[str, Any], transcript: Transcript) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        path.write_text(render_markdown(report, transcript), encoding="utf-8")
