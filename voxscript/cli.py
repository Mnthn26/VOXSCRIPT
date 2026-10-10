"""Command-line entry point: record or load audio, transcribe, summarize, and export."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from voxscript import ui
from voxscript.audio import AudioError, EnterKeyWatcher, Recorder, list_input_devices, resolve_device
from voxscript.config import Settings
from voxscript.report import build_report, write_output
from voxscript.summarizer import OllamaClient, SummarizerError
from voxscript.transcriber import Transcriber, TranscriptionError

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_CANCELLED = 130


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voxscript",
        description="Record or transcribe audio offline with faster-whisper, then summarize with Ollama.",
        epilog=(
            "examples:\n"
            "  voxscript --record --output notes.md\n"
            "  voxscript --file meeting.mp3 --output meeting.json\n"
            "  voxscript --file memo.wav --no-summary\n"
            "  voxscript --list-devices"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("-r", "--record", action="store_true", help="record from the microphone (ENTER starts/stops)")
    src.add_argument("-f", "--file", metavar="PATH", help="transcribe an existing audio file (wav, mp3, m4a, flac, ...)")
    src.add_argument("--list-devices", action="store_true", help="list microphone input devices and exit")

    p.add_argument("-o", "--output", metavar="PATH", help="write report to PATH (.md/.txt = Markdown, .json = JSON)")
    p.add_argument("--audio-out", metavar="PATH", help="where to save recordings (default: ./recordings/<timestamp>.wav)")
    p.add_argument("--max-seconds", type=float, default=0, metavar="N", help="auto-stop recording after N seconds (0 = manual stop)")
    p.add_argument("--mic", metavar="DEVICE", help="input device index or name substring (see --list-devices)")

    p.add_argument("--whisper-model", default=settings.whisper_model, help="faster-whisper model size or path (default: %(default)s)")
    p.add_argument("--whisper-device", default=settings.whisper_device, choices=["cpu", "cuda", "auto"], help="inference device (default: %(default)s)")
    p.add_argument("--compute-type", default=settings.whisper_compute_type, help="CTranslate2 compute type, e.g. int8, float16 (default: %(default)s)")
    p.add_argument("--language", default=None, metavar="CODE", help="spoken language code, e.g. en (default: auto-detect)")

    p.add_argument("--no-summary", action="store_true", help="skip the Ollama summarization step")
    p.add_argument("--ollama-host", default=settings.ollama_host, help="Ollama base URL (default: %(default)s)")
    p.add_argument("--ollama-model", default=settings.ollama_model, help="Ollama model name (default: %(default)s)")
    p.add_argument("--timeout", type=float, default=settings.ollama_timeout, metavar="SEC", help="Ollama request timeout (default: %(default)s)")
    return p


def record_interactively(settings: Settings, device: int | None, max_seconds: float, audio_out: Path) -> Path:
    """Wait for ENTER to start, show a live timer, and stop on ENTER (or after max_seconds)."""
    ui.console.print("[bold]Press ENTER[/] to start recording  [dim](Ctrl+C to quit)[/]")
    start_key = EnterKeyWatcher()
    start_key.start()
    while not start_key.pressed():
        time.sleep(0.05)

    recorder = Recorder(settings, device)
    recorder.start()
    stop_key = EnterKeyWatcher()
    stop_key.start()

    def panel() -> Panel:
        elapsed = recorder.elapsed()
        limit = f" / {max_seconds:.0f}s" if max_seconds else ""
        body = Text.assemble(
            ("● REC  ", "bold red"),
            (f"{int(elapsed // 60):02d}:{elapsed % 60:04.1f}{limit}", "bold white"),
            ("\n\nPress ENTER to stop", "dim"),
        )
        return Panel(body, border_style="red", title="Recording", width=44)

    with Live(panel(), console=ui.console, refresh_per_second=10, transient=True) as live:
        while not stop_key.pressed():
            if max_seconds and recorder.elapsed() >= max_seconds:
                break
            live.update(panel())
            time.sleep(0.1)

    path = recorder.stop(audio_out)
    ui.success(f"Recording saved to [bold]{path}[/] ({recorder.elapsed():.1f}s)")
    return path


def run(args: argparse.Namespace, settings: Settings) -> int:
    # 1. Obtain audio: either record it or use the given file.
    if args.record:
        device = resolve_device(args.mic)
        audio_out = Path(args.audio_out) if args.audio_out else Path("recordings") / f"recording-{datetime.now():%Y%m%d-%H%M%S}.wav"
        audio_path = record_interactively(settings, device, args.max_seconds, audio_out)
        source = "microphone"
    else:
        audio_path = Path(args.file).expanduser()
        if not audio_path.is_file():
            ui.error(f"File not found: {audio_path}")
            return EXIT_ERROR
        source = str(audio_path)

    # 2. Transcribe locally.
    settings.whisper_model = args.whisper_model
    settings.whisper_device = args.whisper_device
    settings.whisper_compute_type = args.compute_type
    transcriber = Transcriber(settings)
    with ui.console.status(f"[cyan]Transcribing with faster-whisper ({settings.whisper_model})...[/]", spinner="dots"):
        transcript = transcriber.transcribe(audio_path, language=args.language)
    if not transcript.text:
        ui.warn("No speech detected in the audio.")
    else:
        ui.show_transcript(transcript)

    # 3. Summarize with Ollama (non-fatal: the transcript is still saved on failure).
    summary: str | None = None
    if not args.no_summary and transcript.text:
        settings.ollama_host = args.ollama_host
        settings.ollama_model = args.ollama_model
        settings.ollama_timeout = args.timeout
        client = OllamaClient(settings)
        try:
            with ui.console.status(f"[magenta]Summarizing with Ollama ({settings.ollama_model})...[/]", spinner="dots"):
                client.check()
                summary = client.summarize(transcript.text)
            ui.show_summary(summary)
        except SummarizerError as exc:
            ui.warn(f"Summary skipped: {exc}")

    # 4. Export.
    if args.output:
        out = Path(args.output).expanduser()
        models = {
            "whisper": settings.whisper_model,
            "ollama": None if summary is None else settings.ollama_model,
        }
        report = build_report(transcript, summary, source, models)
        write_output(out, report, transcript)
        ui.success(f"Report written to [bold]{out}[/]")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    settings = Settings()
    parser = build_parser(settings)
    args = parser.parse_args(argv)

    if args.list_devices:
        try:
            ui.show_devices(list_input_devices())
        except AudioError as exc:
            ui.error(str(exc))
            return EXIT_ERROR
        return EXIT_OK
    if not (args.record or args.file):
        parser.error("choose an input: --record or --file PATH")

    ui.banner()
    try:
        return run(args, settings)
    except (AudioError, TranscriptionError) as exc:
        ui.error(str(exc))
        return EXIT_ERROR
    except KeyboardInterrupt:
        ui.warn("Cancelled.")
        return EXIT_CANCELLED
