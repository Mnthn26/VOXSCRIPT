"""Rich-based terminal rendering helpers."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from voxscript import __version__
from voxscript.transcriber import Transcript, fmt_time

console = Console()


def banner() -> None:
    title = Text.assemble(
        ("VOXSCRIPT", "bold cyan"),
        (f" v{__version__}", "dim"),
        ("  ·  offline STT · local AI summaries", "italic"),
    )
    console.print(Panel(title, box=box.ROUNDED, expand=False, padding=(0, 2)))


def error(message: str) -> None:
    console.print(f"[bold red]✗ Error:[/] {message}")


def warn(message: str) -> None:
    console.print(f"[bold yellow]![/] {message}")


def success(message: str) -> None:
    console.print(f"[bold green]✓[/] {message}")


def info(message: str) -> None:
    console.print(f"[cyan]•[/] {message}")


def show_devices(devices: list[tuple[int, str]]) -> None:
    table = Table(title="Input devices", box=box.SIMPLE_HEAVY, header_style="bold magenta")
    table.add_column("Index", justify="right")
    table.add_column("Name")
    for idx, name in devices:
        table.add_row(str(idx), name)
    console.print(table)


def show_transcript(transcript: Transcript) -> None:
    table = Table(title="Transcript", box=box.MINIMAL, show_lines=False, expand=True)
    table.add_column("Time", style="dim", no_wrap=True)
    table.add_column("Text")
    for seg in transcript.segments:
        table.add_row(fmt_time(seg.start), seg.text.strip())
    console.print(table)
    console.print(
        f"[dim]Language: {transcript.language} ({transcript.language_probability:.0%}) · "
        f"Duration: {transcript.duration:.1f}s · Segments: {len(transcript.segments)}[/]"
    )


def show_summary(markdown_text: str) -> None:
    console.print(
        Panel(Markdown(markdown_text), title="[bold green]AI Summary & Action Items", border_style="green", padding=(1, 2))
    )
