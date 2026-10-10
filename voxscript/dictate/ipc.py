"""Unix-socket control channel between the daemon and `voxscript-ctl`.

This is what makes Wayland work: a compositor keybinding (Hyprland, GNOME, KDE, Sway)
runs `voxscript-ctl toggle`, which sends a command to the daemon over this socket.
"""

from __future__ import annotations

import os
import socket
import threading
from pathlib import Path
from typing import Callable


def default_socket_path() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return Path(runtime) / "voxscript.sock"
    return Path(f"/tmp/voxscript-{os.getuid()}.sock")


def _is_alive(path: Path) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            s.connect(str(path))
        return True
    except OSError:
        return False


def send_command(path: Path, command: str, timeout: float = 5.0) -> str:
    """Send one command line to the daemon and return its reply."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(str(path))
        s.sendall((command.strip() + "\n").encode("utf-8"))
        chunks: list[bytes] = []
        while True:
            data = s.recv(4096)
            if not data:
                break
            chunks.append(data)
    return b"".join(chunks).decode("utf-8", "replace").strip()


class ControlServer:
    """Accepts short-lived connections; each connection carries one command and gets one reply."""

    def __init__(self, path: Path, handler: Callable[[str], str]) -> None:
        self.path = path
        self._handler = handler
        self._sock: socket.socket | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            if _is_alive(self.path):
                raise RuntimeError(f"Another VOXSCRIPT daemon is already running ({self.path}).")
            self.path.unlink()  # stale socket left by a crash
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.path))
        os.chmod(self.path, 0o600)  # only this user may send commands
        sock.listen(8)
        sock.settimeout(0.5)
        self._sock = sock
        self._running = True
        self._thread = threading.Thread(target=self._serve, name="control-socket", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while self._running:
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(5.0)
            try:
                line = conn.recv(4096).decode("utf-8", "replace").strip()
                reply = self._handler(line)
            except Exception as exc:  # never let a bad client crash the daemon
                reply = f"error: {exc}"
            try:
                conn.sendall((reply + "\n").encode("utf-8"))
            except OSError:
                pass

    def stop(self) -> None:
        self._running = False
        if self._sock is not None:
            self._sock.close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
