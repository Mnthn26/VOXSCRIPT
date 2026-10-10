"""voxscript-ctl: send a command to the running dictation daemon.

Bind this to a key in your compositor or desktop, for example:
    bind = SUPER, D, exec, voxscript-ctl toggle          # Hyprland
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voxscript.dictate.ipc import default_socket_path, send_command


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="voxscript-ctl", description="Control the VOXSCRIPT dictation daemon.")
    p.add_argument("command", choices=["toggle", "start", "stop", "status", "ping", "quit"],
                   help="toggle: start listening, or stop and paste if already listening")
    p.add_argument("--socket", type=Path, default=default_socket_path(), help="control socket path")
    args = p.parse_args(argv)

    try:
        reply = send_command(args.socket, args.command)
    except (FileNotFoundError, ConnectionRefusedError):
        print(f"voxscript-ctl: daemon is not running (no socket at {args.socket}).", file=sys.stderr)
        print("Start it with: systemctl --user start voxscript-dictate", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"voxscript-ctl: could not reach the daemon: {exc}", file=sys.stderr)
        return 1

    print(reply)
    return 0 if reply.startswith("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
