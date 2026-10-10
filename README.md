# VOXSCRIPT

Linux terminal app that records (or loads) voice audio, transcribes it **offline** with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), and sends the transcript to a
local [Ollama](https://ollama.com) model for bulleted summaries and action items.

## Features

- **Microphone capture**: ENTER starts recording, ENTER stops it, live timer, optional `--max-seconds`.
- **Local STT**: faster-whisper with VAD filtering; works offline after the first model download.
- **AI summaries**: Ollama `/api/chat` → Markdown `## Summary` bullets and `## Action Items` checkboxes. Long transcripts are chunked and merged.
- **Rich CLI**: tables, spinners, and panels; `--record`, `--file`, `--output` (`.md`/`.txt` or `.json`).

## Project layout

```
voxscript/
  __main__.py     python -m voxscript
  cli.py          argparse + orchestration
  config.py       settings and environment defaults
  audio.py        microphone capture and WAV writing (sounddevice)
  transcriber.py  faster-whisper wrapper
  summarizer.py   Ollama client and chunking
  report.py       Markdown/JSON report generation
  ui.py           rich rendering helpers
  streaming.py    real-time VAD + incremental transcription + live view
  dictate/        system-wide daemon: controller, hotkey, injector, control socket, voxscript-ctl
deploy/           systemd user service
tests/test_core.py
tests/test_streaming.py
tests/test_dictate.py
```

## System dependencies

### Arch Linux (pacman)

```bash
sudo pacman -S --needed python python-pip portaudio alsa-utils
# Optional, for GPU inference later:
# sudo pacman -S --needed cuda cudnn
```

### Debian / Ubuntu (apt), for reference

```bash
sudo apt install python3-venv libportaudio2 alsa-utils
```

`sounddevice` needs the PortAudio shared library. faster-whisper decodes mp3/m4a/flac through
its bundled PyAV (FFmpeg libraries), so a separate `ffmpeg` install is not required.

Your user must be able to access the microphone. On PipeWire/PulseAudio desktops this works
out of the box. On plain ALSA, add yourself to the `audio` group if needed:
`sudo usermod -aG audio "$USER"` and log out and back in.

### Ollama

Install Ollama with the official script, or on Arch with `sudo pacman -S ollama`:

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama      # or run `ollama serve` in another terminal
ollama pull llama3.2
```

## Setup

```bash
cd VOXSCRIPT
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                      # optional: installs the `voxscript` command

# Verify microphones
voxscript --list-devices
```

The first run downloads the Whisper model (`base.en`, about 150 MB) from Hugging Face. Pre-fetch
it with `python -c "from faster_whisper import WhisperModel; WhisperModel('base.en')"`.
After that the transcription step needs no network.

## Usage

```bash
# Record from the mic: ENTER to start, ENTER to stop, then transcribe and summarize
voxscript --record --output notes.md

# Record with an auto-stop after 5 minutes, using a specific mic
voxscript --record --max-seconds 300 --mic "USB" --output standup.json

# Transcribe an existing file, with a larger model and a fixed language
voxscript --file meeting.m4a --whisper-model small --language en -o meeting.md

# Transcript only, no Ollama
voxscript --file memo.wav --no-summary

# Use a remote Ollama instance and another model
voxscript -f call.mp3 --ollama-host http://192.168.1.20:11434 --ollama-model qwen2.5:7b -o call.md
```

Environment variables (optional): `VOXSCRIPT_WHISPER_MODEL`, `VOXSCRIPT_WHISPER_DEVICE`,
`VOXSCRIPT_COMPUTE_TYPE`, `VOXSCRIPT_MODEL_DIR`, `VOXSCRIPT_OLLAMA_MODEL`, `OLLAMA_HOST`.

Exit codes: `0` success, `1` runtime error, `2` usage error, `130` cancelled with Ctrl+C.

## Live streaming

```bash
voxscript --live --output notes.md
```

Speak naturally. Text appears in the terminal as you talk. Italic text is provisional and is replaced once a pause closes the phrase. Press ENTER or Ctrl+C to stop. The full transcript is then sent to Ollama, and the report and the full recording (`./recordings/`) are saved.

Tuning: `--silence-ms` (pause that ends a phrase, default 600), `--partial-interval` (live refresh, default 1.0 s), `--vad-threshold` (raise it in noisy rooms), `--max-utterance` (forced phrase split, default 15 s). For faster live updates on CPU, try `--whisper-model tiny.en`.

## Universal dictation daemon (system-wide)

`voxscript-daemon` runs in the background. Press a hotkey, speak, press it again, and the transcribed text is copied to the clipboard and pasted into whichever window has focus.

```bash
voxscript-daemon --check          # shows which tools are installed and what to install
voxscript-daemon                  # run in the foreground to see the logs
voxscript-ctl toggle              # start or stop dictation from any terminal
voxscript-ctl status              # idle | recording | busy
```

### Which trigger works where

| Session | Global hotkey (`pynput`, `--hotkey`) | Compositor binding -> `voxscript-ctl toggle` |
|---|---|---|
| X11 (i3, Xfce, ...) | yes | yes |
| Wayland, X11 apps (XWayland) | yes, only while an X app is focused | yes |
| Wayland, native apps (Hyprland, GNOME, KDE) | no, the compositor hides keys from `pynput` | **yes, use this** |

On Wayland, bind a compositor shortcut to `voxscript-ctl toggle`. Examples:

```conf
# Hyprland (~/.config/hypr/hyprland.conf)
bind = SUPER, D, exec, /home/YOU/VOXSCRIPT/.venv/bin/voxscript-ctl toggle
# GNOME: Settings -> Keyboard -> Custom Shortcuts -> command: /home/YOU/VOXSCRIPT/.venv/bin/voxscript-ctl toggle
```

Ctrl+Space is the default hotkey as requested, but in some apps it also types a character or switches the input method. Try `--hotkey "ctrl+alt+d"` if that happens.

### Arch Linux packages for injection

X11:
```bash
sudo pacman -S --needed xdotool xclip libnotify
```

Wayland (Hyprland, GNOME, KDE, Sway):
```bash
sudo pacman -S --needed wl-clipboard ydotool libnotify

# ydotool writes to /dev/uinput, so your user needs access to it
sudo usermod -aG input "$USER"
echo 'KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"' | sudo tee /etc/udev/rules.d/60-uinput.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo modprobe uinput

# Start the ydotool daemon. If your system has a user unit for it, use that:
systemctl --user list-unit-files | grep -i ydotool
systemctl --user enable --now ydotool.service   # only if the unit above exists
# otherwise run `ydotoold &` in your session startup, e.g. Hyprland exec-once
```

Log out and back in after changing group membership.

### Run as a background service

```bash
mkdir -p ~/.config/systemd/user
cp deploy/voxscript-dictate.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now voxscript-dictate.service

journalctl --user -u voxscript-dictate -f     # live logs: LISTENING, TRANSCRIBING, INJECTING
systemctl --user restart voxscript-dictate    # after an update
```

The user service needs the display variables. On Hyprland, add this to your config:
```conf
exec-once = dbus-update-activation-environment --systemd WAYLAND_DISPLAY DISPLAY XDG_CURRENT_DESKTOP XDG_SESSION_TYPE
```
On GNOME and KDE, the session manager already provides them.

### Tuning

- `--whisper-model tiny.en` or `base.en` for the fastest response. `small.en` is more accurate on a GPU or a fast CPU.
- `--paste-combo ctrl+shift+v` for terminal emulators.
- `--inject-mode type` types characters directly instead of using the clipboard, which is slower but does not touch the clipboard.
- `--max-seconds 120` stops a forgotten recording automatically.

Privacy: the daemon logs a preview of each transcription, and it only opens the microphone while you are recording.

## Tests

```bash
pytest -q
```

The tests run offline. They use a local fake Ollama server and do not need a microphone or Whisper model.
