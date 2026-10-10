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
tests/test_core.py
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

## Tests

```bash
pytest -q
```

The tests run offline. They use a local fake Ollama server and do not need a microphone or Whisper model.
