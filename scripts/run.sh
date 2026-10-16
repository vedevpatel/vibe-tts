#!/usr/bin/env bash
# Synthesize on this machine (CUDA, MPS or CPU, whichever is available) and play the audio.
#
# Usage: scripts/run.sh [--compile] [--style default|sad|whisper [--check]] ["sentence" ...]
#   --style  speak with the Expresso emotion fine-tune (one speaker's voice; noncommercial, see EXPRESSO.md)
#   --check  with --style: also transcribe each clip with Whisper and regenerate garbled ones (a few seconds per clip)
# Env:   NOPLAY=1 to skip playback   VIBE_TTS_DEVICE=cuda|mps|cpu to force a backend
# With no sentences, speaks speak.py's built-in test sentences. For a Colab GPU: run_colab.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/config.sh

N=0; VALUE_NEXT=0
for a in "$@"; do
  if [ "$VALUE_NEXT" = 1 ]; then VALUE_NEXT=0; continue; fi   # the value of --style is not a sentence
  case "$a" in
    --style) VALUE_NEXT=1 ;;
    --*) ;;
    *) N=$((N + 1)) ;;
  esac
done
[ "$N" -gt 0 ] || N=3

[ -x .venv/bin/python ] || { echo "[run] no .venv; run scripts/setup.sh first" >&2; exit 1; }
.venv/bin/python scripts/speak.py "$@"

[ -z "${NOPLAY:-}" ] || exit 0
PLAYER="$(command -v afplay || command -v aplay || true)"
[ -n "$PLAYER" ] || { echo "[run] no audio player found; wavs are in $OUTPUT_DIR/"; exit 0; }
for ((i = 0; i < N; i++)); do "$PLAYER" "$OUTPUT_DIR/speak_$i.wav"; done
