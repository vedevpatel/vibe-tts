#!/usr/bin/env bash
# Synthesize on this machine (CUDA, MPS or CPU, whichever is available) and play the audio.
#
# Usage: scripts/run.sh [--compile] ["sentence" ...]
# Env:   NOPLAY=1 to skip playback   VIBE_TTS_DEVICE=cuda|mps|cpu to force a backend
# With no sentences, speaks speak.py's built-in test sentences. For a Colab GPU: run_colab.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/config.sh

N=0
for a in "$@"; do [[ "$a" == --* ]] || N=$((N + 1)); done
[ "$N" -gt 0 ] || N=3

[ -x .venv/bin/python ] || { echo "[run] no .venv; run scripts/setup.sh first" >&2; exit 1; }
.venv/bin/python scripts/speak.py "$@"

[ -z "${NOPLAY:-}" ] || exit 0
PLAYER="$(command -v afplay || command -v aplay || true)"
[ -n "$PLAYER" ] || { echo "[run] no audio player found; wavs are in $OUTPUT_DIR/"; exit 0; }
for ((i = 0; i < N; i++)); do "$PLAYER" "$OUTPUT_DIR/speak_$i.wav"; done
