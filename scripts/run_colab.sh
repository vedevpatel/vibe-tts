#!/usr/bin/env bash
# Colab variant of run.sh: synthesize on a Colab GPU, fetch the wavs, and play them.
#
# Usage: scripts/run_colab.sh [--compile] ["sentence" ...]
# Env:   GPU=L4 (T4, L4, A100, H100, G4)   NOPLAY=1 to skip playback
# With no sentences, speaks speak.py's built-in test sentences. Needs the colab CLI; see
# scripts/remote_speak.sh. Wavs land in <output dir>/colab/.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/config.sh

N=0
for a in "$@"; do [[ "$a" == --* ]] || N=$((N + 1)); done
[ "$N" -gt 0 ] || N=3

scripts/remote_speak.sh "$@"

[ -z "${NOPLAY:-}" ] || exit 0
PLAYER="$(command -v afplay || command -v aplay || true)"
[ -n "$PLAYER" ] || { echo "[run_colab] no audio player found; wavs are in $OUTPUT_DIR/colab/"; exit 0; }
for ((i = 0; i < N; i++)); do "$PLAYER" "$OUTPUT_DIR/colab/speak_$i.wav"; done
