#!/usr/bin/env bash
# Build the isolated TRAINING environment .venv-train (the inference .venv is never touched).
# Run scripts/setup.sh first: this reuses its pinned third_party/parler-tts checkout.
#
# Usage: scripts/setup_train.sh [--platform mac-arm64|linux-cpu|linux-cuda]
# Installs requirements/base.txt (without its fsspec pin, which conflicts with datasets 3.6.0) +
# the platform's torch file + requirements/train.txt, then parler-tts editable with --no-deps.
# Real training needs a CUDA GPU (see scripts/run_colab_train.sh); on a Mac this environment is for
# preparing/inspecting data and for smoke tests.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/config.sh

die() { echo "[setup_train] error: $*" >&2; exit 1; }
say() { echo "[setup_train] $*"; }

PLATFORM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --platform) [ $# -ge 2 ] || die "--platform needs a value"; PLATFORM="$2"; shift 2 ;;
    -h | --help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done

if [ -z "$PLATFORM" ]; then
  case "$(uname -s)/$(uname -m)" in
    Darwin/arm64) PLATFORM=mac-arm64 ;;
    Linux/x86_64) if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then PLATFORM=linux-cuda; else PLATFORM=linux-cpu; fi ;;
    *) die "unsupported platform $(uname -s)/$(uname -m) (see scripts/setup.sh)" ;;
  esac
fi
[ -f "requirements/$PLATFORM.txt" ] || die "unknown platform '$PLATFORM'"
command -v uv >/dev/null 2>&1 || die "uv not found (brew install uv)"
SRC=third_party/parler-tts
[ -d "$SRC/.git" ] || die "$SRC missing: run scripts/setup.sh first"
[ "$(git -C "$SRC" rev-parse HEAD)" = "$PARLER_REV" ] || die "$SRC is not at the pinned revision $PARLER_REV: run scripts/setup.sh"

VENV=.venv-train
if [ -x "$VENV/bin/python" ]; then
  have="$($VENV/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  [ "$have" = "$PYTHON_VERSION" ] || die "$VENV has Python $have, need $PYTHON_VERSION: rm -rf $VENV and re-run"
  say "$VENV exists; reusing it"
else
  [ ! -e "$VENV" ] || die "$VENV exists but has no bin/python: rm -rf $VENV and re-run"
  uv venv --python "$PYTHON_VERSION" "$VENV"
fi

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
sed '/^fsspec==/d' requirements/base.txt > "$TMP/base-no-fsspec.txt"
say "installing base (no fsspec pin) + $PLATFORM torch + train.txt"
uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match \
  -r "$TMP/base-no-fsspec.txt" -r "requirements/$PLATFORM.txt" -r requirements/train.txt
uv pip install --python "$VENV/bin/python" --no-deps -e "$SRC"
say "done. Check: USE_TF=0 $VENV/bin/python -c 'import accelerate, datasets, parler_tts; print(accelerate.__version__, datasets.__version__)'"
