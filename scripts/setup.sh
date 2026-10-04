#!/usr/bin/env bash
# The one supported bootstrap: repo-local .venv, pinned dependencies, pinned parler-tts source,
# model weights. Safe to re-run.
#
# Usage: scripts/setup.sh [--platform mac-arm64|linux-cpu|linux-cuda] [--no-models]
#   --platform   override auto-detection (default: Darwin arm64 -> mac-arm64; Linux x86_64 ->
#                linux-cuda if nvidia-smi sees a GPU, else linux-cpu)
#   --no-models  skip downloading the model weights
# Machine-specific paths (model dir, output dir, HF cache) go in .env; see .env.example.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/config.sh

die() { echo "[setup] error: $*" >&2; exit 1; }
say() { echo "[setup] $*"; }

PLATFORM=""
GET_MODELS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --platform) [ $# -ge 2 ] || die "--platform needs a value"; PLATFORM="$2"; shift 2 ;;
    --no-models) GET_MODELS=0; shift ;;
    -h | --help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done

has_nvidia_gpu() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }

detect_platform() {
  local os arch
  os="$(uname -s)"; arch="$(uname -m)"
  case "$os/$arch" in
    Darwin/arm64) echo mac-arm64 ;;
    Linux/x86_64) if has_nvidia_gpu; then echo linux-cuda; else echo linux-cpu; fi ;;
    Darwin/x86_64) die "Darwin x86_64 is not supported (Intel Mac, or a shell running under Rosetta; use a native arm64 shell)" ;;
    *) die "unsupported platform $os/$arch (supported: Darwin/arm64, Linux/x86_64)" ;;
  esac
}

[ -n "$PLATFORM" ] || PLATFORM="$(detect_platform)"
[ -f "requirements/$PLATFORM.txt" ] || die "unknown platform '$PLATFORM': no requirements/$PLATFORM.txt"
say "platform: $PLATFORM (override with --platform)"

command -v uv >/dev/null 2>&1 || die "uv not found. Install it, then re-run:
    curl -LsSf https://astral.sh/uv/install.sh | sh      (or: brew install uv)
  Nothing else is needed: uv downloads Python $PYTHON_VERSION itself and installs only into .venv."
command -v git >/dev/null 2>&1 || die "git not found"

# --- virtual environment (.venv is disposable: rm -rf .venv && scripts/setup.sh rebuilds it) ---
if [ -x .venv/bin/python ]; then
  have="$(.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  [ "$have" = "$PYTHON_VERSION" ] || die ".venv has Python $have but this repo needs $PYTHON_VERSION. Rebuild it: rm -rf .venv && scripts/setup.sh"
  say ".venv exists (Python $have); reusing it"
else
  [ ! -e .venv ] || die ".venv exists but has no bin/python. Remove it and re-run: rm -rf .venv && scripts/setup.sh"
  say "creating .venv (Python $PYTHON_VERSION)"
  uv venv --python "$PYTHON_VERSION" .venv
fi

# --- third-party source, pinned to PARLER_REV ---
SRC=third_party/parler-tts
mkdir -p third_party
if [ ! -d "$SRC/.git" ]; then
  [ ! -e "$SRC" ] || [ -z "$(ls -A "$SRC")" ] || die "$SRC exists but is not a git checkout; move it aside and re-run"
  say "cloning $PARLER_REPO"
  git clone --quiet "$PARLER_REPO" "$SRC" || die "could not clone $PARLER_REPO"
fi
[ -z "$(git -C "$SRC" status --porcelain)" ] || die "$SRC has local changes; it must stay unmodified. Stash or discard them (git -C $SRC status) and re-run"
if ! git -C "$SRC" cat-file -e "$PARLER_REV^{commit}" 2>/dev/null; then
  say "fetching $SRC"
  git -C "$SRC" fetch --quiet origin || die "git fetch failed in $SRC"
  git -C "$SRC" cat-file -e "$PARLER_REV^{commit}" 2>/dev/null || git -C "$SRC" fetch --quiet origin "$PARLER_REV" || true
  git -C "$SRC" cat-file -e "$PARLER_REV^{commit}" 2>/dev/null || die "commit $PARLER_REV not found in $PARLER_REPO"
fi
git -C "$SRC" checkout --quiet --detach "$PARLER_REV" || die "could not check out $PARLER_REV in $SRC"
say "$SRC at $(git -C "$SRC" rev-parse --short=12 HEAD)"

# --- dependencies: shared pins + this platform's torch wheels, then parler-tts editable ---
say "installing requirements/base.txt + requirements/$PLATFORM.txt"
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  -r requirements/base.txt -r "requirements/$PLATFORM.txt"
# --no-deps: base.txt already pins everything parler-tts needs, so upstream's loose ranges
# (and its unpinned audiotools git URL) cannot change the environment.
uv pip install --python .venv/bin/python --no-deps -e "$SRC"

# --- local directories (models/ and outputs/ are tracked as empty dirs, so this is a no-op on a clone) ---
mkdir -p models outputs

# --- model weights (download_models.py is idempotent and pinned to a revision) ---
if [ "$GET_MODELS" = 1 ]; then
  say "model weights (skip with --no-models)"
  .venv/bin/python scripts/download_models.py
fi

say "done. Next: .venv/bin/python scripts/smoke_test.py"
