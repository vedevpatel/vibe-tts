#!/usr/bin/env bash
# Run scripts/speak.py on a Colab GPU and fetch the wavs into <output dir>/colab/.
# Colab only: local development never needs this (see scripts/setup.sh and scripts/run.sh).
#
# Usage: scripts/remote_speak.sh [--compile] ["sentence" ...]
# Env:   GPU=L4 (T4, L4, A100, H100, G4)   SESSION=tts-speak
#        SCRIPT=scripts/speak.py  FILES="a.wav b.wav" (outputs to fetch; default speak_0..N-1.wav)
# Needs the `colab` CLI (uv tool install google-colab-cli) signed in to a Google account with
# Colab GPU access (check with `colab usage`). No other credentials: the model is public, so no
# Hugging Face token is needed, and .env is not uploaded (the VM uses default paths).
# The VM is always stopped on exit, including on errors and Ctrl-C.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/config.sh

GPU="${GPU:-L4}"
SCRIPT="${SCRIPT:-scripts/speak.py}"
SESSION="${SESSION:-tts-speak}"
TMP="$(mktemp -d)"
REMOTE=/content/vibe-tts

cleanup() {
  echo "[remote_speak] stopping session $SESSION"
  colab stop -s "$SESSION" >/dev/null 2>&1 || true
  rm -rf "$TMP"
}
trap cleanup EXIT

# Runs a small Python file on the VM's kernel. Long jobs go through nohup (see below):
# the CLI's client times out on calls that run longer than about a minute.
remote() { colab exec -s "$SESSION" -f "$1"; }

# Count sentences so we know how many wavs to fetch (flags don't count).
N=0
for a in "$@"; do [[ "$a" == --* ]] || N=$((N + 1)); done
[ "$N" -gt 0 ] || N=3   # speak.py's built-in default sentences

echo "[remote_speak] starting $GPU session $SESSION"
colab new -s "$SESSION" --gpu "$GPU"

cat > "$TMP/mkdir.py" <<EOF
import os
os.makedirs("$REMOTE/scripts", exist_ok=True)
EOF
remote "$TMP/mkdir.py"
colab upload -s "$SESSION" scripts/_common.py "$REMOTE/scripts/_common.py"
colab upload -s "$SESSION" scripts/download_models.py "$REMOTE/scripts/download_models.py"
colab upload -s "$SESSION" "$SCRIPT" "$REMOTE/$SCRIPT"

# Install, fetch weights and synthesize in the background; the shell below polls the log.
ARGS_JSON="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@")"
cat > "$TMP/job.py" <<EOF
import json, shlex, subprocess
args = " ".join(shlex.quote(a) for a in json.loads(r'''$ARGS_JSON'''))
script = f"""cd $REMOTE && export USE_TF=0 USE_FLAX=0 &&
pip install -q 'git+$PARLER_REPO.git@$PARLER_REV' 'transformers==4.46.1' soundfile &&
pip install -q -U protobuf &&
python scripts/download_models.py &&
python $SCRIPT {args}"""
_ = subprocess.Popen(["nohup", "sh", "-c", script + "; echo EXIT=\$? >> /content/job.log"],
                 stdout=open("/content/job.log", "w"), stderr=subprocess.STDOUT)
EOF
remote "$TMP/job.py"

cat > "$TMP/poll.py" <<'EOF'
import subprocess
log = open("/content/job.log").read()
lines = [l for l in log.splitlines() if l.startswith(("device=", "warm-up", "speak_", "EXIT=")) or "Error" in l]
print("\n".join(lines))
EOF

echo "[remote_speak] waiting for job (install + weights + synthesis, a few minutes)"
SHOWN=0
FAILS=0
while true; do
  sleep 20
  if ! OUT="$(remote "$TMP/poll.py" 2>/dev/null)"; then
    FAILS=$((FAILS + 1))
    [ "$FAILS" -lt 6 ] || { echo "[remote_speak] lost the connection to the VM (6 polls failed in a row)" >&2; exit 1; }
    continue
  fi
  FAILS=0
  TOTAL="$(printf '%s\n' "$OUT" | grep -c . || true)"
  [ "$TOTAL" -gt "$SHOWN" ] && printf '%s\n' "$OUT" | tail -n +"$((SHOWN + 1))"
  SHOWN="$TOTAL"
  if printf '%s\n' "$OUT" | grep -q '^EXIT='; then break; fi
done
echo "$OUT" | grep -q '^EXIT=0' || { echo "[remote_speak] remote job failed; see lines above" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR/colab"
if [ -z "${FILES:-}" ]; then
  FILES=""; for ((i = 0; i < N; i++)); do FILES="$FILES speak_$i.wav"; done
fi
for f in $FILES; do
  colab download -s "$SESSION" "$REMOTE/outputs/$f" "$OUTPUT_DIR/colab/$f"
done
echo "[remote_speak] done: $FILES -> $OUTPUT_DIR/colab/"
