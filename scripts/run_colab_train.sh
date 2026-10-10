#!/usr/bin/env bash
# Fine-tune on a Colab GPU, evaluate before/after on that same GPU, and fetch the results.
# Colab only: the 880M-parameter fine-tune does not fit a 16 GB Mac. Local inference is untouched.
#
# Usage: scripts/run_colab_train.sh [--smoke] [--run NAME]
#   --smoke     4 training steps, 2 evaluation clips, no checkpoint download (checks the whole pipeline cheaply)
#   --run NAME  run directory under <data root>/runs (default emotion-v1)
#   --keep-vm-on-failure  if the remote job fails, leave the VM running (and say so) so finished work can be recovered
# Env:   GPU=L4 (T4, L4, A100, H100, G4)   SESSION=tts-train-expresso
# Needs: the `colab` CLI signed in to an account with GPU access (`colab usage`), prepared data
#        (scripts/prepare_expresso.py) and a frozen eval set (evaluate_emotion.py select).
# Uploads: scripts, config, and a tarball of processed audio + splits (NOT the raw 48 kHz release, NOT .env).
# Fetches: <output dir>/colab_train/ (eval audio, logs) and, unless --smoke, the exported checkpoint into
#          <data root>/runs/<run>/export. Colab's file API rejects large single files, so everything big moves in
#          chunks with retries and is verified against a sha256. The VM is always stopped on exit, and only the
#          session named here is ever touched.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/config.sh

SMOKE=0; RUN=emotion-v1; KEEP_ON_FAILURE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --keep-vm-on-failure) KEEP_ON_FAILURE=1; shift ;;
    --run) [ $# -ge 2 ] || { echo "--run needs a value" >&2; exit 1; }; RUN="$2"; shift 2 ;;
    -h | --help) sed -n "2,17p" "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

GPU="${GPU:-L4}"
SESSION="${SESSION:-tts-train-expresso}"
REMOTE=/content/vibe-tts
UP_CHUNK=8m          # upload chunk size (the contents API fails on large bodies)
DOWN_CHUNK=100m      # download chunk size
TMP="$(mktemp -d)"
PY=.venv/bin/python
[ -x "$PY" ] || { echo "[colab_train] no .venv; run scripts/setup.sh first" >&2; exit 1; }
sha256() { $PY -c 'import sys; sys.path.insert(0, "scripts"); from expresso_common import sha256_file; print(sha256_file(sys.argv[1]))' "$1"; }

DATA_ROOT="$($PY -c 'import sys; sys.path.insert(0, "scripts"); from expresso_common import Config; print(Config().data_root)')"
for f in processed/manifest.jsonl splits/train.jsonl splits/validation.jsonl splits/test.jsonl splits/eval_set.json; do
  [ -f "$DATA_ROOT/$f" ] || { echo "[colab_train] missing $DATA_ROOT/$f (run prepare_expresso.py and evaluate_emotion.py select)" >&2; exit 1; }
done

# One run per session name at a time: a second launch would fight the first over the same VM (and bill for both).
LOCK="${TMPDIR:-/tmp}/vibe-tts-colab-$SESSION.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  if kill -0 "$(cat "$LOCK/pid" 2>/dev/null)" 2>/dev/null; then
    echo "[colab_train] another run (pid $(cat "$LOCK/pid")) already holds session $SESSION; refusing to start a second" >&2
    rm -rf "$TMP"; exit 1
  fi
  rm -rf "$LOCK"; mkdir "$LOCK"
fi
echo $$ > "$LOCK/pid"
STARTED_VM=0
KEEP_VM_ON_FAIL=0   # set once the remote job has succeeded: a failed fetch must not destroy an hour of training
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ "$STARTED_VM" = 1 ]; then
    if [ "$rc" != 0 ] && { [ "$KEEP_VM_ON_FAIL" = 1 ] || [ "$KEEP_ON_FAILURE" = 1 ]; }; then
      echo "[colab_train] LEAVING session $SESSION RUNNING (it bills until stopped) so nothing is lost: the remote job failed or its results could not be fetched." >&2
      echo "[colab_train] Fetch by hand (colab download -s $SESSION /content/export.tar.part-000 ...; results are /content/results.tgz.part-*), then: colab stop -s $SESSION" >&2
    else
      echo "[colab_train] stopping session $SESSION"
      colab stop -s "$SESSION" >/dev/null 2>&1 || true
    fi
  fi
  rm -rf "$TMP" "$LOCK"
}
trap cleanup EXIT
trap 'echo "[colab_train] interrupted" >&2; exit 130' INT TERM

ERR="$TMP/colab.err"   # the CLI's background threads print noisy tracebacks on stderr; keep them out of the log
: > "$ERR"
with_timeout() {       # with_timeout <seconds> cmd...   (macOS has no `timeout`; perl is always there)
  local secs="$1"; shift
  perl -e 'alarm shift; exec @ARGV' "$secs" "$@"
}
remote() {   # run a python file on the VM's kernel; a fresh kernel sometimes drops its first connection, so retry
  local try out
  for try in 1 2 3 4 5; do
    # exit codes are unreliable (the CLI can print a traceback and still exit 0): success means the snippet reached its sentinel
    if out="$(with_timeout 180 colab exec -s "$SESSION" -f "$1" 2>>"$ERR")" && [ "$(printf '%s\n' "$out" | tail -1)" = "@@OK@@" ]; then
      printf '%s\n' "$out" | sed '$d'; return 0
    fi
    sleep 8
  done
  echo "[colab_train] remote command failed: $(basename "$1"); CLI errors (tail):" >&2; tail -5 "$ERR" >&2
  return 1
}
upload() {   # upload <local> <remote path>, with retries
  local try
  for try in 1 2 3 4; do
    with_timeout 300 colab upload -s "$SESSION" "$1" "$2" >/dev/null 2>>"$ERR" && return 0
    echo "[colab_train] upload retry $try: $(basename "$1")" >&2; sleep 4
  done
  echo "[colab_train] could not upload $1" >&2; return 1
}
download() {  # download <remote path> <local>, with retries
  local try
  for try in 1 2 3 4; do
    with_timeout 600 colab download -s "$SESSION" "$1" "$2" >/dev/null 2>>"$ERR" && return 0
    echo "[colab_train] download retry $try: $1" >&2; sleep 4
  done
  echo "[colab_train] could not download $1" >&2; return 1
}

echo "[colab_train] packing data from $DATA_ROOT"
# Straight from the data root, without macOS AppleDouble '._*' sidecars (external volumes create them) or Finder files.
COPYFILE_DISABLE=1 tar -czf "$TMP/data.tgz" -C "$DATA_ROOT" -s ',^,data/expresso/,' --exclude '._*' --exclude .DS_Store \
  processed splits raw/SOURCES.json
DATA_SHA="$(sha256 "$TMP/data.tgz")"
echo "[colab_train] data.tgz $(du -h "$TMP/data.tgz" | cut -f1), sha256 ${DATA_SHA:0:12}"

echo "[colab_train] starting $GPU session $SESSION"
case "$(colab status -s "$SESSION" 2>&1 || true)" in
  *"not found"*) ;;
  *) echo "[colab_train] a session named $SESSION already exists and was not started by this run; stop it first: colab stop -s $SESSION" >&2; exit 1 ;;
esac
STARTED_VM=1    # from here on the cleanup trap stops the VM, on any exit including Ctrl-C
colab new -s "$SESSION" --gpu "$GPU" 2>>"$ERR"

cat > "$TMP/mkdir.py" <<EOF
import os
for d in ("scripts", "configs", "parts"):
    os.makedirs("$REMOTE/" + d, exist_ok=True)
print("READY")
print("@@OK@@")
EOF
remote "$TMP/mkdir.py" | grep -q READY || { echo "[colab_train] the VM kernel did not come up" >&2; exit 1; }
for f in _common.py expresso_common.py download_models.py train_emotion.py evaluate_emotion.py config.sh; do
  upload "scripts/$f" "$REMOTE/scripts/$f"
done
upload configs/expresso_emotion.toml "$REMOTE/configs/expresso_emotion.toml"
split -a 3 -b "$UP_CHUNK" "$TMP/data.tgz" "$TMP/data.tgz.part-"
NPARTS="$(ls "$TMP"/data.tgz.part-* | wc -l | tr -d ' ')"
echo "[colab_train] uploading data in $NPARTS chunks"
for p in "$TMP"/data.tgz.part-*; do upload "$p" "$REMOTE/parts/$(basename "$p")"; done

if [ "$SMOKE" = 1 ]; then TRAIN_FLAGS="--max-steps 4"; GEN_FLAGS="--limit 2"; else TRAIN_FLAGS=""; GEN_FLAGS=""; fi
cat > "$TMP/job.sh" <<EOF
set -e
cd $REMOTE
export USE_TF=0 USE_FLAX=0 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
echo "[job] reassembling and checking the data upload"
cat parts/data.tgz.part-* > data.tgz
echo "$DATA_SHA  data.tgz" | sha256sum -c -
tar xzf data.tgz
echo "[job] installing pinned parler-tts $PARLER_REV and training dependencies"
mkdir -p third_party
[ -d third_party/parler-tts ] || git clone -q $PARLER_REPO third_party/parler-tts
git -C third_party/parler-tts checkout -q --detach $PARLER_REV
pip install -q -e third_party/parler-tts tomli 'transformers==4.46.1' 'accelerate==1.15.0' 'datasets==3.6.0' 'evaluate==0.4.6' 'jiwer==4.0.0' soundfile
pip install -q -U protobuf
echo "[job] model weights"
python scripts/download_models.py
python -c "import torch,transformers,accelerate,datasets;print('[job] versions torch',torch.__version__,'transformers',transformers.__version__,'accelerate',accelerate.__version__,'datasets',datasets.__version__,'gpu',torch.cuda.get_device_name(0))"
echo "[job] baseline generation (unchanged model)"
python scripts/evaluate_emotion.py generate --label baseline $GEN_FLAGS
echo "[job] training"
python scripts/train_emotion.py train --run $RUN $TRAIN_FLAGS
if [ "$SMOKE" = 0 ]; then
  echo "[job] fine-tuned generation (same eval set, descriptions, seeds, settings, GPU)"
  python scripts/evaluate_emotion.py generate --label finetuned --model data/expresso/runs/$RUN/export $GEN_FLAGS
  echo "[job] packing the checkpoint"
  tar -C data/expresso/runs/$RUN -cf /content/export.tar export
  split -a 3 -b $DOWN_CHUNK -d /content/export.tar /content/export.tar.part-
  (cd /content && sha256sum export.tar > export.tar.sha256)
fi
tar -czf /content/results.tgz outputs/expresso data/expresso/runs/$RUN/train.log data/expresso/runs/$RUN/train_config.json data/expresso/runs/$RUN/environment.json
split -a 3 -b $DOWN_CHUNK -d /content/results.tgz /content/results.tgz.part-
(cd /content && sha256sum results.tgz > results.tgz.sha256)
echo "[job] finished"
EOF
upload "$TMP/job.sh" "$REMOTE/job.sh"

cat > "$TMP/start.py" <<EOF
import subprocess
subprocess.Popen(["nohup", "sh", "-c", "bash $REMOTE/job.sh > /content/job.log 2>&1; echo EXIT=\$? >> /content/job.log"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("@@OK@@")
EOF
remote "$TMP/start.py" >/dev/null

cat > "$TMP/poll.py" <<'EOF'
log = open("/content/job.log").read().splitlines()
keep = [l for l in log if l.startswith(("[job]", "EXIT=", "data.tgz")) or "Eval results" in l or "Step... (" in l
        or "Error" in l or "Traceback" in l or l.startswith("[") and "/" in l[:12]]
print("\n".join(keep))
print("@@OK@@")
EOF

echo "[colab_train] job started; polling (install, weights, baseline, training, fine-tuned generation)"
SHOWN=0; FAILS=0
while true; do
  sleep 60
  if ! OUT="$(remote "$TMP/poll.py" 2>/dev/null)"; then
    FAILS=$((FAILS + 1))
    [ "$FAILS" -lt 10 ] || { echo "[colab_train] lost the connection to the VM (10 polls failed in a row)" >&2; exit 1; }
    continue
  fi
  FAILS=0
  TOTAL="$(printf '%s\n' "$OUT" | grep -c . || true)"
  [ "$TOTAL" -gt "$SHOWN" ] && printf '%s\n' "$OUT" | tail -n +"$((SHOWN + 1))"
  SHOWN="$TOTAL"
  if printf '%s\n' "$OUT" | grep -q '^EXIT='; then break; fi
done
if ! printf '%s\n' "$OUT" | grep -q '^EXIT=0'; then
  echo "[colab_train] remote job failed; last log lines:" >&2
  cat > "$TMP/tail.py" <<'EOF'
print("\n".join(open("/content/job.log").read().splitlines()[-40:]))
print("@@OK@@")
EOF
  remote "$TMP/tail.py" >&2 || true
  exit 1
fi

KEEP_VM_ON_FAIL=1   # from here on a failure keeps the VM up
# fetch_chunked <remote prefix, e.g. /content/results.tgz> <local file>: download parts, reassemble, verify sha256
fetch_chunked() {
  local prefix="$1" out="$2" parts p want got
  cat > "$TMP/ls.py" <<EOF
import glob, os
print(" ".join(sorted(os.path.basename(p) for p in glob.glob("$prefix.part-*"))))
print("@@OK@@")
EOF
  parts="$(remote "$TMP/ls.py" | tail -1)"
  : > "$out"
  for p in $parts; do
    download "/content/$p" "$TMP/$p"
    cat "$TMP/$p" >> "$out"; rm -f "$TMP/$p"
  done
  download "$prefix.sha256" "$TMP/$(basename "$prefix").sha256"
  want="$(cut -d' ' -f1 "$TMP/$(basename "$prefix").sha256")"
  got="$(sha256 "$out")"
  [ "$want" = "$got" ] || { echo "[colab_train] checksum mismatch for $(basename "$prefix") ($got != $want)" >&2; return 1; }
}

mkdir -p "$OUTPUT_DIR/colab_train"
fetch_chunked /content/results.tgz "$OUTPUT_DIR/colab_train/results.tgz"
tar -xzf "$OUTPUT_DIR/colab_train/results.tgz" -C "$OUTPUT_DIR/colab_train"
# Put each generated run where evaluate_emotion.py (page, measure, score) looks for it: <output dir>/expresso/runs/<label>
mkdir -p "$OUTPUT_DIR/expresso/runs"
for d in "$OUTPUT_DIR/colab_train/outputs/expresso/runs/"*/; do
  [ -d "$d" ] || continue
  rm -rf "$OUTPUT_DIR/expresso/runs/$(basename "$d")"; cp -R "$d" "$OUTPUT_DIR/expresso/runs/$(basename "$d")"
done
echo "[colab_train] results verified: runs are in $OUTPUT_DIR/expresso/runs/, logs in $OUTPUT_DIR/colab_train/"

if [ "$SMOKE" = 0 ]; then
  DEST="$DATA_ROOT/runs/$RUN"; mkdir -p "$DEST"
  fetch_chunked /content/export.tar "$DEST/export.tar"
  rm -rf "$DEST/export"; tar -xf "$DEST/export.tar" -C "$DEST"; rm -f "$DEST/export.tar"
  cp "$OUTPUT_DIR/colab_train/data/expresso/runs/$RUN/"{train.log,train_config.json,environment.json} "$DEST/" 2>/dev/null || true
  echo "[colab_train] checkpoint verified and extracted: $DEST/export"
fi
KEEP_VM_ON_FAIL=0
