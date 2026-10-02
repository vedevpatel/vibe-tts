#!/usr/bin/env bash
# Reproduce the environment: pinned upstream source + project .venv.
set -euo pipefail
cd "$(dirname "$0")/.."

PARLER_REV=d108732cd57788ec86bc857d99a6cabd66663d68   # huggingface/parler-tts main, 2024-12-10

if [ ! -d third_party/parler-tts ]; then
  git clone https://github.com/huggingface/parler-tts third_party/parler-tts
fi
git -C third_party/parler-tts checkout "$PARLER_REV"

uv venv --python 3.11 .venv
# requirements.lock pins every transitive dependency; the editable install
# registers the upstream package without modifying it.
uv pip install --python .venv/bin/python -r requirements.lock
uv pip install --python .venv/bin/python --no-deps -e third_party/parler-tts
