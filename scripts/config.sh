# Sourced by the other scripts (not executable on its own). Run from the repo root.
#
# 1. Pins that setup.sh enforces and remote_speak.sh reuses: the single place to bump them.
# 2. Loads .env, if present. Variables already set in the environment win over .env.
#    Plain KEY=VALUE lines; a leading ~ is expanded, nothing else is.

PYTHON_VERSION=3.11
PARLER_REPO=https://github.com/huggingface/parler-tts
PARLER_REV=d108732cd57788ec86bc857d99a6cabd66663d68   # huggingface/parler-tts main, 2024-12-10

load_env() {
  [ -f .env ] || return 0
  local line key val
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    case "$line" in '' | '#'*) continue ;; esac
    line="${line#export }"
    key="${line%%=*}"
    val="${line#*=}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    case "$val" in
      \"*\") val="${val#\"}"; val="${val%\"}" ;;
      \'*\') val="${val#\'}"; val="${val%\'}" ;;
    esac
    case "$val" in "~" | "~/"*) val="$HOME${val#\~}" ;; esac
    [ -n "${!key+x}" ] || export "$key=$val"
  done < .env
}
load_env

# Where scripts write audio (same default as scripts/_common.py).
OUTPUT_DIR="${VIBE_TTS_OUTPUT_DIR:-outputs}"
