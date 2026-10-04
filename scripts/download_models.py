"""Download parler-tts/parler-tts-mini-v1 into <model dir>/parler-tts-mini-v1.

The model dir is VIBE_TTS_MODEL_DIR (default <repo>/models). Idempotent: files already present
at the pinned revision are reused. The resolved revision is recorded in REVISION.json there.
"""
import json

from _common import PARLER_MODEL, model_root  # first: loads .env before huggingface_hub reads HF_HOME
from huggingface_hub import HfApi, snapshot_download

REPO_ID = "parler-tts/parler-tts-mini-v1"
REVISION = "0392b9451a601e528fd863bbb0598431fee810d9"
LOCAL_DIR = model_root() / PARLER_MODEL

# Weights, model config and tokenizer assets. The text encoder (flan-t5) and DAC
# audio codec weights are bundled in model.safetensors, so no companion repos are needed.
ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "README.md",
]


def main():
    # No parents=True: if the model dir lives on an external drive that is not mounted, fail
    # instead of quietly creating the path on the wrong disk.
    LOCAL_DIR.parent.mkdir(exist_ok=True)
    LOCAL_DIR.mkdir(exist_ok=True)
    path = snapshot_download(
        REPO_ID,
        revision=REVISION,
        local_dir=LOCAL_DIR,
        allow_patterns=ALLOW_PATTERNS,
    )
    info = HfApi().model_info(REPO_ID, revision=REVISION)
    record = {
        "repo_id": REPO_ID,
        "revision": info.sha,
        "files": sorted(p.name for p in LOCAL_DIR.iterdir() if p.is_file()),
    }
    (LOCAL_DIR / "REVISION.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{REPO_ID}@{info.sha} -> {path}")


if __name__ == "__main__":
    main()
