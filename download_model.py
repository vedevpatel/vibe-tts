"""Download parler-tts/parler-tts-mini-v1 into pretrained_models/.

Idempotent: files already present at the pinned revision are reused. The resolved
revision is recorded in pretrained_models/parler-tts-mini-v1/REVISION.json.
"""
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "parler-tts/parler-tts-mini-v1"
REVISION = "0392b9451a601e528fd863bbb0598431fee810d9"
LOCAL_DIR = Path(__file__).resolve().parent / "pretrained_models" / "parler-tts-mini-v1"

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
