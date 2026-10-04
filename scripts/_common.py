"""Shared runtime helpers: .env loading, model/output paths, device selection.

Import this module BEFORE transformers / huggingface_hub: it loads .env at import time so that
HF_HOME and friends are in place before those libraries read them.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARLER_MODEL = "parler-tts-mini-v1"


def _load_env(path=ROOT / ".env"):
    """KEY=VALUE lines from .env into os.environ; variables already set in the environment win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, val = line.removeprefix("export ").partition("=")
        key, val = key.strip(), val.strip()
        if sep and len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if sep:
            os.environ.setdefault(key, val)


_load_env()


def _path_from_env(var, default):
    return Path(os.environ.get(var) or default).expanduser()


def model_root():
    """Directory holding downloaded models. VIBE_TTS_MODEL_DIR, default <repo>/models."""
    return _path_from_env("VIBE_TTS_MODEL_DIR", ROOT / "models")


def model_dir(name=PARLER_MODEL):
    """Local directory of one downloaded model; fails clearly if it has not been downloaded."""
    path = model_root() / name
    if not (path / "config.json").is_file():
        raise SystemExit(
            f"Model not found at {path}\n"
            f"Download it with: .venv/bin/python scripts/download_models.py "
            f"(models go to VIBE_TTS_MODEL_DIR, default <repo>/models)"
        )
    return path


def output_dir():
    """Where scripts write audio. VIBE_TTS_OUTPUT_DIR, default <repo>/outputs."""
    return _path_from_env("VIBE_TTS_OUTPUT_DIR", ROOT / "outputs")


def pick_device():
    """cuda, else mps, else cpu. VIBE_TTS_DEVICE=cuda|mps|cpu forces one (e.g. to compare numerics)."""
    import torch

    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    forced = os.environ.get("VIBE_TTS_DEVICE")
    if forced:
        if forced not in available:
            raise SystemExit(f"VIBE_TTS_DEVICE={forced!r}: expected one of {', '.join(available)}")
        if not available[forced]:
            raise SystemExit(f"VIBE_TTS_DEVICE={forced} but torch reports it is not available on this machine")
        return forced
    return next(name for name, ok in available.items() if ok)
