# Setup

One command: `./scripts/setup.sh`, then `.venv/bin/python scripts/smoke_test.py`.
Flags: `--platform mac-arm64|linux-cpu|linux-cuda` overrides detection, `--no-models` skips the
weights download. Re-running is safe: it reuses `.venv`, re-pins `third_party/`, and installs
only what differs.

## What setup.sh does
1. Detects the platform: Darwin arm64 → `mac-arm64`; Linux x86_64 → `linux-cuda` if `nvidia-smi`
   sees a GPU, otherwise `linux-cpu`. Anything else stops with an error.
2. Creates `.venv` with `uv` (Python 3.11, downloaded by uv if missing). `.venv` is disposable:
   `rm -rf .venv && ./scripts/setup.sh` rebuilds it. An existing `.venv` on a different Python
   version is never replaced silently.
3. Clones `third_party/parler-tts` if absent and checks out the pinned commit. It refuses to
   continue if that checkout has local changes (it must stay unmodified).
4. Installs `requirements/base.txt` plus `requirements/<platform>.txt`, then installs `parler-tts`
   editable with `--no-deps`.
5. Downloads the model to the model dir with `scripts/download_models.py` (pinned revision).

## Support matrix
| Platform | Status | Backend | Notes |
|---|---|---|---|
| macOS arm64 | tested (macOS 15.5, Apple M4) | MPS, CPU | torch 2.14.1 wheels need macOS 14+ |
| Linux x86_64 | CPU path resolved against Linux wheels, not yet run on a real Ubuntu host | CPU | CPU-only torch wheels from the PyTorch index |
| Linux x86_64 + NVIDIA | resolved, not yet run on hardware | CUDA | PyPI torch is a CUDA 13.0 build: needs driver 580+, else use the cu126 fallback in `requirements/linux-cuda.txt` (resolves, not run on a GPU) |
| Colab | separate path: `scripts/run_colab.sh` | CUDA | uses Colab's own torch, not `.venv` |
| Linux aarch64, Intel Mac, Windows | not supported | | |

Device selection is centralised in `scripts/_common.py` (`pick_device()`): cuda, then mps, then
cpu. `VIBE_TTS_DEVICE=cpu|mps|cuda` forces one. Same seed ≠ same audio across backends.
`speak.py --compile` (static cache, bf16) is CUDA-only and ignored elsewhere.

## Dependency files
| File | Contents |
|---|---|
| `requirements/base.txt` | every platform-independent package, exactly pinned (direct deps first, then transitive) |
| `requirements/mac-arm64.txt` | torch 2.14.1, torchaudio 2.11.0 from PyPI (MPS included) |
| `requirements/linux-cpu.txt` | `+cpu` wheels of the same versions from the PyTorch CPU index |
| `requirements/linux-cuda.txt` | PyPI Linux wheels of the same versions (CUDA 13.0 runtime as `nvidia-*` wheels) |

`base.txt` was derived from the original macOS install and is identical to it once torch is added.
For Linux x86_64 it was verified with `uv pip compile --python-platform x86_64-unknown-linux-gnu`
for both Linux files: the resolver changes none of its pins. Only `argbind`, `randomname` and
`descript-audiotools` build from source (pure Python). torchaudio 2.11.0 is the newest release on PyPI and declares no torch requirement, so it pairs
with torch 2.14.1 on every platform. `accelerate` is a training-only extra of
`parler-tts` and is deliberately not installed.

To bump a dependency, edit the pin, then re-run that `uv pip compile` check for `linux-cpu`,
`linux-cuda` and a real macOS install before committing.

`setup.sh` passes `--index-strategy unsafe-best-match` so uv can take `torch+cpu` from the PyTorch
index and everything else from PyPI. Every package is pinned with `==`, which bounds the risk.

## Pinned sources
| Component | Source | Revision | Enforced by |
|---|---|---|---|
| Code | https://github.com/huggingface/parler-tts (cloned to `third_party/parler-tts`, unmodified) | `d108732cd57788ec86bc857d99a6cabd66663d68` (2024-12-10) | `scripts/config.sh` → `setup.sh`, `remote_speak.sh` |
| Checkpoint | https://huggingface.co/parler-tts/parler-tts-mini-v1 (Apache-2.0) | `0392b9451a601e528fd863bbb0598431fee810d9` | `scripts/download_models.py` |
| descript-audiotools | https://github.com/descriptinc/audiotools (upstream dependency) | `348ebf2034ce24e2a91a553e3171cb00c0c71678` | `requirements/base.txt` |
| Python | uv-managed | 3.11 | `scripts/config.sh` |

The DAC audio codec and flan-t5 text encoder are bundled in the checkpoint's `model.safetensors`:
local runs need no other Hugging Face repo (verified with an empty `HF_HOME` and
`HF_HUB_OFFLINE=1`).

## Configuration
Everything is optional. Copy `.env.example` to `.env` (gitignored); variables already set in your
shell win over `.env`.

| Variable | Default | Use |
|---|---|---|
| `VIBE_TTS_MODEL_DIR` | `./models` | weights directory; point at external storage if needed. It must exist already (the model goes in `<dir>/parler-tts-mini-v1`) |
| `VIBE_TTS_OUTPUT_DIR` | `./outputs` | where audio is written |
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face cache/token location |
| `VIBE_TTS_DEVICE` | auto | force `cuda`, `mps` or `cpu` |

Moving an existing `pretrained_models/` directory over is just
`VIBE_TTS_MODEL_DIR=<that directory>`: the layout inside is unchanged.

## Training environment (Expresso fine-tune)
Separate from `.venv`, which is never modified by it. `scripts/setup_train.sh` builds `.venv-train` from
`requirements/base.txt` **without its `fsspec` line**, the platform torch file, and `requirements/train.txt`.

Why separate: `datasets` 3.6.0 needs `fsspec<=2025.3.0` while `base.txt` pins `fsspec==2026.9.0`, a pin the
inference setup was verified with. `datasets` stays below 4.0 because 4.x decodes audio via torchcodec and no
longer returns the `{"array": ...}` dict that the pinned upstream training script indexes.

| Package | Inference `.venv` | Training `.venv-train` / Colab |
|---|---|---|
| transformers | 4.46.1 | 4.46.1 (unchanged; `USE_TF=0` is set by every script that needs it) |
| torch / torchaudio | 2.14.1 / 2.11.0 | 2.14.1 / 2.11.0 locally; Colab uses its own torch |
| huggingface-hub, numpy, protobuf | 0.36.2, 2.4.6, 4.25.9 | same |
| fsspec | 2026.9.0 | 2025.3.0 |
| accelerate, datasets, evaluate, jiwer, wandb | not installed | 1.15.0, 3.6.0, 0.4.6, 4.0.0, 0.28.0 |
| parler-tts | `d108732cd577` (editable, `--no-deps`) | same checkout |
| model | `parler-tts-mini-v1` @ `0392b9451a60` | same (`REVISION.json` records it) |

`wandb` is only installed because the upstream `training/utils.py` imports it unconditionally; runs use
tensorboard with `WANDB_MODE=disabled` and `HF_HUB_DISABLE_TELEMETRY=1`. A real fine-tune needs a CUDA GPU
(`scripts/run_colab_train.sh`); the Mac environment is for data preparation and pipeline smoke tests.

## Colab
`scripts/run_colab.sh` (wraps `scripts/remote_speak.sh`) is entirely separate from the local setup
and nothing local depends on it. It uses Colab's preinstalled torch and installs the pinned
`parler-tts` commit and `transformers==4.46.1` on the VM, downloads the pinned model there, runs a
script, fetches the wavs to `<output dir>/colab/` and always stops the VM.
- Needs the `colab` CLI (`uv tool install google-colab-cli`) signed in to a Google account with
  Colab GPU access; check with `colab usage`. `GPU=L4|T4|A100|H100|G4` picks the accelerator.
- No Hugging Face token is needed (the model is public) and `.env` is not uploaded.
- It does not use the `requirements/` files, so Colab's torch version is whatever Colab ships.

## Provenance
No Chinese-developed or owned model, service or component is used: Parler-TTS (Hugging Face),
DAC (Descript), flan-t5 (Google), transformers/torch. `requirements/base.txt` has no known
Chinese-origin packages (checked by name for modelscope, funasr, paddle, jieba, etc.).

## Layout
Upstream code stays in `third_party/` (gitignored, never edited). Our code lives in `scripts/`;
`scripts/_common.py` holds the shared paths/device helpers and `scripts/config.sh` the shared
shell config (pins, `.env`). `models/` and `outputs/` are tracked only as empty dirs.
