# vibe-tts

Text-to-speech experiments built on [Parler-TTS](https://github.com/huggingface/parler-tts)
(`parler-tts/parler-tts-mini-v1`).

## Setup

```
git clone <repo-url> vibe-tts && cd vibe-tts
./scripts/setup.sh
.venv/bin/python scripts/smoke_test.py
```

That is the whole setup. The only prerequisites are [`uv`](https://docs.astral.sh/uv/) and `git`;
no Conda, no system Python changes, no `sudo`. `setup.sh` creates a repo-local `.venv`, installs the
pinned dependencies for your platform, checks out the pinned `parler-tts` source into
`third_party/`, and downloads the model (about 3.3 GB) into `models/`. It is safe to re-run.

| Platform | Supported | Backend |
|---|---|---|
| macOS arm64 (macOS 14+) | yes, tested | MPS (CPU fallback) |
| Linux x86_64, no NVIDIA GPU (e.g. Ubuntu 26.04) | yes | CPU |
| Linux x86_64 + NVIDIA GPU | supported, not yet run on hardware | CUDA |
| Google Colab | separate path, see below | CUDA |

## Everyday use

```
scripts/run.sh                          # built-in test sentences, then play them
scripts/run.sh "Say this." "And this."  # your own text (NOPLAY=1 to skip playback)
scripts/run.sh --style sad "Say this."  # the Expresso emotion fine-tune: default, sad or whisper (noncommercial, see below)
scripts/run.sh --style whisper --check "Say this."  # also check each clip with Whisper ASR and regenerate garbled ones
```

Audio is written to `outputs/`. Per-machine settings (model location, output location, device)
go in an optional, gitignored `.env`; copy `.env.example`.

## Run on a Colab GPU

```
scripts/run_colab.sh "Say this."
```

Optional and independent of the local setup. Needs the `colab` CLI; see
[SETUP.md](SETUP.md#colab).

See [SETUP.md](SETUP.md) for how the environment is pinned, platform notes, configuration,
model provenance, and how to bump dependencies. Upstream code lives in `third_party/`
(gitignored, unmodified); our code is in `scripts/`.

## Expresso emotion fine-tuning (experimental, noncommercial)

A one-speaker fine-tune of Parler-TTS on Expresso's default, sad and whispered read speech, with blinded
listening checks. `run.sh --style default|sad|whisper` speaks with it (the checkpoint must have been
trained and fetched first; it lives with the experiment data); without `--style`, `run.sh` and
`run_colab.sh` are unchanged. The dataset is CC BY-NC 4.0, so any checkpoint trained on it is
noncommercial, and the voice is a synthetic copy of a real speaker: do not present it as real.
See [EXPRESSO.md](EXPRESSO.md) for the workflow, provenance and results.

