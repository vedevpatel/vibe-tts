# vibe-tts

Text-to-speech experiments built on [Parler-TTS](https://github.com/huggingface/parler-tts)
(`parler-tts/parler-tts-mini-v1`).

## Quick start

```
scripts/setup_parler.sh
.venv/bin/python scripts/smoke_test.py
```

See [SETUP.md](SETUP.md) for environment details, pinned revisions and model provenance.
Upstream code lives in `third_party/` (gitignored, unmodified); our code is in `scripts/`.

## Run on a Colab GPU

```
scripts/run.sh                         # built-in test sentences
scripts/run.sh "Say this." "And this." # your own text
```

Starts an L4 session, synthesizes, downloads to `outputs/colab/`, plays the audio and
stops the VM. Options and env vars are documented in `scripts/remote_speak.sh`.
