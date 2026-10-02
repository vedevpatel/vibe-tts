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
