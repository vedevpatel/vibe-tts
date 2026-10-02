# Parler-TTS setup

Reproduce: `scripts/setup_parler.sh`, then `.venv/bin/python scripts/smoke_test.py`.

## Environment (verified)
- macOS 15.5, Apple M4 (arm64), no NVIDIA GPU; inference runs on PyTorch MPS
- Python 3.11 (uv-managed `.venv` at project root), torch 2.14.1, torchaudio 2.11.0, transformers 4.46.1 (upstream pin)
- Full pinned dependency set: `requirements.lock`

## Pinned sources
| Component | Source | Revision |
|---|---|---|
| Code | https://github.com/huggingface/parler-tts (cloned to `third_party/parler-tts`, unmodified) | `d108732cd57788ec86bc857d99a6cabd66663d68` (2024-12-10) |
| Checkpoint | https://huggingface.co/parler-tts/parler-tts-mini-v1 (Apache-2.0) | `0392b9451a601e528fd863bbb0598431fee810d9` |
| Audio codec (loaded by checkpoint) | https://huggingface.co/parler-tts/dac_44khZ_8kbps (MIT; Descript Audio Codec) | `db52bea859d9411e0beb44a3ea923a8731ee4197` |
| Text encoder (loaded by checkpoint) | google/flan-t5-large (Google) | not pinned yet |
| descript-audiotools | https://github.com/descriptinc/audiotools (upstream dependency) | `348ebf2034ce24e2a91a553e3171cb00c0c71678` |

## Provenance
No Chinese-developed or owned model, service or component is used: Parler-TTS (Hugging Face),
DAC (Descript), flan-t5 (Google), transformers/torch. `requirements.lock` has no known
Chinese-origin packages (checked by name for modelscope, funasr, paddle, jieba, etc.).

## Layout
Upstream code stays in `third_party/` (gitignored, never edited). Our code lives in `scripts/`.
