"""Compare the upstream README inference path against ours, to look for an inference bug.

Usage: .venv/bin/python scripts/reference_check.py [variant ...]   (default: all)
Uses the README's named-speaker description and a longer ordinary paragraph. No compile,
fp32, and generate() with its default settings (generation_config.json). Waveforms are
saved untouched: outputs/ref_<variant>.wav (soundfile default, as upstream) and .npy (raw float32).

Variants (same seed each):
  upstream    default attention (SDPA), no attention masks -- exactly the README snippet
  ours        eager attention + attention masks            -- what scripts/speak.py does
  eager_only  eager attention, no masks
  masks_only  default attention (SDPA), with masks
"""
import sys
import time

import numpy as np
import soundfile as sf
import torch
from _common import model_dir, output_dir, pick_device  # before transformers: loads .env (HF_HOME)
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from transformers.cache_utils import StaticCache

if not hasattr(StaticCache, "max_batch_size"):
    StaticCache.max_batch_size = property(lambda self: self.batch_size)

MODEL_DIR = model_dir()
OUT_DIR = output_dir()

# Verbatim from the model card's "Using a specific speaker" section.
DESCRIPTION = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."
PROMPT = (
    "The morning train was a little late, so I stood on the platform and watched the sky turn from grey to pale gold. "
    "A man nearby was reading a newspaper, and a child kept asking her mother why the pigeons never seemed to be in a hurry. "
    "When the train finally arrived, everyone climbed aboard quietly, found a seat, and settled in for the ride into the city."
)
SEED = 0
VARIANTS = {  # name: (attn_implementation or None for library default, pass masks)
    "upstream": (None, False),
    "ours": ("eager", True),
    "eager_only": ("eager", False),
    "masks_only": (None, True),
}

device = pick_device()
print(f"device={device}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tok = lambda t: tokenizer(t, return_tensors="pt").to(device)
desc, prompt = tok(DESCRIPTION), tok(PROMPT)
print(f"prompt tokens={prompt.input_ids.shape[1]} description tokens={desc.input_ids.shape[1]}")

OUT_DIR.mkdir(exist_ok=True)
for name, (attn, masks) in VARIANTS.items():
    if sys.argv[1:] and name not in sys.argv[1:]:
        continue
    kw = {"attn_implementation": attn} if attn else {}
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_DIR, **kw).to(device)
    gen_kw = dict(input_ids=desc.input_ids, prompt_input_ids=prompt.input_ids)
    if masks:
        gen_kw.update(attention_mask=desc.attention_mask, prompt_attention_mask=prompt.attention_mask)
    torch.manual_seed(SEED)
    start = time.time()
    generation = model.generate(**gen_kw)
    audio = generation.cpu().numpy().squeeze()
    sr = model.config.sampling_rate
    sf.write(OUT_DIR / f"ref_{name}.wav", audio, sr)
    np.save(OUT_DIR / f"ref_{name}.npy", audio)
    print(f"speak_{name}: attn={model.config._attn_implementation} {len(audio) / sr:.2f}s dtype={audio.dtype} "
          f"peak={np.abs(audio).max():.3f} rms={np.sqrt((audio ** 2).mean()):.4f} nan={bool(np.isnan(audio).any())} in {time.time() - start:.1f}s")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
