"""Baseline acting test: the reference paragraph and Jon description, only the delivery phrase changes.

Usage: .venv/bin/python scripts/acting_baseline.py
Writes outputs/base_<delivery>_s<seed>.wav. Two seeds per delivery to check consistency.
No compile, fp32, default generate() settings.
"""
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

# Same paragraph as scripts/reference_check.py.
PROMPT = (
    "The morning train was a little late, so I stood on the platform and watched the sky turn from grey to pale gold. "
    "A man nearby was reading a newspaper, and a child kept asking her mother why the pigeons never seemed to be in a hurry. "
    "When the train finally arrived, everyone climbed aboard quietly, found a seat, and settled in for the ride into the city."
)
TEMPLATE = "Jon's voice is {} in delivery, with a very close recording that almost has no background noise."
DELIVERIES = {
    "warm": "warm and reassuring",
    "sad": "sad and subdued",
    "angry": "angry and forceful",
}
SEEDS = [0, 1]

device = pick_device()
print(f"device={device}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
prompt = tokenizer(PROMPT, return_tensors="pt").to(device)

OUT_DIR.mkdir(exist_ok=True)
for label, phrase in DELIVERIES.items():
    description = TEMPLATE.format(phrase)
    desc = tokenizer(description, return_tensors="pt").to(device)
    for seed in SEEDS:
        torch.manual_seed(seed)
        start = time.time()
        audio = model.generate(input_ids=desc.input_ids, prompt_input_ids=prompt.input_ids).cpu().numpy().squeeze()
        sr = model.config.sampling_rate
        sf.write(OUT_DIR / f"base_{label}_s{seed}.wav", audio, sr)
        print(f"speak_{label}_s{seed}: {len(audio) / sr:.2f}s rms={np.sqrt((audio ** 2).mean()):.4f} in {time.time() - start:.1f}s <- {description!r}")
print("ALLDONE")
