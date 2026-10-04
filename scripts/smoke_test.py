"""Smoke test: load the local parler-tts-mini-v1 weights (scripts/download_models.py) and write one clip."""
import time

import soundfile as sf
from _common import model_dir, output_dir, pick_device  # first: loads .env before transformers reads HF_HOME
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

MODEL_DIR = model_dir()
OUTPUT = output_dir() / "smoke_test.wav"

device = pick_device()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)

prompt = "I thought you'd forgotten."
description = "A female speaker delivers a slightly expressive speech at a moderate pace, in a very clear, close-sounding recording."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

start = time.time()
generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio = generation.cpu().float().numpy().squeeze()
OUTPUT.parent.mkdir(exist_ok=True)
sf.write(OUTPUT, audio, model.config.sampling_rate)
print(f"device={device} wrote {OUTPUT} ({len(audio) / model.config.sampling_rate:.2f}s audio in {time.time() - start:.1f}s)")
