"""Smoke test: load parler-tts-mini-v1 (pinned revision) and write one clip."""
import time
from pathlib import Path

import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

MODEL_ID = "parler-tts/parler-tts-mini-v1"
MODEL_REVISION = "0392b9451a601e528fd863bbb0598431fee810d9"
OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "smoke_test.wav"

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(device)

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
