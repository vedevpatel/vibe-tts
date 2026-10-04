"""Benchmark Parler-TTS inference variants on the local weights.

Usage: python scripts/bench_speed.py VARIANT
  VARIANT: fp32 | bf16 | compile_default | compile_reduce
Each variant should run in its own process (compile state is per-process).
Prints real-time factor (seconds of audio per second of wall time; >1 is faster than real time).
"""
import sys
import time

import soundfile as sf
import torch
from _common import model_dir, output_dir, pick_device  # before transformers: loads .env (HF_HOME)
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from transformers.cache_utils import StaticCache

# parler-tts reads StaticCache.max_batch_size, which transformers 4.46.1 renamed to batch_size.
if not hasattr(StaticCache, "max_batch_size"):
    StaticCache.max_batch_size = property(lambda self: self.batch_size)

MODEL_DIR = model_dir()
OUT_DIR = output_dir() / "bench"

DESCRIPTION = "A female speaker delivers a slightly expressive speech at a moderate pace, in a very clear, close-sounding recording."
SENTENCES = [
    "Hello, this is a test of the text to speech model.",
    "I thought you'd forgotten.",
    "The quick brown fox jumps over the lazy dog.",
]
# compiled graphs need fixed input shapes, so pad every input to these lengths
DESC_LEN, PROMPT_LEN = 48, 32

variant = sys.argv[1]
device = pick_device()
dtype = torch.float32 if variant == "fp32" else torch.bfloat16
compiled = variant.startswith("compile")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_DIR, attn_implementation="eager"  # the T5 text encoder has no SDPA support
).to(device, dtype=dtype)

if compiled:
    model.generation_config.cache_implementation = "static"
    model.forward = torch.compile(model.forward, mode="reduce-overhead" if variant == "compile_reduce" else "default")

pad = dict(return_tensors="pt", padding="max_length")
desc = tokenizer(DESCRIPTION, max_length=DESC_LEN, **pad).to(device)


def synth(text):
    prompt = tokenizer(text, max_length=PROMPT_LEN, **pad).to(device)
    out = model.generate(
        input_ids=desc.input_ids,
        attention_mask=desc.attention_mask,
        prompt_input_ids=prompt.input_ids,
        prompt_attention_mask=prompt.attention_mask,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    return out.cpu().float().numpy().squeeze()


print(f"variant={variant} device={device} dtype={dtype}")
t = time.time()
for _ in range(1 if variant == "compile_default" else 2 if compiled else 1):
    synth("This is for compilation")
print(f"warmup: {time.time() - t:.1f}s")

OUT_DIR.mkdir(parents=True, exist_ok=True)
total_audio = total_time = 0.0
repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 1
for i, text in enumerate(SENTENCES * repeats):
    t = time.time()
    audio = synth(text)
    dt = time.time() - t
    secs = len(audio) / model.config.sampling_rate
    total_audio += secs
    total_time += dt
    sf.write(OUT_DIR / f"{variant}_{i % len(SENTENCES)}.wav", audio, model.config.sampling_rate)
    print(f"  {secs:.2f}s audio in {dt:.1f}s  <- {text!r}")
print(f"RESULT {variant}: {total_audio / total_time:.2f}x real time ({total_time:.1f}s for {total_audio:.1f}s audio)")
