"""Does Parler-TTS follow acting direction? One speaker, one line, four deliveries.

Usage: .venv/bin/python scripts/acting_test.py
Writes outputs/acting_<label>.wav. Only the delivery description changes between takes;
speaker name, recording-quality words, text and seed are held fixed.
"""
import time

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

TEXT = "It's okay. I already knew."
SPEAKER = "Laura's voice"
QUALITY = "The recording is very clear and close-sounding, with no background noise."
TAKES = {
    "neutral": "speaks in a neutral, conversational tone at a moderate pace, with natural, flat intonation.",
    "warm": "speaks in a warm, soft, reassuring tone, slowly and gently, as if comforting someone she cares about.",
    "hurt": "speaks in a quiet, shaky, trembling voice, slowly, with hesitation, as if holding back tears.",
    "cold": "speaks in a cold, tight, restrained voice, clipped and monotone, with controlled anger, trying to sound completely unaffected.",
}
SEED = 0

device = pick_device()
print(f"device={device}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_DIR, attn_implementation="eager").to(device)
prompt = tokenizer(TEXT, return_tensors="pt").to(device)

OUT_DIR.mkdir(exist_ok=True)
for label, direction in TAKES.items():
    description = f"{SPEAKER} {direction} {QUALITY}"
    desc = tokenizer(description, return_tensors="pt").to(device)
    torch.manual_seed(SEED)
    start = time.time()
    out = model.generate(
        input_ids=desc.input_ids,
        attention_mask=desc.attention_mask,
        prompt_input_ids=prompt.input_ids,
        prompt_attention_mask=prompt.attention_mask,
    )
    audio = out.cpu().float().numpy().squeeze()
    path = OUT_DIR / f"acting_{label}.wav"
    sf.write(path, audio, model.config.sampling_rate)
    print(f"speak_{label}: {len(audio) / model.config.sampling_rate:.2f}s audio in {time.time() - start:.1f}s  <- {description!r}")
