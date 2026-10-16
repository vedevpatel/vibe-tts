"""Synthesize sentences with the local parler-tts-mini-v1 weights (see scripts/download_models.py).

Usage: .venv/bin/python scripts/speak.py [--compile] [--style default|sad|whisper|enunciated [--check]] ["sentence one" "sentence two" ...]
Writes speak_<n>.wav to the output dir (VIBE_TTS_OUTPUT_DIR, default outputs/). With no sentences, runs a few built-in test sentences.

--style speaks with the Expresso emotion fine-tune instead (one speaker's voice; default, sad, whisper or enunciated; noncommercial,
see EXPRESSO.md), using the descriptions and generation settings it was evaluated with. A clip that is cut off or far
too short or long is generated again; --check also transcribes each clip with Whisper and regenerates garbled ones.
Whisper text is spoken in pieces of at most 14 words joined with pauses (long whispers otherwise often stop early).

--compile (CUDA only) compiles the forward pass with a static cache. It costs ~30s of
warm-up per process and pads every input to a fixed length, but runs near real time on
an L4 once warm. Worth it for many sentences in one run; not for a single one.
"""
import argparse
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
OUT_DIR = output_dir()

DESCRIPTION = "A female speaker delivers a slightly expressive speech at a moderate pace, in a very clear, close-sounding recording."
DEFAULT_SENTENCES = [
    "Hello, this is a test of the text to speech model.",
    "I thought you'd forgotten.",
    "The quick brown fox jumps over the lazy dog.",
]

parser = argparse.ArgumentParser()
parser.add_argument("sentences", nargs="*")
parser.add_argument("--compile", action="store_true", help="torch.compile with a static cache (CUDA only)")
parser.add_argument("--style", help="speak with the Expresso emotion fine-tune in this style (default, sad, whisper or enunciated): Expresso "
                    "speaker ex02's voice, noncommercial use only, see EXPRESSO.md. Which checkpoint: [speak] in "
                    "configs/expresso_emotion.toml")
parser.add_argument("--check", action="store_true", help="with --style: also transcribe each clip with Whisper and generate it "
                    "again if words are missing or garbled (auto_evaluate.py completeness's rule; a few seconds per clip)")
args = parser.parse_args()
sentences = args.sentences or DEFAULT_SENTENCES
if args.check and not args.style:
    raise SystemExit("--check only applies with --style")

if args.style:
    # The fine-tune: splitting, retries and --check live in styled_voice.py (shared with serve.py, the local page).
    if args.compile:
        raise SystemExit("--style uses the float32 settings the fine-tune was evaluated with; --compile is not supported with it")
    from styled_voice import StyledVoice

    voice = StyledVoice()
    if args.style not in voice.styles:
        raise SystemExit(f"style {args.style!r} not available; this checkpoint has {voice.styles}")
    print(f"style={args.style}: {voice.notice()}")
    print(f"device={voice.device}")
    OUT_DIR.mkdir(exist_ok=True)
    for i, text in enumerate(sentences):
        path = OUT_DIR / f"speak_{i}.wav"
        start = time.time()
        audio = voice.speak(text, args.style, check=args.check, log=lambda msg: print(f"{path.name}: {msg}"))
        sf.write(path, audio, voice.sr)
        print(f"{path.name}: {len(audio) / voice.sr:.2f}s audio in {time.time() - start:.1f}s  <- {text!r}")
    raise SystemExit(0)

device = pick_device()
compiled = args.compile
if compiled and device != "cuda":
    print(f"--compile is only supported on CUDA (device={device}); ignoring")
    compiled = False
dtype = torch.bfloat16 if compiled else torch.float32
print(f"device={device} compile={compiled}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# eager attention: the T5 text encoder has no SDPA support
model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_DIR, attn_implementation="eager").to(device, dtype=dtype)

if compiled:
    model.generation_config.cache_implementation = "static"
    model.forward = torch.compile(model.forward, mode="reduce-overhead")

# Compiled graphs need fixed shapes: pad the description and every prompt to one length.
pad = dict(return_tensors="pt", padding="max_length") if compiled else dict(return_tensors="pt")
desc_len = prompt_len = None
if compiled:
    desc_len = 48
    longest = max(len(tokenizer(s).input_ids) for s in sentences)
    prompt_len = max(32, -(-longest // 16) * 16)
desc = tokenizer(DESCRIPTION, max_length=desc_len, **pad).to(device)


def synth(text):
    prompt = tokenizer(text, max_length=prompt_len, **pad).to(device)
    out = model.generate(
        input_ids=desc.input_ids,
        attention_mask=desc.attention_mask,
        prompt_input_ids=prompt.input_ids,
        prompt_attention_mask=prompt.attention_mask,
    )
    return out.cpu().float().numpy().squeeze()


if compiled:
    start = time.time()
    for _ in range(2):  # reduce-overhead needs two generations before the speed-up shows
        synth("This is for compilation")
    print(f"warm-up: {time.time() - start:.1f}s")


OUT_DIR.mkdir(exist_ok=True)
for i, text in enumerate(sentences):
    path = OUT_DIR / f"speak_{i}.wav"
    start = time.time()
    audio = synth(text)
    sf.write(path, audio, model.config.sampling_rate)
    print(f"{path.name}: {len(audio) / model.config.sampling_rate:.2f}s audio in {time.time() - start:.1f}s  <- {text!r}")
