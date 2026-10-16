"""Synthesize sentences with the local parler-tts-mini-v1 weights (see scripts/download_models.py).

Usage: .venv/bin/python scripts/speak.py [--compile] [--style default|sad|whisper [--check]] ["sentence one" "sentence two" ...]
Writes speak_<n>.wav to the output dir (VIBE_TTS_OUTPUT_DIR, default outputs/). With no sentences, runs a few built-in test sentences.

--style speaks with the Expresso emotion fine-tune instead (one speaker's voice; default, sad or whisper; noncommercial,
see EXPRESSO.md), using the descriptions and generation settings it was evaluated with. A clip that is cut off or far
too short or long is generated again; --check also transcribes each clip with Whisper and regenerates garbled ones.

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
parser.add_argument("--style", help="speak with the Expresso emotion fine-tune in this style (default, sad or whisper): Expresso "
                    "speaker ex02's voice, noncommercial use only, see EXPRESSO.md. Which checkpoint: [speak] in "
                    "configs/expresso_emotion.toml")
parser.add_argument("--check", action="store_true", help="with --style: also transcribe each clip with Whisper and generate it "
                    "again if words are missing or garbled (auto_evaluate.py completeness's rule; a few seconds per clip)")
args = parser.parse_args()
sentences = args.sentences or DEFAULT_SENTENCES
if args.check and not args.style:
    raise SystemExit("--check only applies with --style")

RETRIES = 2   # --style only: extra tries for a clip that audio_problem() flags
model_path, description, gen_kwargs = MODEL_DIR, DESCRIPTION, {}
if args.style:
    from expresso_common import Config, audio_problem, read_json, speaking_pace

    if args.compile:
        raise SystemExit("--style uses the float32 settings the fine-tune was evaluated with; --compile is not supported with it")
    cfg = Config()
    cfg.require_data_root()
    model_path = cfg.path("runs") / cfg["speak"]["run"] / "export"
    info = read_json(model_path / "export_info.json") if (model_path / "export_info.json").is_file() else None
    if info is None:
        raise SystemExit(f"no fine-tuned checkpoint at {model_path} (see EXPRESSO.md)")
    if args.style not in info.get("descriptions", {}):
        raise SystemExit(f"style {args.style!r} not available; this checkpoint has {sorted(info.get('descriptions', {}))}")
    description = info["descriptions"][args.style]   # exactly what it was trained and evaluated with
    g = cfg["eval"]["generation"]
    gen_kwargs = {k: g[k] for k in ("do_sample", "temperature", "max_length", "min_new_tokens")}
    pace = speaking_pace(cfg)
    if args.check:
        from auto_evaluate import Measures, completeness_flags

        measures = Measures(cfg)   # Whisper loads on the first clip
    print(f"style={args.style}: {cfg['speak']['run']} fine-tune, Expresso speaker {info['speaker']}. Trained on CC BY-NC 4.0 "
          f"data: noncommercial use only, and a synthetic voice of a real person; do not present it as real.")

device = pick_device()
compiled = args.compile
if compiled and device != "cuda":
    print(f"--compile is only supported on CUDA (device={device}); ignoring")
    compiled = False
dtype = torch.bfloat16 if compiled else torch.float32
print(f"device={device} compile={compiled}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# eager attention: the T5 text encoder has no SDPA support
model = ParlerTTSForConditionalGeneration.from_pretrained(model_path, attn_implementation="eager").to(device, dtype=dtype)

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
desc = tokenizer(description, max_length=desc_len, **pad).to(device)


def synth(text):
    prompt = tokenizer(text, max_length=prompt_len, **pad).to(device)
    out = model.generate(
        input_ids=desc.input_ids,
        attention_mask=desc.attention_mask,
        prompt_input_ids=prompt.input_ids,
        prompt_attention_mask=prompt.attention_mask,
        **gen_kwargs,
    )
    return out.cpu().float().numpy().squeeze()


if compiled:
    start = time.time()
    for _ in range(2):  # reduce-overhead needs two generations before the speed-up shows
        synth("This is for compilation")
    print(f"warm-up: {time.time() - start:.1f}s")


def problem_with(path, text):
    """(severity, why) for a --style clip: 0 = fine, 1 = complete but worth a listen (garbled, or far too long),
    2 = incomplete (cut off, too short, last words missing). Without --check: the audio-only checks. With --check: those
    plus Whisper's transcript, exactly as auto_evaluate.py completeness flags clips."""
    if not args.check:
        problem = audio_problem(path, text, args.style, pace)
        return (0, None) if not problem else (1 if problem[0] == "too_long" else 2, problem[1])
    heard = measures.transcript(path)
    flag, detail = completeness_flags(measures.normalize_text(text), measures.normalize_text(heard), path, text, args.style, pace)
    return {None: 0, "check": 1, "incomplete": 2}[flag], flag and f"{detail} (heard {heard!r})"


OUT_DIR.mkdir(exist_ok=True)
for i, text in enumerate(sentences):
    path = OUT_DIR / f"speak_{i}.wav"
    # With --style, a clip that is cut off, far too short or far too long for its text is generated again (sampling makes
    # each try different). The fine-tune does this about 1 time in 100 (EXPRESSO.md); the checks need no transcript.
    # --check adds Whisper's transcript, for garbled or missing words (most of all in long whispers). If every try has a
    # problem, the least bad one is kept (complete but garbled over incomplete; the earliest on a tie).
    best = None
    for attempt in range(1 + (RETRIES if args.style else 0)):
        start = time.time()
        audio = synth(text)
        sf.write(path, audio, model.config.sampling_rate)
        severity, problem = problem_with(path, text) if args.style else (0, None)
        if best is None or severity < best[0]:
            best = (severity, attempt, audio)
        if not problem:
            break
        print(f"{path.name}: {problem}; " + ("generating again" if attempt < RETRIES else f"kept try {best[1] + 1} of {1 + RETRIES}"))
    if best[2] is not audio:
        audio = best[2]
        sf.write(path, audio, model.config.sampling_rate)
    print(f"{path.name}: {len(audio) / model.config.sampling_rate:.2f}s audio in {time.time() - start:.1f}s  <- {text!r}")
