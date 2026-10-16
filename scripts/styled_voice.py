"""The Expresso emotion fine-tune as a reusable voice: load once, then speak any text in any of its styles.

Used by speak.py --style (one process per run) and serve.py (the local page, one model kept loaded). The checkpoint is
the [speak] run in configs/expresso_emotion.toml; its styles and descriptions come from its export_info.json, and the
generation settings are the ones it was evaluated with (float32). Expresso speaker ex02's voice: CC BY-NC 4.0 data,
noncommercial use only, and a synthetic voice of a real person; do not present it as real (EXPRESSO.md).

How a text is spoken:
- Whisper text is cut into pieces of at most MAX_WORDS words (long whispers otherwise often stop early: 12 of 20 long
  ones on v3, 0 of 20 short; EXPRESSO.md), each generated on its own and joined with short pauses.
- A clip that is cut off, far too short or far too long for its text is generated again, up to RETRIES times. With
  check=True, Whisper's transcript is checked too (last words missing, or 25%+ of the words wrong), exactly as
  auto_evaluate.py completeness flags clips. If every try has a problem, the least bad one is kept.
"""
import re
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from _common import model_dir, pick_device  # before transformers: loads .env (HF_HOME)
from expresso_common import TOO_LONG, Config, audio_problem, read_json, speaking_pace, strip_emphasis_markers
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from transformers.cache_utils import StaticCache

# parler-tts reads StaticCache.max_batch_size, which transformers 4.46.1 renamed to batch_size.
if not hasattr(StaticCache, "max_batch_size"):
    StaticCache.max_batch_size = property(lambda self: self.batch_size)

RETRIES = 2   # extra tries for a clip with a problem
SPLIT_STYLES, MAX_WORDS = {"whisper"}, 14
PAUSE_S = {",": 0.25, ";": 0.3, ":": 0.3, ".": 0.45, "!": 0.45, "?": 0.45}   # after a piece ending in this mark
BREAK_BEFORE = {"and", "but", "or", "so", "because", "until", "before", "after", "while", "when", "where", "who", "which",
                "that", "if", "though", "although", "unless", "since", "with", "without", "to", "from", "into", "at", "on", "in"}


def bisect_words(words):
    """Halve a clause with no punctuation until every part fits: before a conjunction or preposition nearest the middle
    (at least 3 words on each side), else at the middle."""
    if len(words) <= MAX_WORDS:
        return [words]
    mid = len(words) // 2
    cuts = [k for k in range(3, len(words) - 2) if words[k].lower() in BREAK_BEFORE]
    k = min(cuts, key=lambda c: abs(c - mid)) if cuts else mid
    return bisect_words(words[:k]) + bisect_words(words[k:])


def pieces(text):
    """Split text into pieces of at most MAX_WORDS words: at sentence ends, then at commas/semicolons/colons (adjacent
    clauses merged while they fit), and an over-long clause with bisect_words."""
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        cur = []
        for clause in re.split(r"(?<=[,;:])\s+", sentence):
            words = clause.split()
            if len(words) > MAX_WORDS:
                if len(cur) <= 2:   # a lone "So," or "Well," stays with what follows
                    words, cur = cur + words, []
                if cur:
                    out.append(" ".join(cur))
                    cur = []
                out += [" ".join(p) for p in bisect_words(words)]
            elif len(cur) + len(words) > MAX_WORDS:
                out.append(" ".join(cur))
                cur = words
            else:
                cur += words
        if cur:
            out.append(" ".join(cur))
    return out


class StyledVoice:
    def __init__(self, config=None):
        self.cfg = Config(config)
        self.cfg.require_data_root()
        self.run = self.cfg["speak"]["run"]
        path = self.cfg.path("runs") / self.run / "export"
        if not (path / "export_info.json").is_file():
            raise SystemExit(f"no fine-tuned checkpoint at {path} (see EXPRESSO.md)")
        self.info = read_json(path / "export_info.json")
        self.descriptions = self.info["descriptions"]   # exactly what it was trained and evaluated with
        self.styles = list(self.descriptions)
        g = self.cfg["eval"]["generation"]
        self.gen_kwargs = {k: g[k] for k in ("do_sample", "temperature", "max_length", "min_new_tokens")}
        self.pace = speaking_pace(self.cfg)
        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir())
        # eager attention: the T5 text encoder has no SDPA support
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(path, attn_implementation="eager").to(self.device, dtype=torch.float32)
        self.sr = self.model.config.sampling_rate
        self.measures = None   # Whisper, loaded on the first checked clip
        self.tmp = Path(tempfile.mkdtemp(prefix="styled_voice_"))

    def notice(self):
        return (f"{self.run} fine-tune, Expresso speaker {self.info['speaker']}. Trained on CC BY-NC 4.0 data: noncommercial "
                f"use only, and a synthetic voice of a real person; do not present it as real.")

    def synth(self, text, style):
        # Cap the length at what audio_problem would reject as too long anyway (at least 3 s): on very short text ("Hi")
        # the model sometimes never stops and babbles to the 30 s limit, which takes minutes per try on a Mac.
        expected = len(strip_emphasis_markers(text)) / self.pace[style]
        frames = int(self.model.config.audio_encoder.frame_rate * max(3.0, TOO_LONG * expected + 1.5))
        kwargs = self.gen_kwargs | {"max_length": min(self.gen_kwargs["max_length"], frames + 10)}   # +10: codebook delays
        desc = self.tokenizer(self.descriptions[style], return_tensors="pt").to(self.device)
        prompt = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model.generate(input_ids=desc.input_ids, attention_mask=desc.attention_mask,
                                  prompt_input_ids=prompt.input_ids, prompt_attention_mask=prompt.attention_mask, **kwargs)
        return out.cpu().float().numpy().squeeze()

    def problem_with(self, path, text, style, check):
        """(severity, why, heard): severity 0 = fine, 1 = complete but worth a listen (garbled, or far too long),
        2 = incomplete; heard = Whisper's transcript (None without check)."""
        if not check:
            problem = audio_problem(path, text, style, self.pace)
            return (0, None, None) if not problem else (1 if problem[0] == "too_long" else 2, problem[1], None)
        if self.measures is None:
            from auto_evaluate import Measures

            self.measures = Measures(self.cfg)
        from auto_evaluate import completeness_flags

        m = self.measures
        heard = m.transcript(path)
        flag, detail = completeness_flags(m.normalize_text(text), m.normalize_text(heard), path, text, style, self.pace)
        return {None: 0, "check": 1, "incomplete": 2}[flag], flag and f"{detail} (heard {heard!r})", heard

    def generate(self, text, style, check, log):
        """One clip for text, regenerated on a problem; the least bad try if all have one. A complete clip that Whisper
        hears exactly as it heard the try before is kept: the words are consistent, so the text itself is written
        differently from how it is heard ("can not" / "cannot"), and another try would only repeat it."""
        path = self.tmp / "try.wav"
        best, last_heard = None, None
        for attempt in range(1 + RETRIES):
            audio = self.synth(text, style)
            sf.write(path, audio, self.sr)
            severity, problem, heard = self.problem_with(path, text, style, check)
            if best is None or severity < best[0]:
                best = (severity, attempt, audio)
            if not problem:
                break
            if severity == 1 and heard is not None and heard == last_heard:
                log(f"{problem}; heard the same twice, so kept (the text and Whisper's spelling differ)")
                break
            last_heard = heard if severity == 1 else None
            log(f"{problem}; " + ("generating again" if attempt < RETRIES else f"kept try {best[1] + 1} of {1 + RETRIES}"))
        return best[2]

    def speak(self, text, style, check=False, log=print):
        """Audio (float32, self.sr) for text in style."""
        if style not in self.descriptions:
            raise ValueError(f"style {style!r} not available; this checkpoint has {self.styles}")
        parts = pieces(text) if style in SPLIT_STYLES else [text.strip()]
        if len(parts) > 1:
            log(f"spoken in {len(parts)} pieces: " + " | ".join(parts))
        joined = []
        for n, part in enumerate(parts):
            audio = self.generate(part, style, check, log)
            joined.append(audio)
            if n < len(parts) - 1:
                joined.append(np.zeros(int(PAUSE_S.get(part[-1], 0.15) * self.sr), dtype=audio.dtype))
        return np.concatenate(joined)
