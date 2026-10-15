"""Automatic stand-ins for the Expresso listening test, checked on real recordings and human ratings first.

Subcommands (all read configs/expresso_emotion.toml; run from any directory):

  download   Fetch the pinned evaluation models (EVAL_MODELS) into <model dir>/eval/.
  calibrate  Check every measure on REAL recordings only, before it is used on synthetic audio:
               style    a classifier over the configured styles (logistic regression on audEERING emotion
                        embeddings), trained on the speaker's train split and tested on validation + test;
                        plus one cue per style that needs no training (STYLE_CUES: low arousal for sad,
                        little voicing for whisper), checked on the other speakers too
               speaker  WavLM x-vector cosine similarity to the speaker's train-split centroid: held-out
                        recordings of the speaker against recordings of the other Expresso speakers
               words    Whisper word error rate on held-out recordings (what real speech scores)
             No naturalness measure: torchaudio's SQUIM predicted MOS was tried and gave 4.43-4.44 for every clip,
             real or synthetic, while a listener's naturalness ratings ranged from 2.4 to 4.5 (set-b).
             Writes <eval dir>/auto/calibration.md and the classifier.
  score      Apply the measures to generated runs (the same eval set as the listening pages), next to the
             real recordings of the same sentences. With --key and --ratings, also compare every measure
             with a listener's ratings of the same clips.
  completeness  Flag clips that stop early (the end of the text missing from Whisper's transcript, or far too short
             for the text) or need a listen, with the same rule first applied to real held-out recordings.

These are proxies for what a listener hears, not ratings. Each one is reported next to what real recordings
score, and the sadness measure is only meaningful if calibrate shows that it separates the real styles.
"""
import argparse
import json
import pickle
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from _common import model_root, pick_device  # first: loads .env before transformers reads HF_HOME
from expresso_common import (CUT_OFF_DB, TOO_LONG, TOO_SHORT, Config, audio_problem, log, parse_stem, read_json, sha256_file,
                             speaking_pace, strip_emphasis_markers, utc_now, write_json)

SR = 16000
# (repo, pinned revision, files, license). Nothing here is Chinese-developed (see EXPRESSO.md).
EVAL_MODELS = {
    "emotion": ("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", "6eba34a2485ea31cb03600241787c3a5edab8626",
                ["config.json", "preprocessor_config.json", "model.safetensors", "LICENSE", "README.md"],
                "CC BY-NC-SA 4.0 (noncommercial)"),
    "speaker": ("microsoft/wavlm-base-plus-sv", "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e",
                ["config.json", "preprocessor_config.json", "pytorch_model.bin", "README.md"],
                "not stated on the model card; WavLM is released by Microsoft under MIT"),
    "asr": ("openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
            ["config.json", "generation_config.json", "preprocessor_config.json", "model.safetensors", "added_tokens.json",
             "merges.txt", "normalizer.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
             "vocab.json", "README.md"],
            "MIT"),
}
REAL = "real recordings"
OTHER_SPEAKER_CLIPS = 15   # per other speaker and style, for the speaker and cross-speaker sadness checks


def eval_model_dir(name):
    return model_root() / "eval" / EVAL_MODELS[name][0].split("/")[1]


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def auto_dir(cfg):
    return cfg.path("eval") / "auto"


# ---------------------------------------------------------------- download
def cmd_download(cfg, args):
    from huggingface_hub import snapshot_download

    root = model_root()
    if not root.is_dir():  # an unmounted external drive must fail, not be recreated on the wrong disk
        raise SystemExit(f"model dir {root} does not exist")
    for name, (repo, rev, files, lic) in EVAL_MODELS.items():
        d = eval_model_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo, revision=rev, local_dir=d, allow_patterns=files)
        write_json(d / "REVISION.json", {"repo_id": repo, "revision": rev, "license": lic,
                                         "files": sorted(p.name for p in d.iterdir() if p.is_file() and p.name != "REVISION.json")})
        log(f"{name}: {repo}@{rev[:12]} -> {d}")


# ---------------------------------------------------------------- measures
class Measures:
    """Lazy-loaded evaluation models plus a per-file cache (keyed by the audio's sha256, so a regenerated
    clip is never scored from a stale entry)."""

    def __init__(self, cfg):
        import torch

        self.torch = torch
        self.device = pick_device()
        # pickle is safe here: cache.pkl and style_classifier.pkl are only ever written by this script, locally
        self.cache_path = auto_dir(cfg) / "cache.pkl"
        self.cache = pickle.loads(self.cache_path.read_bytes()) if self.cache_path.is_file() else {}
        self.models = {}
        for name in EVAL_MODELS:
            if not (eval_model_dir(name) / "REVISION.json").is_file():
                raise SystemExit(f"evaluation model '{name}' is missing: run scripts/auto_evaluate.py download")

    def save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_name(self.cache_path.name + ".tmp")
        tmp.write_bytes(pickle.dumps(self.cache))
        tmp.replace(self.cache_path)

    @staticmethod
    def load_16k(path):
        import librosa
        import numpy as np
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        return np.clip(audio, -1.0, 1.0).astype("float32")

    def _cached(self, measure, path, fn):
        key = (measure, sha256_file(path))
        if key not in self.cache:
            self.cache[key] = fn(self.load_16k(path))
        return self.cache[key]

    def _emotion_model(self):
        if "emotion" not in self.models:
            import torch.nn as nn
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2PreTrainedModel

            torch = self.torch

            class RegressionHead(nn.Module):  # as published on the model card
                def __init__(self, config):
                    super().__init__()
                    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                    self.dropout = nn.Dropout(config.final_dropout)
                    self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

                def forward(self, x):
                    return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

            class EmotionModel(Wav2Vec2PreTrainedModel):
                def __init__(self, config):
                    super().__init__(config)
                    self.wav2vec2 = Wav2Vec2Model(config)
                    self.classifier = RegressionHead(config)
                    self.init_weights()

                def forward(self, input_values):
                    pooled = self.wav2vec2(input_values)[0].mean(dim=1)
                    return pooled, self.classifier(pooled)

            d = eval_model_dir("emotion")
            model = EmotionModel.from_pretrained(d).to(self.device).eval()
            labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
            self.models["emotion"] = (Wav2Vec2FeatureExtractor.from_pretrained(d), model, labels)
        return self.models["emotion"]

    def emotion(self, path):
        """{'arousal', 'dominance', 'valence'} (about 0-1, as the model card describes) plus the pooled embedding."""
        def run(audio):
            fe, model, labels = self._emotion_model()
            x = fe(audio, sampling_rate=SR, return_tensors="pt").input_values.to(self.device)
            with self.torch.no_grad():
                pooled, out = model(x)
            rec = {lab: float(v) for lab, v in zip(labels, out[0].cpu())}
            rec["embedding"] = pooled[0].cpu().numpy()
            return rec
        return self._cached("emotion", path, run)

    def xvector(self, path):
        def run(audio):
            if "speaker" not in self.models:
                from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

                d = eval_model_dir("speaker")
                self.models["speaker"] = (Wav2Vec2FeatureExtractor.from_pretrained(d),
                                          WavLMForXVector.from_pretrained(d).to(self.device).eval())
            fe, model = self.models["speaker"]
            x = fe(audio, sampling_rate=SR, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                e = model(**x).embeddings[0]
            return self.torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
        return self._cached("xvector", path, run)

    def transcript(self, path):
        def run(audio):
            if "asr" not in self.models:
                from transformers import WhisperForConditionalGeneration, WhisperProcessor

                d = eval_model_dir("asr")
                self.models["asr"] = (WhisperProcessor.from_pretrained(d),
                                      WhisperForConditionalGeneration.from_pretrained(d).to(self.device).eval())
            proc, model = self.models["asr"]
            feats = proc(audio, sampling_rate=SR, return_tensors="pt").input_features.to(self.device)
            with self.torch.no_grad():
                ids = model.generate(feats, language="en", task="transcribe")
            return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return self._cached("transcript", path, run)

    def voicing(self, path):
        """Share of 16 ms frames with a detectable pitch (pyin, the same settings as the inventory and the description
        checks). Real ex02: default 67%, whisper 9%."""
        def run(audio):
            import librosa

            _, voiced, _ = librosa.pyin(audio, fmin=65, fmax=500, sr=SR, frame_length=1024, hop_length=256)
            return float(voiced.mean())
        return self._cached("voicing", path, run)

    def normalize_text(self, text):
        """Whisper's English normalizer (case, punctuation, spelling variants, numbers), as words."""
        if "normalizer" not in self.models:
            from transformers import WhisperTokenizer

            self.models["normalizer"] = WhisperTokenizer.from_pretrained(eval_model_dir("asr"))
        words = self.models["normalizer"].normalize(strip_emphasis_markers(text)).split()
        return [w for w in words if any(ch.isalnum() for ch in w)]   # it leaves "..." as a lone "."


def word_errors(ref, hyp):
    """Word-level edit distance (substitutions + deletions + insertions)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def trailing_missing(ref, hyp):
    """How many words at the END of the reference were not heard: the reference words after the last word heard exactly,
    minus the words heard after it (edit-distance alignment). 'Well, that is one way to look at it.' heard as 'Well,'
    gives 8; '... and nobody ever asked us to be anywhere else' heard as '... and no one' gives 6; '... at six forty five
    AM' heard as '... at 6.45 a.m.' gives 0 (the end was said, Whisper just wrote it differently)."""
    n, m = len(ref), len(hyp)
    d = [[i + j if i == 0 or j == 0 else 0 for j in range(m + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j = n, m
    while i and j:   # backtrace from the end to the last exact match
        if ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            return max(0, (n - i) - (m - j))
        if d[i][j] == d[i - 1][j - 1] + 1:
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return max(0, n - m)


# ---------------------------------------------------------------- real-recording sets
def speaker_clips(cfg, split):
    return [{"path": cfg.path("processed") / ex["audio_path"], "style": ex["style"], "text": ex["text"], "id": ex["id"]}
            for ex in load_jsonl(cfg.path("splits") / f"{split}.jsonl")]


def other_speaker_clips(cfg, speaker, styles):
    """A fixed, seeded sample of the other speakers' plain read recordings (no emphasis/essentials sub-corpora)."""
    read = cfg.path("raw") / "expresso" / "audio_48khz" / "read"
    out = []
    for spk_dir in sorted(p for p in read.iterdir() if p.is_dir() and p.name != speaker and not p.name.startswith(".")):
        for style in styles:
            cands = sorted(p for p in (spk_dir / style / "base").glob("*.wav")
                           if not p.name.startswith(".") and (parse_stem(p.stem) or {}).get("substyle") is None)
            rng = random.Random(f"{cfg['eval']['seed']}-{spk_dir.name}-{style}")   # adding a style leaves the others' samples alone
            for p in rng.sample(cands, min(OTHER_SPEAKER_CLIPS, len(cands))):
                out.append({"path": p, "style": style, "speaker": spk_dir.name, "id": p.stem})
    if not out:
        raise SystemExit(f"no other-speaker recordings under {read}")
    return out


# ---------------------------------------------------------------- calibrate
# One speaker-independent cue per style (needs no training), next to the speaker-specific classifier: the direction in
# which the style differs from default. Styles without an entry are judged by the classifier alone.
STYLE_CUES = {"sad": ("arousal", -1), "whisper": ("voicing", -1)}


def cue_score(clip, style):
    name, sign = STYLE_CUES[style]
    return sign * clip[name]


def cmd_calibrate(cfg, args):
    import numpy as np
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cfg.require_data_root()
    speaker, styles = cfg.speaker(), cfg["speaker"]["styles"]
    if styles[0] != "default":
        raise SystemExit(f"calibrate compares every style with 'default', which must come first; config has {styles}")
    m = Measures(cfg)
    train, held = speaker_clips(cfg, "train"), speaker_clips(cfg, "validation") + speaker_clips(cfg, "test")
    others = other_speaker_clips(cfg, speaker, styles)

    def progress(clips, what, fn):
        out = []
        for i, c in enumerate(clips, 1):
            out.append(fn(c["path"]))
            if i % 100 == 0 or i == len(clips):
                log(f"{what}: {i}/{len(clips)}")
                m.save()
        return out

    em_train = progress(train, "emotion, train split", m.emotion)
    for clips, what in ((held, "held-out"), (others, "other speakers")):
        for c, e in zip(clips, progress(clips, f"emotion, {what}", m.emotion)):
            c.update({d: e[d] for d in ("arousal", "valence", "dominance")}, embedding=e["embedding"])
        for c, v in zip(clips, progress(clips, f"voicing, {what}", m.voicing)):
            c["voicing"] = v
        for c, x in zip(clips, progress(clips, f"x-vector, {what}", m.xvector)):
            c["xvector"] = x
    xv_train = progress(train, "x-vector, train split", m.xvector)
    tr_held = progress(held, "transcripts, held-out", m.transcript)

    # style classifier: speaker-specific, real recordings only
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, cv=5, max_iter=5000))
    clf.fit(np.stack([e["embedding"] for e in em_train]), [c["style"] for c in train])
    classes = list(clf.classes_)
    probs = clf.predict_proba(np.stack([c["embedding"] for c in held]))
    for c, p in zip(held, probs):
        c["p"] = dict(zip(classes, map(float, p)))
    confusion = {s: {t: float(np.mean([max(c["p"], key=c["p"].get) == t for c in held if c["style"] == s])) for t in classes}
                 for s in styles}
    record_styles = {}
    for s in styles[1:]:
        pair = [c for c in held if c["style"] in ("default", s)]
        rec = {"classifier_auc": float(roc_auc_score([c["style"] == s for c in pair], [c["p"][s] for c in pair]))}
        if s in STYLE_CUES:
            other = [c for c in others if c["style"] in ("default", s)]
            rec["cue"] = STYLE_CUES[s][0]
            rec["cue_auc_held_out"] = float(roc_auc_score([c["style"] == s for c in pair], [cue_score(c, s) for c in pair]))
            rec["cue_auc_other_speakers"] = float(roc_auc_score([c["style"] == s for c in other], [cue_score(c, s) for c in other]))
        record_styles[s] = rec

    centroid = np.mean(np.stack(xv_train), axis=0)
    centroid /= np.linalg.norm(centroid)
    sim_held = [float(c["xvector"] @ centroid) for c in held]
    sim_other = [float(c["xvector"] @ centroid) for c in others]
    auc_spk = float(roc_auc_score([1] * len(sim_held) + [0] * len(sim_other), sim_held + sim_other))

    wer = {}
    for s in styles:
        errs = words = 0
        for c, hyp in zip(held, tr_held):
            if c["style"] == s:
                ref = m.normalize_text(c["text"])
                errs += word_errors(ref, m.normalize_text(hyp))
                words += len(ref)
        wer[s] = errs / words
    m.save()

    by_style = {s: {"classified_as": confusion[s],
                    **{d: statistics.median([c[d] for c in held if c["style"] == s]) for d in ("arousal", "valence", "dominance", "voicing")},
                    "speaker_similarity": statistics.median([v for v, c in zip(sim_held, held) if c["style"] == s]),
                    "wer": wer[s]} for s in styles}
    record = {
        "created_at": utc_now(), "speaker": speaker, "styles": styles, "classes": classes,
        "models": {k: {"repo": v[0], "revision": v[1], "license": v[3]} for k, v in EVAL_MODELS.items()},
        "n": {"train": len(train), "held_out": len(held), "other_speakers": len(others)},
        "classifier": {"held_out_accuracy": float(np.mean([max(c["p"], key=c["p"].get) == c["style"] for c in held])), "C": clf[-1].C_.tolist()},
        "styles_vs_default": record_styles,
        "speaker_similarity": {"held_out_median": statistics.median(sim_held), "other_speakers_median": statistics.median(sim_other), "auc": auc_spk},
        "held_out_by_style": by_style,
    }
    out = auto_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "style_classifier.pkl").write_bytes(pickle.dumps({"pipeline": clf, "centroid": centroid, "record": record}))
    write_json(out / "calibration.json", record)

    lines = [f"# Automatic measures: calibration on real recordings ({speaker})", "",
             f"Train split {len(train)} clips; held-out (validation + test) {len(held)} clips; other Expresso speakers {len(others)} clips. "
             f"Styles: {', '.join(styles)}.", "AUC: 0.5 = no better than chance, 1.0 = perfect separation.", "",
             "| measure | check | result |", "|---|---|---|",
             f"| style classifier ({len(classes)} styles) | held-out, share classified as the right style | {record['classifier']['held_out_accuracy']:.0%} |"]
    for s, r in record_styles.items():
        lines.append(f"| P({s}) | held-out default vs {s}, AUC | {r['classifier_auc']:.3f} |")
        if "cue" in r:
            sign = "low" if STYLE_CUES[s][1] < 0 else "high"
            lines.append(f"| {sign} {r['cue']} (no training) | held-out {speaker} default vs {s}, AUC | {r['cue_auc_held_out']:.3f} |")
            lines.append(f"| {sign} {r['cue']} (no training) | other speakers default vs {s}, AUC | {r['cue_auc_other_speakers']:.3f} |")
    ss = record["speaker_similarity"]
    lines += [f"| speaker similarity | held-out {speaker} vs other speakers, AUC | {ss['auc']:.3f} |",
              f"| speaker similarity | median: held-out {speaker} / other speakers | {ss['held_out_median']:.3f} / {ss['other_speakers_median']:.3f} |"]
    for s in styles:
        lines.append(f"| word error rate | held-out real {s} recordings | {wer[s]:.1%} |")
    lines += ["", "Held-out real recordings by style (classified as: share of clips the classifier puts in each style):", "",
              "| style | " + " | ".join(f"classified {t}" for t in classes) + " | median arousal | median valence | median voicing |",
              "|---|" + "---|" * (len(classes) + 3)]
    for s, v in by_style.items():
        lines.append(f"| {s} | " + " | ".join(f"{v['classified_as'][t]:.0%}" for t in classes)
                     + f" | {v['arousal']:.3f} | {v['valence']:.3f} | {v['voicing']:.0%} |")
    text = "\n".join(lines)
    (out / "calibration.md").write_text(text + "\n")
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


# ---------------------------------------------------------------- score
def cmd_score(cfg, args):
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    cfg.require_data_root()
    clf_path = auto_dir(cfg) / "style_classifier.pkl"
    if not clf_path.is_file():
        raise SystemExit("no classifier yet: run scripts/auto_evaluate.py calibrate first")
    saved = pickle.loads(clf_path.read_bytes())
    clf, centroid, classes = saved["pipeline"], saved["centroid"], list(saved["pipeline"].classes_)
    m = Measures(cfg)

    clips = []  # {run, style, item_id, seed, clip_id, path, text}
    for label in args.runs:
        d = cfg.path("eval") / "runs" / label
        if not (d / "metadata.jsonl").is_file():
            raise SystemExit(f"run '{label}' has no metadata at {d}")
        for r in load_jsonl(d / "metadata.jsonl"):
            clips.append({"run": label, "style": r["style"], "item_id": r["item_id"], "seed": r["seed"],
                          "clip_id": r["clip_id"], "path": d / r["wav"], "text": r["text"]})
    es = read_json(cfg.path("splits") / "eval_set.json")
    test = {ex["id"]: ex for ex in load_jsonl(cfg.path("splits") / "test.jsonl")}
    for it in es["items"]:
        for style, ex_id in it["reference_examples"].items():
            ex = test[ex_id]
            clips.append({"run": REAL, "style": style, "item_id": it["item_id"], "seed": None, "clip_id": ex_id,
                          "path": cfg.path("processed") / ex["audio_path"], "text": ex["text"]})

    for i, c in enumerate(clips, 1):
        em = m.emotion(c["path"])
        c["p"] = dict(zip(classes, map(float, clf.predict_proba(em["embedding"][None])[0])))
        c["classified"] = max(c["p"], key=c["p"].get)
        c.update({d: em[d] for d in ("arousal", "valence", "dominance")}, voicing=m.voicing(c["path"]))
        c["speaker_sim"] = float(m.xvector(c["path"]) @ centroid)
        ref, hyp = m.normalize_text(c["text"]), m.normalize_text(m.transcript(c["path"]))
        c["errors"], c["words"] = word_errors(ref, hyp), len(ref)
        if i % 25 == 0 or i == len(clips):
            log(f"scored {i}/{len(clips)}")
            m.save()

    runs = list(args.runs) + [REAL]
    styles = [s for s in es["styles"] if any(c["style"] == s for c in clips)]
    cell = defaultdict(list)
    for c in clips:
        cell[(c["run"], c["style"])].append(c)

    def med(xs):
        return statistics.median(xs)

    cal = saved["record"]
    lines = ["# Automatic measures on the evaluation set", "",
             f"Classifier and speaker centroid from `calibrate` ({cal['created_at']}; styles {', '.join(classes)}; held-out real "
             f"recordings classified correctly: {cal['classifier']['held_out_accuracy']:.0%}).",
             f"'{REAL}' are the speaker's own held-out recordings of the same test sentences (no new sentences exist for them).", "",
             "| run | style | n | " + " | ".join(f"classified {t}" for t in classes)
             + " | median arousal | median voicing | WER | median speaker similarity |",
             "|---|---|---|" + "---|" * (len(classes) + 4)]
    for run in runs:
        for s in styles:
            xs = cell[(run, s)]
            if xs:
                lines.append(f"| {run} | {s} | {len(xs)} | " + " | ".join(f"{np.mean([x['classified'] == t for x in xs]):.0%}" for t in classes)
                             + f" | {med([x['arousal'] for x in xs]):.3f} | {med([x['voicing'] for x in xs]):.0%} | "
                             f"{sum(x['errors'] for x in xs) / sum(x['words'] for x in xs):.1%} | {med([x['speaker_sim'] for x in xs]):.3f} |")

    # paired: the same sentence and seed in default and in each other style
    pairs = defaultdict(dict)
    for c in clips:
        pairs[(c["run"], c["item_id"], c["seed"])][c["style"]] = c
    lines += ["", "Paired comparison with default, the automatic version of \"which of these two is more <style>?\": for each sentence "
              "(and seed), the share of pairs where the style's version scores higher on P(style) or on the style's own cue "
              "(low arousal for sad, little voicing for whisper) than the default version. 50% = not told apart.", "",
              "| run | style | pairs | higher P(style) | stronger cue | separation in classified style (points) |",
              "|---|---|---|---|---|---|"]
    summary = defaultdict(dict)
    for run in runs:
        for s in styles[1:]:
            ps = [v for k, v in pairs.items() if k[0] == run and s in v and "default" in v]
            if not ps:
                continue
            higher = float(np.mean([p[s]["p"][s] > p["default"]["p"][s] for p in ps]))
            cue = float(np.mean([cue_score(p[s], s) > cue_score(p["default"], s) for p in ps])) if s in STYLE_CUES else None
            sep = float(np.mean([x["classified"] == s for x in cell[(run, s)]]) - np.mean([x["classified"] == s for x in cell[(run, "default")]])) * 100
            summary[run][s] = {"pairs": len(ps), "paired_higher_p": higher, "paired_stronger_cue": cue, "separation_points": sep}
            lines.append(f"| {run} | {s} | {len(ps)} | {higher:.0%} | {'-' if cue is None else f'{cue:.0%}'} | {sep:+.0f} |")

    if args.key and args.ratings:
        key = read_json(args.key)["labels"]
        by_id = {(c["run"] if c["run"] != REAL else "reference", c["clip_id"]): c for c in clips}
        rated = []
        for rf in args.ratings:
            for lab, x in read_json(rf)["ratings"].items():
                if lab not in key or not all(x.get(k) for k in ("emotion", "naturalness", "intelligibility", "speaker_consistency")):
                    continue
                c = by_id.get((key[lab]["run"], key[lab]["clip_id"]))
                if c:
                    rated.append((c, x))
        if rated:
            heard_sad = [x["emotion"] == "sad" for _, x in rated]
            lines += ["", f"## Agreement with listener ratings ({len(rated)} rated clips from {len(args.ratings)} file(s))", "",
                      "| automatic measure | listener rating | agreement |", "|---|---|---|"]
            if 0 < sum(heard_sad) < len(heard_sad):
                lines.append(f"| P(sad) | heard as sad ({sum(heard_sad)} clips) vs not | AUC {roc_auc_score(heard_sad, [c['p']['sad'] for c, _ in rated]):.3f} |")
                lines.append(f"| low arousal | heard as sad vs not | AUC {roc_auc_score(heard_sad, [-c['arousal'] for c, _ in rated]):.3f} |")
            for auto_k, human_k, sign, name in (("speaker_sim", "speaker_consistency", 1, "speaker similarity"),
                                                ("errors", "intelligibility", -1, "word errors (negated)")):
                rho = spearmanr([sign * c[auto_k] for c, _ in rated], [int(x[human_k]) for _, x in rated]).statistic
                lines.append(f"| {name} | {human_k.replace('_', ' ')} (1-5) | Spearman {rho:+.2f} |")
            lines += ["", "| run | style | n | listener: heard as sad | automatic: classified sad |", "|---|---|---|---|---|"]
            rc = defaultdict(list)
            for c, x in rated:
                rc[(c["run"], c["style"])].append((c, x))
            for (run, s), xs in sorted(rc.items(), key=lambda kv: (kv[0][0] == REAL, kv[0])):
                lines.append(f"| {run} | {s} | {len(xs)} | {np.mean([x['emotion'] == 'sad' for _, x in xs]):.0%} | "
                             f"{np.mean([c['classified'] == 'sad' for c, _ in xs]):.0%} |")
            lines.append("\nAUC 0.5 = no agreement, 1.0 = perfect; Spearman 0 = no agreement, +1 = same ranking.")

    lines += ["", "These are automatic proxies, not ratings. The classifier is trained on this speaker's real recordings: it is "
              "meaningless for a voice that is not this speaker (the unchanged baseline), and for sad it was more lenient than a "
              "listener (set-b), so use it to compare models of this voice, not as proof of how a clip sounds. The cues (arousal, "
              "voicing) need no training and work across speakers."]
    text = "\n".join(lines)
    out = auto_dir(cfg)
    write_json(out / f"score_{'_'.join(args.runs)}.json",
               {"created_at": utc_now(), "runs": args.runs, "classes": classes, "summary": summary,
                "clips": [{k: (str(v) if isinstance(v, Path) else v) for k, v in c.items()} for c in clips]})
    (out / f"score_{'_'.join(args.runs)}.md").write_text(text + "\n")
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


# ---------------------------------------------------------------- completeness
GARBLED_WER = 0.25     # complete but with a quarter of the words wrong: worth a listen


def completeness_flags(ref, hyp, path, text, style, pace):
    """('incomplete' | 'check' | None, detail): the transcript checks here, plus the audio-only ones shared with
    speak.py (expresso_common.audio_problem). pace: speaking_pace(cfg)."""
    miss = trailing_missing(ref, hyp)
    if miss >= 2 or (miss and miss / len(ref) >= 1 / 3):
        return "incomplete", f"last {miss} of {len(ref)} words missing"
    problem = audio_problem(path, text, style, pace)
    if problem:
        return ("check" if problem[0] == "too_long" else "incomplete"), problem[1]
    wer = word_errors(ref, hyp) / max(len(ref), 1)
    if wer >= GARBLED_WER:
        return "check", f"word error rate {wer:.0%}"
    return None, ""


def cmd_completeness(cfg, args):
    """Flag generated clips that stop early (or need a listen), after checking the same rule on real recordings."""
    from scipy.stats import binomtest

    cfg.require_data_root()
    m = Measures(cfg)
    pace = speaking_pace(cfg)

    def evaluate(path, text, style):
        ref, hyp = m.normalize_text(text), m.transcript(path)
        flag, detail = completeness_flags(ref, m.normalize_text(hyp), path, text, style, pace)
        return {"flag": flag, "detail": detail, "transcript": hyp}

    real = speaker_clips(cfg, "validation") + speaker_clips(cfg, "test")
    real_flags = []
    for c in real:
        r = evaluate(c["path"], c["text"], c["style"])
        real_flags.append(r | {"id": c["id"], "text": c["text"]})
    m.save()

    clips = []
    for label in args.runs:
        d = cfg.path("eval") / "runs" / label
        if not (d / "metadata.jsonl").is_file():
            raise SystemExit(f"run '{label}' has no metadata at {d}")
        recs = load_jsonl(d / "metadata.jsonl")
        for i, r in enumerate(recs, 1):
            group = r["item_id"].rstrip("0123456789")
            clips.append({"run": label, "group": group, "style": r["style"], "clip_id": r["clip_id"], "text": r["text"],
                          "duration_s": r["duration_s"], "path": str(d / r["wav"])}
                         | evaluate(d / r["wav"], r["text"], r["style"]))
            if i % 20 == 0 or i == len(recs):
                log(f"{label}: {i}/{len(recs)}")
                m.save()

    def rate(k, n):
        ci = binomtest(k, n).proportion_ci(method="wilson")
        return f"{k}/{n} ({k / n:.0%}; 95% CI {ci.low:.0%}-{ci.high:.0%})"

    n_real_inc = sum(r["flag"] == "incomplete" for r in real_flags)
    lines = ["# Early stops and other incomplete speech", "",
             f"Rule: incomplete if Whisper's transcript lacks the last 2+ words (or the last third) of the text, the clip ends "
             f"mid-sound (last 100 ms less than {CUT_OFF_DB:.0f} dB below its speech level), or it is "
             f"under {TOO_SHORT}x the time ex02 needs for the text (characters / ex02's speaking pace for that style). "
             f"'Check' = complete but over {TOO_LONG}x that time, or a word error rate of {GARBLED_WER:.0%} or more.", "",
             f"**The rule on {len(real)} real held-out recordings:** {rate(n_real_inc, len(real))} flagged incomplete, "
             f"{sum(r['flag'] == 'check' for r in real_flags)} flagged to check.", ""]
    for r in real_flags:
        if r["flag"]:
            lines.append(f"- real `{r['id']}` ({r['flag']}: {r['detail']}): \"{r['text']}\" heard as \"{r['transcript']}\"")
    for label in args.runs:
        cs = [c for c in clips if c["run"] == label]
        order = ["short", "medium", "long"]
        groups = sorted({c["group"] for c in cs}, key=lambda g: (order.index(g) if g in order else len(order), g))
        lines += ["", f"## {label}", "", "| length | style | clips | incomplete | to check |", "|---|---|---|---|---|"]
        for g in groups:
            for s in sorted({c["style"] for c in cs}):
                xs = [c for c in cs if c["group"] == g and c["style"] == s]
                if xs:
                    lines.append(f"| {g} | {s} | {len(xs)} | {sum(c['flag'] == 'incomplete' for c in xs)} | {sum(c['flag'] == 'check' for c in xs)} |")
        inc = sum(c["flag"] == "incomplete" for c in cs)
        lines += ["", f"**Incomplete overall: {rate(inc, len(cs))}.**"]
        for s in sorted({c["style"] for c in cs}):
            xs = [c for c in cs if c["style"] == s]
            lines.append(f"- {s}: {rate(sum(c['flag'] == 'incomplete' for c in xs), len(xs))}")
        flagged = [c for c in cs if c["flag"]]
        if flagged:
            lines += ["", "Flagged clips (listen to these):", ""]
            for c in flagged:
                lines.append(f"- `{c['clip_id']}` {c['flag']}: {c['detail']}. \"{c['text']}\" heard as \"{c['transcript']}\" "
                             f"(`{c['path']}`)")
    text = "\n".join(lines)
    out = auto_dir(cfg)
    write_json(out / f"completeness_{'_'.join(args.runs)}.json", {"created_at": utc_now(), "runs": args.runs, "pace": pace,
                                                                  "real": real_flags, "clips": clips})
    (out / f"completeness_{'_'.join(args.runs)}.md").write_text(text + "\n")
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", help="path to the experiment config")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="fetch the pinned evaluation models")
    c = sub.add_parser("calibrate", help="check the measures on real recordings and fit the sadness classifier")
    c.add_argument("--out")
    s = sub.add_parser("score", help="apply the measures to generated runs")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--key", help="listening-page answer key, to compare with --ratings")
    s.add_argument("--ratings", nargs="+")
    s.add_argument("--out")
    k = sub.add_parser("completeness", help="flag clips that stop early or need a listen")
    k.add_argument("--runs", nargs="+", required=True)
    k.add_argument("--out")
    args = ap.parse_args()
    cfg = Config(args.config)
    {"download": cmd_download, "calibrate": cmd_calibrate, "score": cmd_score, "completeness": cmd_completeness}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
