"""Shared helpers for the Expresso emotion experiment: config, paths, transcript normalization.

Import _common first (it loads .env). Nothing here depends on the shell's current directory:
the config path and every relative path in it resolve against the repo root.
"""
import hashlib
import json
import os
import re
import statistics
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from _common import ROOT, output_dir

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 (older Colab images): pip install tomli
    import tomli as tomllib

DEFAULT_CONFIG = ROOT / "configs" / "expresso_emotion.toml"


class Config:
    """The experiment config plus resolved absolute paths.

    cfg.raw["section"]["key"] is the TOML content; cfg.path("raw") etc. are absolute Paths.
    """

    def __init__(self, path=None):
        self.file = Path(path or os.environ.get("VIBE_TTS_EXPRESSO_CONFIG") or DEFAULT_CONFIG).expanduser().resolve()
        if not self.file.is_file():
            raise SystemExit(f"config not found: {self.file}")
        with open(self.file, "rb") as f:
            self.raw = tomllib.load(f)
        p = self.raw["paths"]
        env_root = os.environ.get("VIBE_TTS_DATA_DIR")
        root = Path(env_root).expanduser() if env_root else Path(p["data_root"])
        self.data_root = root if root.is_absolute() else ROOT / root
        self._paths = {k: self.data_root / v for k, v in p.items() if k not in ("data_root", "eval_subdir")}
        self._paths["eval"] = output_dir() / p["eval_subdir"]

    def __getitem__(self, section):
        return self.raw[section]

    def path(self, name):
        return self._paths[name]

    def require_data_root(self):
        """Fail clearly if data_root is a dangling symlink / unmounted volume (do not create it elsewhere)."""
        if self.data_root.is_symlink() and not self.data_root.exists():
            raise SystemExit(
                f"{self.data_root} points to {os.readlink(self.data_root)}, which does not exist "
                f"(external drive not mounted?)"
            )
        self.data_root.mkdir(parents=True, exist_ok=True)

    def speaker(self):
        """The pinned speaker id. 'auto' is resolved from the inventory's recorded selection."""
        sid = self.raw["speaker"]["id"]
        if sid != "auto":
            return sid
        sel = self.path("inventory") / "speaker_selection.json"
        if not sel.is_file():
            raise SystemExit(
                "speaker.id is 'auto' but there is no inventory yet. Run scripts/download_expresso.py "
                "and scripts/inventory_expresso.py, then pin speaker.id in "
                f"{self.file.relative_to(ROOT)}."
            )
        return json.loads(sel.read_text())["selected"]


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def md5_file(path, chunk=1 << 22, progress=None):
    h = hashlib.md5()
    done = 0
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
            done += len(block)
            if progress:
                progress(done)
    return h.hexdigest()


def write_json(path, obj):
    """Atomic JSON write (a crashed run never leaves a half-written record)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def normalize_transcript(text):
    """Key used to group 'the same words' across styles/recordings: NFKC, lowercase, no punctuation,
    collapsed whitespace. Apostrophes are dropped so "don't" == "dont" == "don’t"."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = t.replace("’", "'").replace("‘", "'")
    t = re.sub(r"['`]", "", t)
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def is_junk(path):
    """macOS AppleDouble sidecars ('._name', created on non-HFS volumes) and Finder droppings are not data."""
    n = Path(path).name
    return n.startswith("._") or n == ".DS_Store"


def remove_tree(path):
    """Delete a directory tree, tolerating entries that vanish mid-way. On exFAT/FAT volumes macOS stores extended
    attributes in '._name' sidecars that disappear together with their main file, which breaks shutil.rmtree."""
    path = Path(path)
    if not path.exists():
        return
    for p in sorted(path.rglob("*"), key=lambda q: len(q.parts), reverse=True):
        try:
            p.rmdir() if p.is_dir() and not p.is_symlink() else p.unlink()
        except FileNotFoundError:
            pass
    try:
        path.rmdir()
    except FileNotFoundError:
        pass


def log(msg, *, err=False):
    print(msg, file=sys.stderr if err else sys.stdout, flush=True)


# 20 ms frames: short enough to see pauses, long enough for a stable RMS.
FRAME_S = 0.02


def analyze_audio(path, clip_level=0.999):
    """Decode a WAV completely and measure it. Never raises: problems come back in rec["error"].

    The measurements are descriptive only; nothing here modifies audio. Levels are dBFS of the
    mono mix. noise_floor_dbfs is the 10th percentile of 20 ms frame RMS (the quietest frames,
    i.e. room noise or digital silence), speech_level_dbfs the 95th percentile.
    """
    import numpy as np
    import soundfile as sf

    rec = {"error": None}
    try:
        info = sf.info(str(path))
        rec.update(format=info.format, subtype=info.subtype, sample_rate=info.samplerate,
                   channels=info.channels, header_frames=info.frames)
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:
        rec["error"] = f"unreadable: {e}"
        return rec
    n = x.shape[0]
    rec["decoded_frames"] = int(n)
    rec["duration_s"] = round(n / sr, 4)
    if n == 0:
        rec["error"] = "empty audio"
        return rec
    if n != rec["header_frames"]:
        rec["error"] = f"header says {rec['header_frames']} frames, decoded {n}"
    bad = int((~np.isfinite(x)).sum())
    rec["nonfinite_samples"] = bad
    if bad:
        rec["error"] = rec["error"] or f"{bad} non-finite samples"
        return rec
    m = x.mean(axis=1)
    a = np.abs(m)
    peak = float(a.max())
    rec["peak"] = round(peak, 5)
    rec["clipping_fraction"] = float((a >= clip_level).mean())
    k = int(FRAME_S * sr)
    nf = n // k
    if nf >= 5:
        rms = np.sqrt((m[: nf * k].reshape(nf, k) ** 2).mean(axis=1))
        db = 20 * np.log10(np.maximum(rms, 1e-10))
        floor, speech = float(np.percentile(db, 10)), float(np.percentile(db, 95))
        thr = max(floor + 10.0, speech - 35.0)
        voiced = np.flatnonzero(db > thr)
        rec.update(
            noise_floor_dbfs=round(floor, 2), speech_level_dbfs=round(speech, 2),
            snr_proxy_db=round(speech - floor, 2),
            lead_silence_s=round(float(voiced[0] * FRAME_S), 3) if len(voiced) else None,
            trail_silence_s=round(float((nf - 1 - voiced[-1]) * FRAME_S), 3) if len(voiced) else None,
            active_fraction=round(float(len(voiced) / nf), 4),
        )
    rec["rms_dbfs"] = round(float(20 * np.log10(max(np.sqrt((m ** 2).mean()), 1e-10))), 2)
    return rec


def description_for(cfg, speaker, style):
    """Conditioning description. Identical structure for every style; only the style phrase differs."""
    d = cfg["description"]
    return d["template"].format(speaker=speaker, style_phrase=d["style_phrase"][style])


# ---------------------------------------------------------------------------------------------
# Release layout: expresso/audio_48khz/read/{speaker}/{style}/{corpus}/{speaker}_{style[_substyle]}_{id}.wav
# Transcripts: expresso/read_transcriptions.txt, one "{stem}<TAB>{text}" per line (verified against the
# real file; text may contain *word* emphasis markers).
READ_STYLES = {"confused", "default", "enunciated", "happy", "laughing", "narration", "sad", "whisper"}
STEM_RE = re.compile(r"^(?P<speaker>ex\d+)_(?P<middle>[a-z]+(?:_[a-z]+)*)_(?P<num>\d+)$")


def parse_stem(stem):
    """'ex01_default_emphasis_00010' -> dict(speaker='ex01', style='default', substyle='emphasis', num='00010').
    Returns None for stems that do not follow the release naming."""
    m = STEM_RE.match(stem)
    if not m:
        return None
    first, _, rest = m["middle"].partition("_")
    return {"speaker": m["speaker"], "style": first, "substyle": rest or None, "num": m["num"]}


def load_transcripts(raw_root, with_problems=False):
    """{stem: text} from the release's read_transcriptions.txt. With with_problems=True also returns a
    list of {line, problem} for malformed or duplicate lines (the dict keeps the first occurrence)."""
    path = Path(raw_root) / "expresso" / "read_transcriptions.txt"
    if not path.is_file():
        raise SystemExit(f"{path} not found: run scripts/download_expresso.py")
    texts, problems = {}, []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        stem, tab, text = line.partition("\t")
        if not tab or not stem.strip():
            problems.append({"line": n, "problem": "no tab separator", "content": line[:80]})
        elif stem in texts:
            problems.append({"line": n, "problem": f"duplicate transcript for {stem}"})
        else:
            texts[stem.strip()] = text.strip()
    return (texts, problems) if with_problems else texts


def strip_emphasis_markers(text):
    """The release marks emphasized words as *word*. Those asterisks are annotation, not speech, so the
    training text drops them (and nothing else). The original line is always kept alongside."""
    return re.sub(r"\s+", " ", text.replace("*", "")).strip()


def quality_issues(rec, text, prep):
    """Reasons a clip should not be used, as (code, detail) pairs; [] means usable.
    `rec` is analyze_audio() output, `text` the transcript (or None), `prep` the [prepare] config."""
    issues = []
    if rec.get("error"):
        issues.append(("corrupt", rec["error"]))
        return issues
    if rec.get("channels") != 1:
        issues.append(("not_mono", f"{rec.get('channels')} channels"))
    d = rec["duration_s"]
    if d < prep["min_duration_s"]:
        issues.append(("too_short", f"{d:.2f}s < {prep['min_duration_s']}s"))
    if d > prep["max_duration_s"]:
        issues.append(("too_long", f"{d:.2f}s > {prep['max_duration_s']}s"))
    if rec["clipping_fraction"] > prep["max_clipping_fraction"]:
        issues.append(("clipping", f"{rec['clipping_fraction']:.4%} of samples at full scale"))
    if rec["peak"] < prep["min_peak"]:
        issues.append(("near_silent", f"peak {rec['peak']}"))
    if text is None:
        issues.append(("no_transcript", "no line in read_transcriptions.txt"))
    elif not strip_emphasis_markers(text):
        issues.append(("empty_transcript", "transcript is empty"))
    elif len(strip_emphasis_markers(text)) > prep["max_text_chars"]:
        issues.append(("text_too_long", f"{len(text)} characters"))
    return issues


def check_descriptions(cfg, speaker, styles, accepted, pitch):
    """Test what the conditioning descriptions CLAIM against what the accepted recordings MEASURE.
    `accepted` maps style -> list of per-clip analyze_audio() dicts (+ chars_per_active_s); `pitch` maps
    style -> {"f0_median_hz": [...], "f0_range_st": [...]} from the inventory sample (may be empty).
    Perceptual claims (e.g. 'conversational') cannot be verified by measurement and are reported as such."""
    chk = cfg["description"]["checks"]

    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.median(xs), 3) if xs else None

    def p95(xs):
        xs = sorted(xs)
        return round(xs[int(0.95 * (len(xs) - 1))], 2)

    out = {"speaker": speaker, "per_style": {}, "claims": []}
    for s in styles:
        c = accepted[s]
        out["per_style"][s] = {
            "clips": len(c), "noise_floor_median_dbfs": med([x["noise_floor_dbfs"] for x in c]),
            "noise_floor_p95_dbfs": p95([x["noise_floor_dbfs"] for x in c]),
            "speech_level_dbfs": med([x["speech_level_dbfs"] for x in c]),
            "active_fraction": med([x["active_fraction"] for x in c]),
            "chars_per_active_s": med([x["chars_per_active_s"] for x in c]),
            "f0_median_hz": med(pitch.get(s, {}).get("f0_median_hz", [])),
            "f0_range_st": med(pitch.get(s, {}).get("f0_range_st", [])),
            "pitch_clips_measured": len(pitch.get(s, {}).get("f0_median_hz", [])),
        }
    for s in styles:
        m = out["per_style"][s]
        ok = m["noise_floor_median_dbfs"] <= chk["noise_floor_median_max_dbfs"] and m["noise_floor_p95_dbfs"] <= chk["noise_floor_p95_max_dbfs"]
        out["claims"].append({"claim": f"'{s}': the recording is clear with little background noise", "verdict": "supported" if ok else "NOT supported",
                              "evidence": f"noise floor median {m['noise_floor_median_dbfs']} dBFS (limit {chk['noise_floor_median_max_dbfs']}), "
                                          f"95th percentile {m['noise_floor_p95_dbfs']} dBFS (limit {chk['noise_floor_p95_max_dbfs']})"})
    if "default" in styles and "sad" in styles:
        d, sd = out["per_style"]["default"], out["per_style"]["sad"]
        ind = {}
        for k, label in (("speech_level_dbfs", "speech level"), ("f0_median_hz", "F0 median"), ("f0_range_st", "F0 range"),
                         ("chars_per_active_s", "speaking pace"), ("active_fraction", "share of time speaking")):
            if d[k] is not None and sd[k] is not None:
                ind[label] = {"default": d[k], "sad": sd[k], "sad_is_lower": sd[k] < d[k]}
        n_low = sum(v["sad_is_lower"] for v in ind.values())
        out["claims"].append({"claim": "'sad': a subdued delivery (lower and slower than default)",
                              "verdict": "supported" if n_low >= chk["subdued_min_indicators"] else "NOT supported",
                              "evidence": f"sad lower in {n_low} of {len(ind)} measured indicators: " + "; ".join(
                                  f"{k} {v['sad']} vs {v['default']}" for k, v in ind.items())})
    out["claims"].append({"claim": "'default': a neutral delivery", "verdict": "label only",
                          "evidence": "Expresso's 'default' style is the unmarked read style; neutrality is not measurable here"})
    out["claims"].append({"claim": "'conversational' (if present in the style phrase)", "verdict": "NOT verifiable",
                          "evidence": "the clips are read-aloud sentences, not conversation; no measurement can support the word"})
    return out

