"""Prepare training examples for one Expresso speaker and create frozen, leak-free splits.

Usage: .venv/bin/python scripts/prepare_expresso.py [--config PATH] [--workers N]
                                                     [--verify-only] [--refreeze]

Reads <data root>/raw (scripts/download_expresso.py) and the speaker / styles / gates in the config.
Writes:
  processed/audio/<speaker>/<id>.wav   resampled audio at the model's sample rate (24-bit)
  processed/manifest.jsonl             every accepted example, with its split
  processed/exclusions.jsonl           every excluded clip and why
  splits/{train,validation,test}.jsonl + split_report.{json,md} + FROZEN.json

Per example: id (the release file stem), source_file, audio_path, text (exact transcript from
read_transcriptions.txt; only the *emphasis* asterisks are dropped, the original line is kept as
text_original), speaker, style, description, duration_s, sample_rate, split.

Audio handling: decode, check, resample with soxr (VHQ), write. NO denoising, silence trimming, or
loudness normalization: pauses and quiet speech carry the delivery being learned.

Splits: examples are grouped (union-find) by normalized transcript and by source-audio hash, then whole
groups are assigned to train/validation/test (~80/10/10, seeded, styles balanced), so the same words in the
other style, or a repeated recording, can never cross a split. The result is verified programmatically and the
test split is frozen: re-running reproduces it bit for bit or refuses.
"""
import argparse
import json
import os
import random
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from _common import ROOT, model_dir  # first: loads .env
from expresso_common import (Config, analyze_audio, check_descriptions, description_for, load_transcripts, log, normalize_transcript, remove_tree,
                             quality_issues, read_json, sha256_file, strip_emphasis_markers, utc_now, write_json)
from inventory_expresso import discover

SPLITS = ("train", "validation", "test")


def model_sample_rate(model_path):
    """The sample rate the model was trained at, read from its own config (never assumed)."""
    conf = read_json(model_path / "config.json")
    rates = {"audio_encoder.sampling_rate": conf.get("audio_encoder", {}).get("sampling_rate"),
             "config.sampling_rate": conf.get("sampling_rate")}
    pre = model_path / "preprocessor_config.json"
    if pre.is_file():
        rates["preprocessor_config.sampling_rate"] = read_json(pre).get("sampling_rate")
    found = {k: v for k, v in rates.items() if v}
    if not found or len(set(found.values())) != 1:
        raise SystemExit(f"cannot determine a single model sample rate from {model_path}: {rates}")
    return next(iter(found.values())), found


def _process(job):
    """Worker: decode + gate + resample + write one clip. Returns the record (with issues)."""
    import numpy as np
    import soundfile as sf
    import soxr

    src, dst, target_sr, text, prep = job
    rec = analyze_audio(src)
    rec["sha256"] = sha256_file(src)
    issues = quality_issues(rec, text, prep)
    rec["issues"] = [list(i) for i in issues]
    if issues:
        return rec
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    m = x[:, 0]
    y = soxr.resample(m, sr, target_sr, quality=prep["resample_quality"]).astype("float32")
    # Resampling must preserve length and level; anything else would be a header rewrite in disguise.
    want = round(len(m) * target_sr / sr)
    rms_in, rms_out = float(np.sqrt((m ** 2).mean())), float(np.sqrt((y ** 2).mean()))
    rec["resample_len_error"] = abs(len(y) - want)
    rec["resample_rms_db_diff"] = round(20 * np.log10(max(rms_out, 1e-12) / max(rms_in, 1e-12)), 4)
    rec["out_peak"] = round(float(np.abs(y).max()), 5)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, y, target_sr, subtype=prep["output_subtype"])
    info = sf.info(dst)
    rec.update(out_sample_rate=info.samplerate, out_frames=info.frames, out_duration_s=round(info.frames / info.samplerate, 4))
    return rec


# ------------------------------------------------------------------ splitting
def group_examples(examples):
    """Union-find over shared normalized transcript, shared source hash, shared source file."""
    parent = list(range(len(examples)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for key in ("norm_text", "source_sha256", "source_file"):
        seen = {}
        for i, ex in enumerate(examples):
            k = ex[key]
            if k in seen:
                union(i, seen[k])
            else:
                seen[k] = i
    groups = defaultdict(list)
    for i in range(len(examples)):
        groups[find(i)].append(i)
    return list(groups.values())


def assign_splits(examples, ratios, styles, seed):
    """Greedy, seeded assignment of whole groups to splits so each split gets its share of EACH style."""
    groups = group_examples(examples)
    rng = random.Random(seed)
    groups.sort(key=lambda g: (-len(g), min(examples[i]["id"] for i in g)))       # big groups first: they are the hardest to place
    # shuffle within equal-size runs, deterministically
    by_size = defaultdict(list)
    for g in groups:
        by_size[len(g)].append(g)
    ordered = []
    for size in sorted(by_size, reverse=True):
        gs = by_size[size]
        rng.shuffle(gs)
        ordered.extend(gs)
    total = {s: sum(1 for e in examples if e["style"] == s) for s in styles}
    target = {sp: {s: total[s] * ratios[sp] for s in styles} for sp in SPLITS}
    have = {sp: Counter() for sp in SPLITS}
    assign = {}
    for g in ordered:
        comp = Counter(examples[i]["style"] for i in g)

        def deficit(sp):  # how far below target this split is, in the styles this group would add to
            return sum((target[sp][s] - have[sp][s]) / max(target[sp][s], 1e-9) * c for s, c in comp.items())

        best = max(SPLITS, key=lambda sp: (deficit(sp), ratios[sp]))
        for i in g:
            assign[i] = best
        have[best].update(comp)
    return assign


def split_report(examples, styles):
    rep = {}
    for sp in SPLITS:
        rows = [e for e in examples if e["split"] == sp]
        rep[sp] = {"examples": len(rows), "distinct_transcripts": len({e["norm_text"] for e in rows}),
                   "duration_s": round(sum(e["duration_s"] for e in rows), 1), "per_style": {}}
        for s in styles:
            r = [e for e in rows if e["style"] == s]
            rep[sp]["per_style"][s] = {"examples": len(r), "distinct_transcripts": len({e["norm_text"] for e in r}),
                                       "duration_s": round(sum(e["duration_s"] for e in r), 1)}
    return rep


def verify(examples, cfg, styles, tol=0.02):
    """Programmatic leak and balance checks. Returns a list of failures (empty = all good)."""
    fails = []
    ids = [e["id"] for e in examples]
    if len(set(ids)) != len(ids):
        fails.append("duplicate example ids")
    if any(e["split"] not in SPLITS for e in examples):
        fails.append("example without a valid split")
    for key, label in (("norm_text", "normalized transcript"), ("source_sha256", "source audio hash"),
                       ("source_file", "source file"), ("audio_path", "prepared audio path")):
        where = defaultdict(set)
        for e in examples:
            where[e[key]].add(e["split"])
        leaks = [k for k, v in where.items() if len(v) > 1]
        if leaks:
            fails.append(f"{len(leaks)} {label}(s) appear in more than one split, e.g. {leaks[0]!r}")
    segs = defaultdict(list)   # overlapping segments of one source file must stay together (none expected for base clips)
    for e in examples:
        if e.get("segment"):
            segs[e["source_file"]].append(e)
    for f, es in segs.items():
        es.sort(key=lambda e: e["segment"][0])
        for a, b in zip(es, es[1:]):
            if a["segment"][1] > b["segment"][0] and a["split"] != b["split"]:
                fails.append(f"overlapping segments of {f} cross splits")
    n = len(examples)
    ratios = cfg["split"]
    for sp in SPLITS:
        share = sum(1 for e in examples if e["split"] == sp) / n
        if abs(share - ratios[sp]) > tol:
            fails.append(f"{sp} holds {share:.1%} of the examples, wanted {ratios[sp]:.0%} +- {tol:.0%}")
        for s in styles:
            if not any(e["split"] == sp and e["style"] == s for e in examples):
                fails.append(f"style '{s}' is missing from the {sp} split")
    for s in styles:
        tot = sum(1 for e in examples if e["style"] == s)
        for sp in SPLITS:
            sh = sum(1 for e in examples if e["style"] == s and e["split"] == sp) / tot
            if abs(sh - ratios[sp]) > 0.04:
                fails.append(f"style '{s}': {sp} share {sh:.1%} is far from {ratios[sp]:.0%}")
    for e in examples:   # prepared audio really exists at the model rate
        p = cfg.path("processed") / e["audio_path"]
        if not p.is_file():
            fails.append(f"prepared audio missing: {e['audio_path']}")
            break
    return fails


def write_splits(examples, cfg, styles):
    out = cfg.path("splits")
    out.mkdir(parents=True, exist_ok=True)
    for sp in SPLITS:
        with open(out / f"{sp}.jsonl", "w") as f:
            for e in sorted((e for e in examples if e["split"] == sp), key=lambda e: e["id"]):
                f.write(json.dumps(manifest_row(e), ensure_ascii=False) + "\n")
    rep = split_report(examples, styles)
    write_json(out / "split_report.json", rep)
    lines = ["# Split report", "", f"Seed {cfg['split']['seed']}, ratios {cfg['split']['train']}/{cfg['split']['validation']}/{cfg['split']['test']}. "
             "Counts are examples; 'distinct texts' are normalized transcripts.", "",
             "| split | style | examples | distinct texts | duration |", "|---|---|---|---|---|"]
    for sp in SPLITS:
        for s in styles:
            r = rep[sp]["per_style"][s]
            lines.append(f"| {sp} | {s} | {r['examples']} | {r['distinct_transcripts']} | {r['duration_s'] / 60:.1f} min |")
        lines.append(f"| **{sp}** | all | {rep[sp]['examples']} | {rep[sp]['distinct_transcripts']} | {rep[sp]['duration_s'] / 60:.1f} min |")
    (out / "split_report.md").write_text("\n".join(lines) + "\n")
    return rep


MANIFEST_FIELDS = ("id", "source_file", "audio_path", "text", "text_original", "speaker", "style", "description",
                   "duration_s", "sample_rate", "split", "group_id", "source_sha256", "segment")


def manifest_row(e):
    return {k: e[k] for k in MANIFEST_FIELDS}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--verify-only", action="store_true", help="re-verify the saved manifests and splits, change nothing")
    ap.add_argument("--refreeze", action="store_true", help="allow replacing a frozen test split (starts a new experiment)")
    args = ap.parse_args()
    cfg = Config(args.config)
    cfg.require_data_root()
    styles = cfg["speaker"]["styles"]
    speaker = cfg.speaker()
    proc, splits_dir = cfg.path("processed"), cfg.path("splits")
    manifest_path = proc / "manifest.jsonl"

    if args.verify_only:
        examples = [json.loads(l) for l in manifest_path.read_text().splitlines()]
        for e in examples:
            e["norm_text"] = normalize_transcript(e["text"])
            e["segment"] = e.get("segment")
        fails = verify(examples, cfg, styles)
        for sp in SPLITS:   # split files must agree with the manifest
            ids = {json.loads(l)["id"] for l in (splits_dir / f"{sp}.jsonl").read_text().splitlines()}
            if ids != {e["id"] for e in examples if e["split"] == sp}:
                fails.append(f"{sp}.jsonl disagrees with manifest.jsonl")
        frozen = splits_dir / "FROZEN.json"
        if frozen.is_file() and sha256_file(splits_dir / "test.jsonl") != read_json(frozen)["test_sha256"]:
            fails.append("test.jsonl changed after it was frozen")
        print("VERIFY FAILED:\n  " + "\n  ".join(fails) if fails else f"verify OK: {len(examples)} examples, no leaks, balanced")
        sys.exit(1 if fails else 0)

    # ---- model rate
    base = model_dir(cfg["training"]["base_model"])
    target_sr, found = model_sample_rate(base)
    log(f"model sample rate {target_sr} Hz (from {', '.join(found)})")

    # ---- candidates
    items, _ = discover(cfg.path("raw"))
    texts = load_transcripts(cfg.path("raw"))
    corpora, excl_sub = set(cfg["speaker"]["corpora"]), set(cfg["speaker"]["exclude_substyles"])
    sel = [it for it in items if it["speaker"] == speaker and it["style"] in styles and it["corpus"] in corpora
           and it["substyle"] not in excl_sub]
    per_style = {s: sum(1 for i in sel if i["style"] == s) for s in styles}
    log(f"{speaker}: {len(sel)} candidate clips {per_style}")
    if not sel:
        raise SystemExit("no candidate clips: check [speaker] in the config against the inventory")

    # a clean slate for processed audio so stale files from an earlier configuration cannot linger
    audio_root = proc / "audio"
    remove_tree(audio_root)
    jobs = [(str(it["path"]), str(audio_root / speaker / f"{it['stem']}.wav"), target_sr, texts.get(it["stem"]), cfg["prepare"]) for it in sel]
    results = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(_process, jobs, chunksize=8), 1):
            results.append(r)
            if i % 250 == 0:
                log(f"  {i}/{len(jobs)} processed")

    # ---- accept / exclude
    examples, excluded = [], []
    first_by_hash = {}
    for it, job, r in zip(sel, jobs, results):
        text_orig = texts.get(it["stem"])
        reasons = [{"code": c, "detail": d} for c, d in r["issues"]]
        if not reasons:
            if r["resample_len_error"] > 2 or abs(r["resample_rms_db_diff"]) > 0.1:
                reasons.append({"code": "resample_check", "detail": f"length off by {r['resample_len_error']} samples, level {r['resample_rms_db_diff']} dB"})
            elif r["sha256"] in first_by_hash:
                reasons.append({"code": "duplicate_audio", "detail": f"identical bytes to {first_by_hash[r['sha256']]}"})
        if reasons:
            excluded.append({"id": it["stem"], "source_file": it["file"], "style": it["style"], "reasons": reasons})
            Path(job[1]).unlink(missing_ok=True)
            continue
        first_by_hash[r["sha256"]] = it["stem"]
        text = strip_emphasis_markers(text_orig)
        examples.append({
            "id": it["stem"], "source_file": it["file"], "audio_path": f"audio/{speaker}/{it['stem']}.wav", "text": text,
            "text_original": text_orig, "speaker": speaker, "style": it["style"],
            "description": description_for(cfg, speaker, it["style"]), "duration_s": r["out_duration_s"],
            "sample_rate": r["out_sample_rate"], "source_sha256": r["sha256"], "segment": None,
            "norm_text": normalize_transcript(text), "source_duration_s": r["duration_s"],
        })

    # transcript/audio plausibility: characters per second of the whole clip, judged within (style),
    # robust z-score. Catches a transcript attached to the wrong audio without any ASR.
    for s in styles:
        es = [e for e in examples if e["style"] == s]
        rates = [len(e["text"]) / e["duration_s"] for e in es]
        med = statistics.median(rates)
        mad = statistics.median(abs(r - med) for r in rates) or 1e-9
        for e, r in zip(es, rates):
            z = 0.6745 * (r - med) / mad
            if abs(z) > 6:
                excluded.append({"id": e["id"], "source_file": e["source_file"], "style": s, "reasons": [
                    {"code": "transcript_audio_mismatch_suspected", "detail": f"{r:.1f} chars/s vs style median {med:.1f} (robust z {z:.1f}); text={e['text']!r}"}]})
                (proc / e["audio_path"]).unlink(missing_ok=True)
        examples = [e for e in examples if e["id"] not in {x["id"] for x in excluded}]

    by_id = {it["stem"]: r for it, r in zip(sel, results)}
    accepted = {s: [] for s in styles}
    for e in examples:
        r = by_id[e["id"]]
        accepted[e["style"]].append(dict(r, chars_per_active_s=len(e["text"]) / max(e["duration_s"] * r["active_fraction"], 0.1)))
    pitch = defaultdict(lambda: defaultdict(list))
    inv_files = cfg.path("inventory") / "files.jsonl"
    if inv_files.is_file():
        keep = {e["id"] for e in examples}
        for line in inv_files.read_text().splitlines():
            f = json.loads(line)
            if f["stem"] in keep and f.get("f0_median_hz") is not None:
                pitch[f["style"]]["f0_median_hz"].append(f["f0_median_hz"])
                pitch[f["style"]]["f0_range_st"].append(f["f0_range_st"])
    checks = check_descriptions(cfg, speaker, styles, accepted, pitch)
    write_json(proc / "description_checks.json", checks)
    for c in checks["claims"]:
        log(f"  [{c['verdict']}] {c['claim']}: {c['evidence']}")

    log(f"accepted {len(examples)} / {len(sel)}; excluded {len(excluded)}: "
        f"{dict(Counter(r['code'] for x in excluded for r in x['reasons']))}")
    if len(examples) < 100:
        raise SystemExit("too few usable examples to fine-tune on; see exclusions.jsonl")

    # ---- splits
    ratios = {sp: cfg["split"][sp] for sp in SPLITS}
    if abs(sum(ratios.values()) - 1) > 1e-9:
        raise SystemExit(f"split ratios must sum to 1: {ratios}")
    examples.sort(key=lambda e: e["id"])
    assign = assign_splits(examples, ratios, styles, cfg["split"]["seed"])
    groups = group_examples(examples)
    gid = {}
    for g in groups:
        name = "g_" + min(examples[i]["id"] for i in g)
        for i in g:
            gid[i] = name
    for i, e in enumerate(examples):
        e["split"] = assign[i]
        e["group_id"] = gid[i]
    fails = verify(examples, cfg, styles)
    if fails:
        raise SystemExit("split verification failed:\n  " + "\n  ".join(fails))
    again = assign_splits(examples, ratios, styles, cfg["split"]["seed"])
    if again != assign:
        raise SystemExit("split assignment is not deterministic")

    # ---- frozen test split: reproduce exactly or refuse
    frozen_path = splits_dir / "FROZEN.json"
    test_ids_new = sorted(e["id"] for e in examples if e["split"] == "test")
    if frozen_path.is_file() and not args.refreeze:
        old = [json.loads(l)["id"] for l in (splits_dir / "test.jsonl").read_text().splitlines()]
        if sorted(old) != test_ids_new:
            raise SystemExit("this run would change the FROZEN test split (the data, gates or seed changed). "
                             "Fix the cause, or pass --refreeze to deliberately start a new experiment.")
    rep = write_splits(examples, cfg, styles)
    proc.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        for e in examples:
            f.write(json.dumps(manifest_row(e), ensure_ascii=False) + "\n")
    with open(proc / "exclusions.jsonl", "w") as f:
        for x in sorted(excluded, key=lambda x: x["id"]):
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    write_json(frozen_path, {"frozen_at": utc_now(), "test_sha256": sha256_file(splits_dir / "test.jsonl"),
                             "n_test": len(test_ids_new), "seed": cfg["split"]["seed"]})
    write_json(proc / "prepare_info.json", {
        "prepared_at": utc_now(), "speaker": speaker, "styles": styles, "model_sample_rate": target_sr,
        "model_sample_rate_sources": found, "candidates": len(sel), "accepted": len(examples), "excluded": len(excluded),
        "exclusion_counts": dict(Counter(r["code"] for x in excluded for r in x["reasons"])),
        "gates": cfg["prepare"], "source_license": cfg["source"]["license"], "attribution": cfg["source"]["attribution"],
        "processing": "decode -> soxr resample (no denoising, trimming or normalization) -> write", "report": rep})
    log((splits_dir / "split_report.md").read_text())
    log(f"verified: no transcript, source file or audio hash crosses a split. Prepared audio: {proc / 'audio' / speaker}")


if __name__ == "__main__":
    main()
