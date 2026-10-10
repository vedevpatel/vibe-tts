"""Inventory of the extracted Expresso read speech: what is there, what is usable, which speaker to use.

Usage: .venv/bin/python scripts/inventory_expresso.py [--config PATH] [--workers N]

Reads <data root>/raw (see scripts/download_expresso.py) and writes to <data root>/inventory/:
  files.jsonl            one record per WAV: ids, sha256, full-decode measurements, transcript, issues
  inventory.json / .md   per speaker x style x corpus: clips, usable clips, duration, sample rates,
                         channels, transcript coverage; plus every missing / corrupt / duplicate finding
  speaker_selection.json the ranking and the reason for the chosen speaker
  acoustics.json         level / noise floor / pitch / pace per style for the target styles
and exports a listening pack of ORIGINAL default and sad recordings to <data root>/listening_pack/.

Every WAV is decoded completely (not just its header). "Usable" uses exactly the gates that
scripts/prepare_expresso.py applies, so the inventory predicts what training will see.
Only read speech is inventoried; improvised dialogue is not even extracted, and singing / any style
outside the configured ones is reported but never selected.
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

from _common import ROOT  # first: loads .env
from expresso_common import (READ_STYLES, Config, analyze_audio, check_descriptions, is_junk, remove_tree, load_transcripts, log, normalize_transcript, parse_stem,
                             quality_issues, read_json, sha256_file, strip_emphasis_markers, utc_now, write_json)


PITCH_SAMPLE = 150


def _work(args):
    """Worker: hash + full decode + (optionally) pitch for one file. Top-level so it pickles."""
    path, want_pitch, text = args
    rec = analyze_audio(path)
    rec["sha256"] = sha256_file(path)
    rec["size_bytes"] = os.path.getsize(path)
    if want_pitch and not rec["error"]:
        rec.update(pitch_profile(path, text))
    return rec


def pitch_profile(path, text):
    """F0 median / range (pyin on 16 kHz mono) and a pace proxy (characters per second of active speech)."""
    import numpy as np
    import librosa
    import soundfile as sf
    import soxr

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    m = soxr.resample(x.mean(axis=1), sr, 16000)
    f0, voiced, _ = librosa.pyin(m, fmin=65, fmax=500, sr=16000, frame_length=1024, hop_length=256)
    f = f0[voiced & np.isfinite(f0)]
    out = {"f0_median_hz": None, "f0_range_st": None, "voiced_fraction": round(float(voiced.mean()), 4)}
    if len(f) >= 10:
        lo, hi = np.percentile(f, [10, 90])
        out["f0_median_hz"] = round(float(np.median(f)), 1)
        out["f0_range_st"] = round(float(12 * np.log2(hi / lo)), 2)
    return out


def dur_fmt(s):
    s = int(round(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def discover(raw):
    base = raw / "expresso" / "audio_48khz" / "read"
    if not base.is_dir():
        raise SystemExit(f"{base} not found: run scripts/download_expresso.py")
    items, layout_problems = [], []
    for p in sorted(base.rglob("*.wav")):
        if is_junk(p):
            continue
        rel = p.relative_to(base).parts          # speaker/style/corpus/file.wav
        parsed = parse_stem(p.stem)
        if len(rel) != 4 or parsed is None:
            layout_problems.append({"file": str(p.relative_to(raw)), "problem": "path does not follow read/{speaker}/{style}/{corpus}/{file}.wav"})
            continue
        speaker, style, corpus = rel[0], rel[1], rel[2]
        if (parsed["speaker"], parsed["style"]) != (speaker, style):
            layout_problems.append({"file": str(p.relative_to(raw)), "problem": f"directory says {speaker}/{style}, filename says {parsed['speaker']}/{parsed['style']}"})
        items.append({"path": p, "file": str(p.relative_to(raw)), "stem": p.stem, "speaker": speaker, "style": style,
                      "corpus": corpus, "substyle": parsed["substyle"]})
    return items, layout_problems


def official_read_stems(raw):
    stems = set()
    for name in ("train", "dev", "test"):
        f = raw / "expresso" / "splits" / f"{name}.txt"
        if f.is_file():
            for line in f.read_text().splitlines():
                if line and not line.startswith("#"):
                    s = line.split("\t")[0].strip()
                    if "-" not in s.split("_")[0]:    # conversational stems are 'ex01-ex02_...'
                        stems.add(s)
    return stems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--pack-only", action="store_true", help="rebuild only the listening pack from the saved files.jsonl")
    ap.add_argument("--from-saved", action="store_true",
                    help="reuse the measurements in files.jsonl (no re-decoding) and recompute gates, selection and reports; "
                         "valid only while the raw data is unchanged")
    args = ap.parse_args()
    cfg = Config(args.config)
    cfg.require_data_root()
    raw, inv_dir = cfg.path("raw"), cfg.path("inventory")
    if args.pack_only:
        files = [json.loads(l) for l in (inv_dir / "files.jsonl").read_text().splitlines()]
        speaker = read_json(inv_dir / "speaker_selection.json")["selected"]
        log(f"listening pack: {export_listening_pack(cfg, files, speaker, cfg['speaker']['styles'], raw)}")
        return
    prep = cfg["prepare"]
    styles = cfg["speaker"]["styles"]
    corpora = set(cfg["speaker"]["corpora"])
    excl_sub = set(cfg["speaker"]["exclude_substyles"])

    items, layout_problems = discover(raw)
    texts, transcript_problems = load_transcripts(raw, with_problems=True)
    log(f"{len(items)} read-speech WAVs, {len(texts)} transcripts; decoding everything with {args.workers} workers")

    if args.from_saved:
        saved = {}
        for line in (inv_dir / "files.jsonl").read_text().splitlines():
            rec = json.loads(line)
            saved[rec["file"]] = rec
        missing = [it["file"] for it in items if it["file"] not in saved]
        if missing:
            raise SystemExit(f"{len(missing)} files are not in the saved measurements (e.g. {missing[0]}): run without --from-saved")
        recs = [saved[it["file"]] for it in items]
        log(f"reusing saved measurements for {len(recs)} files")
    else:
        # Pitch tracking is the slow part, and medians do not need every clip: measure it on a seeded sample of
        # PITCH_SAMPLE clips per (speaker, target style). Every other measurement covers every file.
        pool = defaultdict(list)
        for i, it in enumerate(items):
            if it["style"] in styles and it["corpus"] in corpora and it["substyle"] not in excl_sub:
                pool[(it["speaker"], it["style"])].append(i)
        want_pitch = set()
        for key in sorted(pool):
            want_pitch.update(random.Random(f"{cfg['split']['seed']}-{key}").sample(pool[key], min(PITCH_SAMPLE, len(pool[key]))))
        jobs = [(str(it["path"]), i in want_pitch, texts.get(it["stem"])) for i, it in enumerate(items)]
        recs = []
        with ProcessPoolExecutor(args.workers) as ex:
            for i, r in enumerate(ex.map(_work, jobs, chunksize=16), 1):
                recs.append(r)
                if i % 1000 == 0:
                    log(f"  {i}/{len(items)} decoded")

    files = []
    for it, r in zip(items, recs):
        it = {k: v for k, v in it.items() if k != "path"}
        text = texts.get(it["stem"])
        issues = quality_issues(r, text, prep)
        if it["style"] not in READ_STYLES:
            issues.append(("not_read_style", f"style '{it['style']}' is not a read-speech style"))
        files.append({**it, **r, "transcript": text, "issues": [list(i) for i in issues]})

    # ---- findings
    problems = {"layout": layout_problems, "transcript_file": transcript_problems}
    problems["corrupt"] = [{"file": f["file"], "problem": f["error"]} for f in files if f["error"]]
    problems["audio_without_transcript"] = [f["file"] for f in files if f["transcript"] is None]
    on_disk = {f["stem"] for f in files}
    problems["transcript_without_audio"] = sorted(s for s in texts if parse_stem(s) and s not in on_disk)
    official = official_read_stems(raw)
    problems["listed_in_official_splits_but_missing"] = sorted(official - on_disk)
    problems["on_disk_but_not_in_official_splits"] = sorted(s for s in on_disk - official if "longform" not in s)
    by_hash = defaultdict(list)
    for f in files:
        if f.get("sha256"):
            by_hash[f["sha256"]].append(f["file"])
    problems["duplicate_files_identical_bytes"] = [v for v in by_hash.values() if len(v) > 1]
    by_text = defaultdict(list)
    for f in files:
        if f["transcript"]:
            by_text[(f["speaker"], f["style"], normalize_transcript(f["transcript"]))].append(f["stem"])
    problems["repeated_transcript_same_speaker_style"] = [v for v in by_text.values() if len(v) > 1]
    problems["unexpected_styles"] = sorted({f["style"] for f in files if f["style"] not in READ_STYLES})
    stem_count = Counter(f["stem"] for f in files)
    problems["duplicate_stems"] = [s for s, c in stem_count.items() if c > 1]

    # ---- summary per speaker x style x corpus (+ substyle)
    def selected(f):
        return f["corpus"] in corpora and f["substyle"] not in excl_sub and f["style"] in styles

    groups = defaultdict(list)
    for f in files:
        groups[(f["speaker"], f["style"], f["corpus"], f["substyle"] or "-")].append(f)
    rows = []
    for (spk, sty, corp, sub), fs in sorted(groups.items()):
        usable = [f for f in fs if not f["issues"]]
        rows.append({
            "speaker": spk, "style": sty, "corpus": corp, "substyle": sub, "clips": len(fs), "usable": len(usable),
            "total_s": round(sum(f.get("duration_s") or 0 for f in fs), 1),
            "usable_s": round(sum(f["duration_s"] for f in usable), 1),
            "sample_rates": sorted({f["sample_rate"] for f in fs if f.get("sample_rate")}),
            "channels": sorted({f["channels"] for f in fs if f.get("channels")}),
            "subtypes": sorted({f["subtype"] for f in fs if f.get("subtype")}),
            "with_transcript": sum(1 for f in fs if f["transcript"] is not None),
            "issue_counts": dict(Counter(code for f in fs for code, _ in f["issues"])),
            "median_noise_floor_dbfs": median([f.get("noise_floor_dbfs") for f in fs]),
        })

    # ---- per-speaker usable data, acoustics, and the measured description claims
    cand, acoustics, claims = {}, {}, {}
    for spk in sorted({f["speaker"] for f in files}):
        per, accepted, pitch = {}, {}, defaultdict(lambda: defaultdict(list))
        for sty in styles:
            u = [f for f in files if f["speaker"] == spk and f["style"] == sty and selected(f) and not f["issues"]]
            per[sty] = {"usable_clips": len(u), "usable_s": round(sum(f["duration_s"] for f in u), 1),
                        "distinct_transcripts": len({normalize_transcript(f["transcript"]) for f in u}),
                        "median_noise_floor_dbfs": median([f["noise_floor_dbfs"] for f in u]),
                        "median_snr_proxy_db": median([f["snr_proxy_db"] for f in u])}
            accepted[sty] = [dict(f, chars_per_active_s=len(strip_emphasis_markers(f["transcript"])) / max(f["duration_s"] * f["active_fraction"], 0.1))
                             for f in u if f.get("active_fraction")]
            for f in u:
                if f.get("f0_median_hz") is not None:
                    pitch[sty]["f0_median_hz"].append(f["f0_median_hz"])
                    pitch[sty]["f0_range_st"].append(f["f0_range_st"])
        cand[spk] = per
        chk = check_descriptions(cfg, spk, styles, accepted, pitch)
        claims[spk] = chk
        acoustics[spk] = chk["per_style"]
    write_json(inv_dir / "acoustics.json", acoustics)

    # ---- speaker selection, one rule applied identically to every speaker:
    #   1. adequate data: >= min_usable_clips_per_style usable clips in EVERY target style;
    #   2. the descriptions must be TRUE of the recordings: no measured claim may come out "NOT supported"
    #      (clean-recording thresholds; sad lower than default in enough indicators);
    #   3. among those, most usable audio in the scarcer style (data is what a small fine-tune is short of).
    need = cfg["speaker"]["min_usable_clips_per_style"]
    verdict = {}
    for spk in cand:
        adequate = all(cand[spk][y]["usable_clips"] >= need for y in styles)
        failed = [c["claim"] for c in claims[spk]["claims"] if c["verdict"] == "NOT supported"]
        verdict[spk] = {"adequate_data": adequate, "claims_failed": failed, "eligible": adequate and not failed,
                        "min_style_usable_s": min(cand[spk][y]["usable_s"] for y in styles),
                        "total_usable_s": sum(cand[spk][y]["usable_s"] for y in styles)}
    ranked = sorted(cand, key=lambda s: (verdict[s]["eligible"], verdict[s]["min_style_usable_s"], verdict[s]["total_usable_s"]), reverse=True)
    eligible = [s for s in ranked if verdict[s]["eligible"]]
    if not eligible:
        raise SystemExit(f"no speaker is eligible (need >= {need} usable clips per style and true description claims): {json.dumps(verdict)}")
    pinned = cfg["speaker"]["id"]
    auto_pick = eligible[0]
    chosen = auto_pick if pinned == "auto" else pinned
    if chosen not in cand:
        raise SystemExit(f"configured speaker {chosen!r} not in the data (have {sorted(cand)})")
    if pinned != "auto" and not verdict[chosen]["eligible"]:
        raise SystemExit(f"pinned speaker {chosen} is not eligible: adequate data={verdict[chosen]['adequate_data']}, failed claims={verdict[chosen]['claims_failed']}")
    why = (f"Rule: (1) >= {need} usable clips in every target style; (2) every measured claim in the descriptions holds for that speaker's "
           f"recordings; (3) of those, the most usable audio in the scarcer style. " +
           "; ".join(f"{s}: " + ("eligible" if verdict[s]["eligible"] else "REJECTED (" + ("; ".join(verdict[s]["claims_failed"]) or "too little data") + ")") +
                     f", scarcer style {dur_fmt(verdict[s]['min_style_usable_s'])}" for s in ranked) + f". Rule picks {auto_pick}.")
    write_json(inv_dir / "speaker_selection.json", {
        "selected": chosen, "rule_pick": auto_pick, "pinned_in_config": pinned != "auto", "reason": why,
        "styles": styles, "corpora": sorted(corpora), "excluded_substyles": sorted(excl_sub),
        "verdicts": verdict, "candidates": cand, "claim_checks": claims, "decided_at": utc_now()})

    # ---- write inventory
    write_json(inv_dir / "inventory.json", {"generated_at": utc_now(), "release": read_json(raw / "SOURCES.json") if (raw / "SOURCES.json").is_file() else None,
                                            "gates": prep, "rows": rows, "problems": problems, "selection": cand})
    with open(inv_dir / "files.jsonl", "w") as f:
        for r in files:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    md = render_markdown(rows, problems, cand, ranked, why, chosen, styles, acoustics, files, prep, verdict, claims)
    (inv_dir / "inventory.md").write_text(md)

    # ---- listening pack of originals
    pack = export_listening_pack(cfg, files, chosen, styles, raw)
    log(md.split("## Findings")[0])
    log(f"selected speaker: {chosen}   ({why})")
    log(f"inventory: {inv_dir}   listening pack: {pack}")


def render_markdown(rows, problems, cand, ranked, why, chosen, styles, acoustics, files, prep, verdict, claims):
    L = ["# Expresso read-speech inventory", "",
         f"Generated {utc_now()}. Every WAV decoded in full. Usable = decodes, finite, mono, {prep['min_duration_s']}-{prep['max_duration_s']} s, "
         f"clipping <= {prep['max_clipping_fraction']:.1%}, not silent, has a transcript. Durations are h:mm:ss.", "",
         "## Target styles per speaker (base corpus, configured sub-styles)", "",
         "| speaker | " + " | ".join(f"{y}: clips / usable / duration / noise floor" for y in styles) + " |",
         "|---|" + "---|" * len(styles)]
    for s in ranked:
        L.append(f"| {s}{' (selected)' if s == chosen else ''} | " + " | ".join(
            f"{cand[s][y]['usable_clips']} clips, {cand[s][y]['distinct_transcripts']} distinct texts, {dur_fmt(cand[s][y]['usable_s'])}, "
            f"{cand[s][y]['median_noise_floor_dbfs']} dBFS" for y in styles) + " |")
    L += ["", f"**Speaker choice.** {why}", "", "### Description claims tested against each speaker's recordings", "",
          "| speaker | eligible | claim | verdict | evidence |", "|---|---|---|---|---|"]
    for sp in ranked:
        for c in claims[sp]["claims"]:
            if c["verdict"] in ("supported", "NOT supported"):
                L.append(f"| {sp} | {'yes' if verdict[sp]['eligible'] else 'no'} | {c['claim']} | {c['verdict']} | {c['evidence']} |")
    L += [""]
    L += ["", "## Every speaker x style x corpus", "",
          "| speaker | style | corpus | sub-style | clips | usable | transcripts | total | usable | sample rates | channels | issues |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['speaker']} | {r['style']} | {r['corpus']} | {r['substyle']} | {r['clips']} | {r['usable']} | {r['with_transcript']} | "
                 f"{dur_fmt(r['total_s'])} | {dur_fmt(r['usable_s'])} | {', '.join(map(str, r['sample_rates']))} | {', '.join(map(str, r['channels']))} | "
                 f"{', '.join(f'{k}: {v}' for k, v in r['issue_counts'].items()) or '-'} |")
    L += ["", "## Acoustics of the target styles (medians over usable clips)", "",
          "Pitch is measured on a seeded sample of clips per speaker and style (column 'pitch n').", "",
          "| speaker | style | speech level dBFS | noise floor median / p95 dBFS | active fraction | chars / active s | F0 median Hz | F0 range st | pitch n |",
          "|---|---|---|---|---|---|---|---|---|"]
    for spk in sorted(acoustics):
        for sty in styles:
            a = acoustics[spk][sty]
            L.append(f"| {spk} | {sty} | {a['speech_level_dbfs']} | {a['noise_floor_median_dbfs']} / {a['noise_floor_p95_dbfs']} | {a['active_fraction']} | "
                     f"{a['chars_per_active_s']} | {a['f0_median_hz']} | {a['f0_range_st']} | {a['pitch_clips_measured']} |")
    L += ["", "## Findings", ""]
    for name, val in problems.items():
        L.append(f"- **{name}**: {len(val)}" + (f" - e.g. {json.dumps(val[:3], ensure_ascii=False)}" if val else ""))
    return "\n".join(L) + "\n"


def export_listening_pack(cfg, files, speaker, styles, raw):
    """Original (48 kHz / 24-bit) recordings of the same sentences in both styles, to listen to before trusting any result."""
    out = cfg.path("listening_pack")
    remove_tree(out)
    out.mkdir(parents=True)
    ok = [f for f in files if f["speaker"] == speaker and f["style"] in styles and f["corpus"] in cfg["speaker"]["corpora"]
          and f["substyle"] not in cfg["speaker"]["exclude_substyles"] and not f["issues"] and 2.5 <= f["duration_s"] <= 7.0]
    by_text = defaultdict(dict)
    for f in sorted(ok, key=lambda f: f["stem"]):
        by_text[normalize_transcript(f["transcript"])].setdefault(f["style"], f)
    both = sorted(k for k, v in by_text.items() if all(s in v for s in styles))
    picks = random.Random(cfg["split"]["seed"]).sample(both, min(6, len(both)))
    lines = [f"# Listening pack: original Expresso recordings, speaker {speaker}", "",
             "Real recordings (48 kHz, 24-bit, mono), unmodified copies. The same sentence is spoken in each style so the",
             "difference in delivery is audible. Listen to these before judging anything the model produces.", "",
             "| n | sentence | " + " | ".join(styles) + " |", "|---|---|" + "---|" * len(styles)]
    for n, key in enumerate(picks, 1):
        cells = []
        for s in styles:
            f = by_text[key][s]
            name = f"{n:02d}_{s}_{f['stem']}.wav"
            shutil.copyfile(raw / f["file"], out / name)
            cells.append(f"`{name}` ({f['duration_s']:.1f}s)")
        lines.append(f"| {n} | {strip_emphasis_markers(by_text[key][styles[0]]['transcript'])} | " + " | ".join(cells) + " |")
    lines += ["", "## License and attribution", "", f"Expresso is distributed under {cfg['source']['license']} ({cfg['source']['license_url']}). "
              "Noncommercial use only; attribution required:", "", f"> {cfg['source']['attribution']}", ""]
    (out / "README.md").write_text("\n".join(lines))
    lic = raw / "expresso" / "LICENSE.txt"
    if lic.is_file():
        shutil.copyfile(lic, out / "LICENSE.txt")
    return out


if __name__ == "__main__":
    main()
