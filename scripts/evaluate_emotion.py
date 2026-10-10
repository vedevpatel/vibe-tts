"""Before/after evaluation for the Expresso emotion experiment.

Subcommands (all read configs/expresso_emotion.toml; run from any directory):

  select    Freeze the evaluation set: N seeded test-split transcripts (both styles exist) plus the
            configured new sentences (checked to be absent from the whole release). Writes
            <splits>/eval_set.json. Run once; generate refuses to run without it.
  generate  Synthesize every (text, style, seed) with one model. The SAME eval set, descriptions,
            seeds, generation settings and waveform handling are used for every model:
              evaluate_emotion.py generate --label baseline
              evaluate_emotion.py generate --label finetuned --model <checkpoint dir>
            Raw 32-bit float WAVs and metadata.jsonl go to <output dir>/expresso/runs/<label>/.
  page      Build a blinded listening page (opaque shuffled clip labels) from generated runs, plus the
            real held-out recordings as hidden controls. The answer key is written NEXT TO the page
            directory, never inside it.
  score     Aggregate the ratings a listener exported from the page, using the answer key.
  measure   Descriptive acoustics (duration, pace, level, pitch, clipping) per run and style, next to the real
            held-out recordings. Measurements, not ratings: they show whether a model's two styles differ in the
            same measurable ways as the real ones, which says nothing about how they sound.

Nothing here rates audio automatically. No claim about emotion, naturalness or speaker identity is
made without listener ratings.

Limitation to keep in mind: the unchanged base model has never heard this Expresso speaker. Its
"ex01" voice is whatever the name token happens to evoke, and it differs from clip to clip. The
baseline therefore cannot show speaker consistency with the target speaker, and ordinary adaptation
does not preserve any other speaker's identity either.
"""
import argparse
import csv
import html
import json
import random
import shutil
import statistics
import string
import sys
import time
from collections import defaultdict
from pathlib import Path

from _common import ROOT, model_dir, pick_device  # first: loads .env before transformers reads HF_HOME
from expresso_common import (Config, description_for, log, normalize_transcript, read_json, remove_tree, sha256_file,
                             utc_now, write_json)

REFERENCE_RUN = "reference"  # label used for the real held-out recordings in the listening page


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def eval_set_path(cfg):
    return cfg.path("splits") / "eval_set.json"


def all_release_transcripts(cfg):
    """Normalized text of every transcript in the release (for the 'new sentences are new' check)."""
    from expresso_common import load_transcripts

    return {normalize_transcript(t) for t in load_transcripts(cfg.path("raw")).values()}


# ---------------------------------------------------------------- select
def cmd_select(cfg, args):
    ev = cfg["eval"]
    speaker = cfg.speaker()
    test_path = cfg.path("splits") / "test.jsonl"
    if not test_path.is_file():
        raise SystemExit(f"{test_path} not found: run scripts/prepare_expresso.py first")
    if eval_set_path(cfg).is_file() and not args.force:
        raise SystemExit(f"{eval_set_path(cfg)} already exists (the evaluation set is frozen). Use --force only to start a new experiment.")
    styles = cfg["speaker"]["styles"]
    by_text = defaultdict(dict)
    for ex in load_jsonl(test_path):
        by_text[normalize_transcript(ex["text"])].setdefault(ex["style"], ex)
    both = sorted(k for k, v in by_text.items() if all(s in v for s in styles))
    n = ev["n_test_transcripts"]
    if len(both) < n:
        log(f"warning: only {len(both)} test transcripts exist in all styles; using all of them (wanted {n})", err=True)
        n = len(both)
    rng = random.Random(ev["seed"])
    chosen = rng.sample(both, n)
    items = []
    for i, key in enumerate(sorted(chosen)):
        exs = by_text[key]
        items.append({
            "item_id": f"test{i:02d}", "source": "test", "text": exs[styles[0]]["text"],
            "reference_examples": {s: exs[s]["id"] for s in styles},
        })
    release = all_release_transcripts(cfg)
    for i, text in enumerate(ev["new_sentences"]):
        if normalize_transcript(text) in release:
            raise SystemExit(f"new sentence {text!r} occurs in the dataset transcripts; replace it in the config")
        items.append({"item_id": f"new{i:02d}", "source": "new", "text": text, "reference_examples": {}})
    write_json(eval_set_path(cfg), {
        "created_at": utc_now(), "speaker": speaker, "styles": styles, "seeds": ev["seeds"],
        "descriptions": {s: description_for(cfg, speaker, s) for s in styles},
        "selection_seed": ev["seed"], "items": items,
    })
    log(f"froze {len(items)} items ({n} test + {len(ev['new_sentences'])} new) x {len(styles)} styles x "
        f"{len(ev['seeds'])} seeds = {len(items) * len(styles) * len(ev['seeds'])} clips -> {eval_set_path(cfg)}")


# ---------------------------------------------------------------- generate
def cmd_generate(cfg, args):
    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    from transformers.cache_utils import StaticCache

    # parler-tts reads StaticCache.max_batch_size, which transformers 4.46.1 renamed to batch_size
    # (same shim as scripts/speak.py).
    if not hasattr(StaticCache, "max_batch_size"):
        StaticCache.max_batch_size = property(lambda self: self.batch_size)

    if not eval_set_path(cfg).is_file():
        raise SystemExit(f"{eval_set_path(cfg)} missing: run `evaluate_emotion.py select` first")
    es = read_json(eval_set_path(cfg))
    gen = cfg["eval"]["generation"]
    base_dir = model_dir(cfg["training"]["base_model"])
    model_path = Path(args.model).expanduser().resolve() if args.model else base_dir
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"no model at {model_path}")
    out = cfg.path("eval") / "runs" / args.label
    (out / "raw").mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.jsonl"

    device = pick_device()
    dtype = {"float32": torch.float32}[gen["dtype"]]
    log(f"run '{args.label}': model={model_path} device={device} dtype={gen['dtype']}")

    rev = {}
    for name in ("REVISION.json", "export_info.json"):
        if (model_path / name).is_file():
            rev = read_json(model_path / name)
            break
    weights = model_path / "model.safetensors"
    checkpoint = {
        "path": str(model_path), "is_base_model": model_path == base_dir, "record": rev,
        "weights_sha256": rev.get("weights_sha256") or (None if args.no_hash else sha256_file(weights)),
    }

    tokenizer = AutoTokenizer.from_pretrained(base_dir)  # identical tokenizer for every model
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_path, attn_implementation="eager")
    model = model.to(device, dtype=dtype).eval()
    sr = model.config.sampling_rate
    gen_kwargs = {"do_sample": gen["do_sample"], "temperature": gen["temperature"],
                  "max_length": gen["max_length"], "min_new_tokens": gen["min_new_tokens"]}

    done = {r["clip_id"] for r in load_jsonl(meta_path)} if meta_path.is_file() else set()
    todo = [(it, s, seed) for it in es["items"] for s in es["styles"] for seed in es["seeds"]]
    if args.limit:
        todo = todo[: args.limit]
    log(f"{len(todo)} clips, {len(done)} already done")
    with open(meta_path, "a") as meta_f:
        for n, (it, style, seed) in enumerate(todo, 1):
            clip_id = f"{it['item_id']}_{style}_s{seed}"
            if clip_id in done and (out / "raw" / f"{clip_id}.wav").is_file():
                continue
            desc = es["descriptions"][style]
            d_in = tokenizer(desc, return_tensors="pt").to(device)
            p_in = tokenizer(it["text"], return_tensors="pt").to(device)
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(seed)
            t0 = time.time()
            with torch.no_grad():
                wav = model.generate(
                    input_ids=d_in.input_ids, attention_mask=d_in.attention_mask,
                    prompt_input_ids=p_in.input_ids, prompt_attention_mask=p_in.attention_mask, **gen_kwargs,
                )
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            audio = wav.cpu().float().numpy().squeeze()
            path = out / "raw" / f"{clip_id}.wav"
            sf.write(path, audio, sr, subtype=gen["save_subtype"])  # raw output, no normalization or trimming
            rec = {
                "clip_id": clip_id, "run": args.label, "checkpoint": checkpoint, "item_id": it["item_id"],
                "source": it["source"], "style": style, "text": it["text"], "description": desc, "seed": seed,
                "duration_s": round(len(audio) / sr, 3), "generation_time_s": round(elapsed, 2),
                "sampling_rate": sr, "device": device, "dtype": gen["dtype"], "generation_kwargs": gen_kwargs,
                "peak": float(np.abs(audio).max()) if audio.size else None,
                "nonfinite": int((~np.isfinite(audio)).sum()),
                "wav": str(path.relative_to(out)), "created_at": utc_now(),
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            }
            meta_f.write(json.dumps(rec) + "\n")
            meta_f.flush()
            log(f"[{n}/{len(todo)}] {clip_id}: {rec['duration_s']:.2f}s audio in {elapsed:.1f}s")
    log(f"done: {out}")


# ---------------------------------------------------------------- page
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Listening test __PAGE_ID__</title>
<style>
:root{color-scheme:light dark;--bg:#fafaf8;--fg:#1d1d1b;--mut:#666;--card:#fff;--line:#d9d9d4;--acc:#2a5db0;--ok:#2e7d4f}
@media(prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e8e8e4;--mut:#9a9a94;--card:#202020;--line:#383836;--acc:#7aa7ec;--ok:#6cc08b}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:46rem;margin:0 auto;padding:1rem 1rem 6rem}
h1{font-size:1.4rem}.mut{color:var(--mut)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:1rem;margin:1rem 0}
.card.done{border-color:var(--ok)}audio{width:100%}fieldset{border:0;padding:0;margin:.8rem 0 0}legend{font-weight:600;margin-bottom:.2rem}
label{margin-right:1rem;white-space:nowrap}.scale label{margin-right:.6rem}textarea{width:100%;box-sizing:border-box}
#bar{position:sticky;top:0;background:var(--bg);padding:.5rem 0;border-bottom:1px solid var(--line);z-index:2}
button{font:inherit;padding:.4rem .8rem;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
</style></head><body><main>
<div id="bar"><b id="prog">0 / 0 rated</b> <span class="mut">(saved automatically in this browser)</span>
 <button class="primary" id="dl">Download ratings (JSON)</button> <button id="dlc">CSV</button></div>
<h1>Listening test</h1>
<p>You will hear short clips in random order. For each clip, answer four questions. Do not look for patterns
in the clip names; they carry no information. Use headphones in a quiet room. Please do not skip the
reference voice below: it is a real recording of the target speaker.</p>
<label>Your name or initials (optional): <input id="rater" size="12"></label>
<div class="card"><b>Reference voice</b> <span class="mut">(real recordings of the target speaker, speaking different sentences)</span>
__REFS__</div>
<div id="trials"></div>
</main>
<script>
const PAGE_ID="__PAGE_ID__", TRIALS=__TRIALS__;
const KEY="listening-"+PAGE_ID; let R={};
try{R=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(R))}catch(e){}};
const EM=[["neutral","Neutral"],["sad","Sad"],["other","Some other emotion / mixed"],["unsure","Cannot tell"]];
const Q=[["naturalness","Naturalness","1 = clearly artificial, 5 = indistinguishable from a real person"],
 ["intelligibility","Intelligibility","1 = I could not make out the words, 5 = every word is clear"],
 ["speaker_consistency","Same speaker as the reference voice?","1 = clearly a different person, 5 = clearly the same person"]];
function scale(id,name,cur){return [1,2,3,4,5].map(v=>`<label><input type=radio name="${id}_${name}" value="${v}" ${cur==v?"checked":""}> ${v}</label>`).join("")}
function render(){
 const box=document.getElementById("trials");
 TRIALS.forEach((t,i)=>{const r=R[t.label]||{};const c=document.createElement("div");c.className="card";c.id="c_"+t.label;
  c.innerHTML=`<b>Clip ${i+1} of ${TRIALS.length}</b><br><audio controls preload="none" src="${t.src}"></audio>
  <fieldset><legend>Which emotion does this clip express?</legend>${EM.map(([v,l])=>`<label><input type=radio name="${t.label}_emotion" value="${v}" ${r.emotion==v?"checked":""}> ${l}</label>`).join("")}</fieldset>
  ${Q.map(([k,l,h])=>`<fieldset class=scale><legend>${l} <span class=mut>${h}</span></legend>${scale(t.label,k,r[k])}</fieldset>`).join("")}
  <fieldset><legend>Notes <span class=mut>(optional)</span></legend><textarea rows=1 name="${t.label}_notes">${(r.notes||"").replace(/</g,"&lt;")}</textarea></fieldset>`;
  c.addEventListener("change",()=>{const x=R[t.label]=R[t.label]||{};
   for(const el of c.querySelectorAll("input[type=radio]:checked")){x[el.name.slice(t.label.length+1)]=el.value}
   x.notes=c.querySelector("textarea").value;save();status()});
  c.addEventListener("input",e=>{if(e.target.tagName=="TEXTAREA"){(R[t.label]=R[t.label]||{}).notes=e.target.value;save()}});
  box.appendChild(c)});status()}
function complete(x){return x&&x.emotion&&x.naturalness&&x.intelligibility&&x.speaker_consistency}
function status(){let n=0;TRIALS.forEach(t=>{const ok=complete(R[t.label]);if(ok)n++;document.getElementById("c_"+t.label).classList.toggle("done",!!ok)});
 document.getElementById("prog").textContent=n+" / "+TRIALS.length+" rated"}
function payload(){return {page_id:PAGE_ID,rater:document.getElementById("rater").value,exported_at:new Date().toISOString(),ratings:R}}
function dl(name,text,type){const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click()}
document.getElementById("dl").onclick=()=>dl("ratings_"+PAGE_ID+".json",JSON.stringify(payload(),null,2),"application/json");
document.getElementById("dlc").onclick=()=>{const rows=[["label","emotion","naturalness","intelligibility","speaker_consistency","notes"]];
 for(const [l,x] of Object.entries(R))rows.push([l,x.emotion||"",x.naturalness||"",x.intelligibility||"",x.speaker_consistency||"",'"'+(x.notes||"").replace(/"/g,'""')+'"']);
 dl("ratings_"+PAGE_ID+".csv",rows.map(r=>r.join(",")).join("\n"),"text/csv")};
render();
</script></body></html>
"""


def cmd_page(cfg, args):
    import numpy as np
    import soundfile as sf

    runs = {}
    for label in args.runs:
        d = cfg.path("eval") / "runs" / label
        if not (d / "metadata.jsonl").is_file():
            raise SystemExit(f"run '{label}' has no metadata at {d}")
        runs[label] = (d, load_jsonl(d / "metadata.jsonl"))
    es = read_json(eval_set_path(cfg))
    rng = random.Random(args.seed if args.seed is not None else cfg["eval"]["seed"])
    page_id = args.name
    leaked = [r for r in list(runs) + ["baseline", "finetuned", "reference", "sad", "default"] if r.lower() in page_id.lower()]
    if leaked:
        raise SystemExit(f"page name {page_id!r} contains {leaked}, which would unblind the listener; choose a neutral --name")
    page_dir = cfg.path("eval") / "listening" / page_id
    key_path = cfg.path("eval") / "listening" / f"{page_id}.answer_key.json"   # outside page_dir on purpose
    remove_tree(page_dir)
    (page_dir / "clips").mkdir(parents=True)

    used = set()

    def new_label():
        while True:
            lab = "c" + "".join(rng.choices(string.ascii_lowercase + string.digits, k=7))
            if lab not in used:
                used.add(lab)
                return lab

    entries = []  # (answer-key record, source wav path or (array, sr))
    for label, (d, recs) in runs.items():
        for r in recs:
            if args.seeds and r["seed"] not in args.seeds:
                continue
            entries.append(({"run": label, "style": r["style"], "text": r["text"], "seed": r["seed"],
                             "item_id": r["item_id"], "source": r["source"], "description": r["description"],
                             "checkpoint": r["checkpoint"]["path"], "clip_id": r["clip_id"]}, d / r["wav"]))
    n_ref = 0
    if not args.no_reference:   # real held-out recordings of the same sentences, as hidden controls
        test = {ex["id"]: ex for ex in load_jsonl(cfg.path("splits") / "test.jsonl")}
        for it in es["items"]:
            for style, ex_id in it["reference_examples"].items():
                ex = test[ex_id]
                entries.append(({"run": REFERENCE_RUN, "style": style, "text": ex["text"], "seed": None,
                                 "item_id": it["item_id"], "source": "test", "description": ex["description"],
                                 "checkpoint": "original Expresso recording", "clip_id": ex_id},
                                cfg.path("processed") / ex["audio_path"]))
                n_ref += 1
    rng.shuffle(entries)
    trials, key, clipped = [], {}, 0
    for rec, src in entries:
        lab = new_label()
        audio, sr = sf.read(str(src), dtype="float32")
        if np.abs(audio).max() > 1.0:
            clipped += 1
        sf.write(page_dir / "clips" / f"{lab}.wav", np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")  # no normalization
        trials.append({"label": lab, "src": f"clips/{lab}.wav"})
        key[lab] = rec

    # reference voice: original recordings (processed, 44.1 kHz) of the speaker, one per style, different sentences
    train = load_jsonl(cfg.path("splits") / "train.jsonl")
    refs_html = []
    for style in es["styles"]:
        cand = sorted((e for e in train if e["style"] == style and 3.0 <= e["duration_s"] <= 7.0), key=lambda e: e["id"])
        pick = random.Random(cfg["eval"]["seed"]).choice(cand)
        shutil.copy(cfg.path("processed") / pick["audio_path"], page_dir / "clips" / f"reference_{style}.wav")
        refs_html.append(f'<p>Reference recording {len(refs_html) + 1}<br><audio controls preload="none" src="clips/reference_{style}.wav"></audio></p>')
    page = (PAGE.replace("__PAGE_ID__", page_id).replace("__TRIALS__", json.dumps(trials))
            .replace("__REFS__", "".join(refs_html)))
    (page_dir / "index.html").write_text(page)
    write_json(key_path, {"page_id": page_id, "created_at": utc_now(), "n_clips": len(trials), "runs": args.runs,
                          "reference_controls": n_ref, "labels": key})
    log(f"page: {page_dir / 'index.html'}  ({len(trials)} clips, {n_ref} real-recording controls, "
        f"{clipped} clips exceeded full scale and were clipped to 16-bit)")
    log(f"answer key (do not open before rating): {key_path}")


# ---------------------------------------------------------------- score
def cmd_score(cfg, args):
    key = read_json(args.key)["labels"]
    rows = defaultdict(list)
    n_files = 0
    for rf in args.ratings:
        data = read_json(rf)
        n_files += 1
        for lab, x in data["ratings"].items():
            if lab not in key or not all(k in x for k in ("emotion", "naturalness", "intelligibility", "speaker_consistency")):
                continue
            k = key[lab]
            rows[(k["run"], k["style"])].append((k["style"], x))
    if not rows:
        raise SystemExit("no complete ratings matched the answer key")

    def mean(v):
        return sum(v) / len(v)

    out = [f"Ratings from {n_files} file(s). Emotion accuracy = rater's choice equals the intended style.\n",
           "| run | intended style | n | emotion correct | naturalness | intelligibility | speaker consistency |",
           "|---|---|---|---|---|---|---|"]
    for (run, style) in sorted(rows):
        xs = rows[(run, style)]
        heard = "neutral" if style == "default" else style  # the page's option for the default style is "neutral"
        acc = mean([1.0 if x["emotion"] == heard else 0.0 for _, x in xs])
        out.append(f"| {run} | {style} | {len(xs)} | {acc:.0%} | {mean([int(x['naturalness']) for _, x in xs]):.2f} | "
                   f"{mean([int(x['intelligibility']) for _, x in xs]):.2f} | {mean([int(x['speaker_consistency']) for _, x in xs]):.2f} |")
    out.append("\nLikert scales are 1-5 (higher is better). Means over clips and raters; with few clips treat differences as indicative only.")
    out.append("Baseline caveat: the unchanged model has never heard this Expresso speaker, so its 'speaker consistency' "
               "against the reference voice measures how close a generic voice happens to be, not preserved identity. "
               "Adaptation to this speaker is the point of the fine-tune; it does not preserve any other speaker's identity.")
    text = "\n".join(out)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


# ---------------------------------------------------------------- measure
def cmd_measure(cfg, args):
    import numpy as np
    from expresso_common import analyze_audio, strip_emphasis_markers
    from inventory_expresso import pitch_profile

    es = read_json(eval_set_path(cfg))
    rows = defaultdict(list)       # (run, style) -> list of per-clip measurements
    pairs = defaultdict(dict)      # (run, item, seed) -> {style: duration}

    def measure(path, text):
        m = analyze_audio(path)
        if m["error"] or "active_fraction" not in m:
            return None
        m.update(pitch_profile(str(path), text))
        m["chars_per_active_s"] = len(strip_emphasis_markers(text)) / max(m["duration_s"] * m["active_fraction"], 0.1)
        return m

    for label in args.runs:
        d = cfg.path("eval") / "runs" / label
        for r in load_jsonl(d / "metadata.jsonl"):
            m = measure(d / r["wav"], r["text"])
            if m is None:
                rows[(label, r["style"])].append(None)
                continue
            m["peak_gen"] = r["peak"]
            rows[(label, r["style"])].append(m)
            pairs[(label, r["item_id"], r["seed"])][r["style"]] = m["duration_s"]
    test = {ex["id"]: ex for ex in load_jsonl(cfg.path("splits") / "test.jsonl")}
    for it in es["items"]:
        for style, ex_id in it["reference_examples"].items():
            ex = test[ex_id]
            m = measure(cfg.path("processed") / ex["audio_path"], ex["text"])
            if m:
                rows[("real recordings", style)].append(m)
                pairs[("real recordings", it["item_id"], 0)][style] = m["duration_s"]

    def med(xs):
        xs = [x for x in xs if x is not None]
        return f"{statistics.median(xs):.2f}" if xs else "-"

    out = ["| run | style | clips | unusable | duration s | chars / active s | speech level dBFS | F0 median Hz | F0 range st | clipped samples |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for (run, style) in sorted(rows, key=lambda k: (k[0] == "real recordings", k)):
        ms = [m for m in rows[(run, style)] if m]
        bad = sum(1 for m in rows[(run, style)] if m is None)
        out.append(f"| {run} | {style} | {len(ms)} | {bad} | {med([m['duration_s'] for m in ms])} | {med([m['chars_per_active_s'] for m in ms])} | "
                   f"{med([m['speech_level_dbfs'] for m in ms])} | {med([m.get('f0_median_hz') for m in ms])} | {med([m.get('f0_range_st') for m in ms])} | "
                   f"{sum(1 for m in ms if m['clipping_fraction'] > 0)} |")
    out += ["", "Same text and seed, sad duration divided by default duration (>1 means the sad version is slower):", "",
            "| run | pairs | median ratio | mean ratio |", "|---|---|---|---|"]
    for run in sorted({k[0] for k in pairs}, key=lambda r: (r == "real recordings", r)):
        ratios = [v["sad"] / v["default"] for k, v in pairs.items() if k[0] == run and "sad" in v and "default" in v and v["default"] > 0]
        if ratios:
            out.append(f"| {run} | {len(ratios)} | {statistics.median(ratios):.2f} | {sum(ratios) / len(ratios):.2f} |")
    text = "\n".join(out)
    note = ("\n\nThese are measurements of the audio, not ratings. A model whose styles differ in pace, level and pitch the way the real ones do "
            "has learned a measurable contrast; whether it sounds sad, natural or like the same person is for listeners to say.")
    print(text + note)
    if args.out:
        Path(args.out).write_text(text + note + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", help="path to the experiment config")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="freeze the evaluation set")
    s.add_argument("--force", action="store_true")
    g = sub.add_parser("generate", help="synthesize the evaluation set with one model")
    g.add_argument("--label", required=True, help="run name, e.g. baseline or finetuned")
    g.add_argument("--model", help="checkpoint directory (default: the unchanged base model)")
    g.add_argument("--limit", type=int, help="only the first N clips (smoke test)")
    g.add_argument("--no-hash", action="store_true", help="skip hashing the weights (metadata then has no weights_sha256)")
    p = sub.add_parser("page", help="build the blinded listening page")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--name", default="listening")
    p.add_argument("--seed", type=int)
    p.add_argument("--no-reference", action="store_true", help="omit the real-recording controls")
    p.add_argument("--seeds", type=int, nargs="+", help="only model clips generated with these seeds (a shorter page)")
    sc = sub.add_parser("score", help="aggregate exported ratings with the answer key")
    sc.add_argument("--key", required=True)
    sc.add_argument("--ratings", nargs="+", required=True)
    sc.add_argument("--out")
    m = sub.add_parser("measure", help="descriptive acoustics per run and style (not ratings)")
    m.add_argument("--runs", nargs="+", required=True)
    m.add_argument("--out")
    args = ap.parse_args()
    cfg = Config(args.config)
    {"select": cmd_select, "generate": cmd_generate, "page": cmd_page, "score": cmd_score, "measure": cmd_measure}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
