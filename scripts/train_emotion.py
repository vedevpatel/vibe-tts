"""Fine-tune Parler-TTS Mini v1 on one Expresso speaker (default + sad). Thin wrapper around the
pinned upstream recipe, third_party/parler-tts/training/run_parler_tts_training.py.

Subcommands (all read configs/expresso_emotion.toml; run from any directory):

  build-dataset  Turn the frozen train/validation manifests into the two local datasets the upstream
                 script expects (audio+text, and descriptions). Test data is never touched.
  train          build-dataset if needed, write the full upstream JSON config into the run directory,
                 run the upstream script, keep one checkpoint per evaluation, then export the best.
  export         Pick the checkpoint with the lowest VALIDATION loss and save it as a normal
                 Parler-TTS model directory with a license notice.

What this wrapper adds to the upstream recipe, and nothing else:
  * a data fingerprint, so a run directory can never silently train on stale precomputed audio codes
    (upstream reuses a non-empty save_to_disk directory without checking);
  * checkpoint slimming: upstream's per-checkpoint optimizer state is ~4 GB, so older checkpoints
    keep only their model weights (the newest keeps everything, so a crash can resume);
  * telemetry off (HF_HUB_DISABLE_TELEMETRY, WANDB_MODE=disabled), tensorboard logging;
  * validation-loss checkpoint selection (the frozen test split is never used for selection).

Runs in the isolated training environment (scripts/setup_train.sh -> .venv-train) or on a Colab GPU
(scripts/run_colab_train.sh). A real run needs a CUDA GPU; see EXPRESSO.md.

Output of this fine-tuning inherits the Expresso license, CC BY-NC 4.0: noncommercial use only.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from _common import ROOT, model_dir  # first: loads .env
from expresso_common import Config, description_for, log, read_json, remove_tree, sha256_file, utc_now, write_json

UPSTREAM = ROOT / "third_party" / "parler-tts"
UPSTREAM_SCRIPT = UPSTREAM / "training" / "run_parler_tts_training.py"
EVAL_RE = re.compile(r"Eval results for step \((\d+) / (\d+) \| Eval Loss: (?:tensor\()?([0-9.eE+-]+)")
CKPT_RE = re.compile(r"^checkpoint-(\d+)-epoch-(\d+)$")

NOTICE = """\
EXPERIMENTAL, NONCOMMERCIAL CHECKPOINT

This model is a fine-tune of parler-tts/parler-tts-mini-v1 (Apache-2.0, Hugging Face) on read speech
of one speaker ({speaker}) from the Expresso dataset (styles: {styles}).

Expresso is distributed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/).
Anything derived from it, including these weights, may only be used for noncommercial purposes, with
attribution:

  {attribution}

Not published. Do not upload, redistribute or use commercially. Synthetic speech from this model
imitates a real speaker's voice and delivery: do not present it as real, and do not use it to
impersonate anyone.
"""


def run_dir(cfg, args):
    return cfg.path("runs") / args.run


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def data_fingerprint(cfg):
    h = hashlib.sha256()
    for name in ("train", "validation"):
        h.update((cfg.path("splits") / f"{name}.jsonl").read_bytes())
    h.update(json.dumps(cfg["description"], sort_keys=True).encode())
    return h.hexdigest()


# ---------------------------------------------------------------- build-dataset
def cmd_build_dataset(cfg, args):
    rd = run_dir(cfg, args)
    ds_dir = rd / "dataset"
    fp = data_fingerprint(cfg)
    stamp = ds_dir / "FINGERPRINT"
    if stamp.is_file():
        if stamp.read_text().strip() == fp:
            log(f"dataset up to date ({ds_dir})")
            return ds_dir
        raise SystemExit(f"{rd} was built from different data (manifests or descriptions changed). Use a new --run name.")
    remove_tree(ds_dir)
    for split in ("train", "validation"):
        rows = load_jsonl(cfg.path("splits") / f"{split}.jsonl")
        a_dir, m_dir = ds_dir / "audio" / split, ds_dir / "meta" / split
        a_dir.mkdir(parents=True)
        m_dir.mkdir(parents=True)
        with open(a_dir / "metadata.jsonl", "w") as fa, open(m_dir / "metadata.jsonl", "w") as fm:
            for ex in rows:
                name = f"{ex['id']}.wav"
                shutil.copyfile(cfg.path("processed") / ex["audio_path"], a_dir / name)
                fa.write(json.dumps({"file_name": name, "id": ex["id"], "text": ex["text"]}) + "\n")
                # description: rebuilt from the config so the training text and evaluation text share one source
                fm.write(json.dumps({"id": ex["id"], "text_description": description_for(cfg, ex["speaker"], ex["style"])}) + "\n")
        log(f"{split}: {len(rows)} examples")
    stamp.write_text(fp + "\n")
    return ds_dir


# ---------------------------------------------------------------- train
def steps_per_epoch(cfg, rows):
    """Mirror upstream: examples kept by its duration filter, divided by the effective batch."""
    t = cfg["training"]
    n = sum(1 for r in rows if t["min_duration_in_seconds"] < r["duration_s"] < t["max_duration_in_seconds"])
    return max(1, n // (t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]))


def upstream_config(cfg, args, rd, ds_dir):
    t = cfg["training"]
    base = model_dir(t["base_model"])
    spe = steps_per_epoch(cfg, load_jsonl(cfg.path("splits") / "train.jsonl"))
    eval_steps = spe * t["eval_every_epochs"]
    conf = {
        "model_name_or_path": str(base),
        "feature_extractor_name": str(base),        # preprocessor_config.json (44.1 kHz) ships with the model
        "description_tokenizer_name": str(base),
        "prompt_tokenizer_name": str(base),
        "report_to": [t["report_to"]],
        # Always true, as in the official commands: upstream creates tensorboard files in output_dir before its own
        # emptiness check, so a fresh run would otherwise always fail. It deletes nothing. Resuming is explicit (--resume).
        "overwrite_output_dir": True,
        "train_dataset_name": str(ds_dir / "audio"),
        "train_metadata_dataset_name": str(ds_dir / "meta"),
        "train_dataset_config_name": "default",
        "train_split_name": "train",
        "eval_dataset_name": str(ds_dir / "audio"),
        "eval_metadata_dataset_name": str(ds_dir / "meta"),
        "eval_dataset_config_name": "default",
        "eval_split_name": "validation",
        "target_audio_column_name": "audio",
        "description_column_name": "text_description",
        "prompt_column_name": "text",
        "id_column_name": "id",
        "max_duration_in_seconds": t["max_duration_in_seconds"],
        "min_duration_in_seconds": t["min_duration_in_seconds"],
        "max_text_length": t["max_text_length"],
        "preprocessing_num_workers": t["preprocessing_num_workers"],
        "do_train": True,
        "num_train_epochs": t["num_train_epochs"],
        "gradient_accumulation_steps": t["gradient_accumulation_steps"],
        "gradient_checkpointing": t["gradient_checkpointing"],
        "per_device_train_batch_size": t["per_device_train_batch_size"],
        "learning_rate": t["learning_rate"],
        "adam_beta1": t["adam_beta1"],
        "adam_beta2": t["adam_beta2"],
        "weight_decay": t["weight_decay"],
        "lr_scheduler_type": t["lr_scheduler_type"],
        "warmup_steps": t["warmup_steps"],
        "logging_steps": 2,
        "freeze_text_encoder": t["freeze_text_encoder"],
        "audio_encoder_per_device_batch_size": t["audio_encoder_per_device_batch_size"],
        "dtype": t["dtype"],
        "seed": t["seed"],
        "output_dir": str(rd / "output"),
        "temporary_save_to_disk": str(rd / "audio_code_tmp"),
        "save_to_disk": str(rd / "precomputed"),
        "dataloader_num_workers": t["dataloader_num_workers"],
        "do_eval": True,
        "predict_with_generate": t["predict_with_generate"],
        "include_inputs_for_metrics": True,
        "per_device_eval_batch_size": t["per_device_eval_batch_size"],
        "group_by_length": True,
        "attn_implementation": t["attn_implementation"],
        "eval_steps": eval_steps,
        "save_steps": eval_steps,                    # a saved checkpoint at every evaluation
        "save_total_limit": t["save_total_limit"],
        "cache_dir": str(rd / "hf_cache"),
    }
    if args.resume:
        full = [p for p in (rd / "output").glob("checkpoint-*-epoch-*") if (p / "optimizer.bin").is_file()]
        if not full:
            raise SystemExit(f"nothing to resume: no checkpoint with optimizer state under {rd / 'output'}")
        conf["resume_from_checkpoint"] = str(max(full, key=lambda p: int(CKPT_RE.match(p.name).group(1))))
    if args.max_steps:
        conf["max_steps"] = args.max_steps
        conf["eval_steps"] = conf["save_steps"] = max(1, args.max_steps)
    if args.preprocessing_only:
        conf["preprocessing_only"] = True
    return conf, spe


def slim_checkpoints(out_dir, final=False):
    """Drop optimizer/scheduler/rng state from every checkpoint that has a successor (all of them when
    `final`). Model weights (pytorch_model.bin) are what checkpoint selection and export need."""
    ckpts = sorted((p for p in out_dir.iterdir() if p.is_dir() and CKPT_RE.match(p.name)),
                   key=lambda p: int(CKPT_RE.match(p.name).group(1)))
    for p in ckpts if final else ckpts[:-1]:
        for f in p.iterdir():
            if f.is_file() and f.name != "pytorch_model.bin" and f.stat().st_size > 50 * 2**20:
                f.unlink()


def cmd_train(cfg, args):
    rd = run_dir(cfg, args)
    cfg.require_data_root()
    for need in ("train", "validation"):
        if not (cfg.path("splits") / f"{need}.jsonl").is_file():
            raise SystemExit(f"{need} manifest missing: run scripts/prepare_expresso.py first")
    if not UPSTREAM_SCRIPT.is_file():
        raise SystemExit(f"{UPSTREAM_SCRIPT} not found: run scripts/setup.sh")
    pinned = re.search(r"^PARLER_REV=(\w+)", (ROOT / "scripts" / "config.sh").read_text(), re.M).group(1)
    rev = subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if rev != pinned:
        raise SystemExit(f"third_party/parler-tts is at {rev or 'unknown'}, pinned revision is {pinned}")
    if args.fresh:
        remove_tree(rd)
    rd.mkdir(parents=True, exist_ok=True)
    ds_dir = cmd_build_dataset(cfg, args)

    conf, spe = upstream_config(cfg, args, rd, ds_dir)
    conf_path = rd / "train_config.json"
    write_json(conf_path, conf)
    cmd = [sys.executable, str(UPSTREAM_SCRIPT), str(conf_path)]
    env = dict(os.environ, HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled", USE_TF="0", USE_FLAX="0",
               TOKENIZERS_PARALLELISM="false", PYTHONPATH=f"{UPSTREAM}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    log(f"upstream {pinned[:12]}, {spe} steps/epoch, eval+save every {conf['eval_steps']} steps")
    log("command: " + " ".join(cmd))
    if args.dry_run:
        log(f"dry run: config written to {conf_path}")
        return
    env_info = {"started_at": utc_now(), "upstream_revision": rev, "python": sys.version.split()[0]}
    try:
        import accelerate, datasets, torch, transformers
        env_info.update(torch=torch.__version__, transformers=transformers.__version__, accelerate=accelerate.__version__,
                        datasets=datasets.__version__, cuda=torch.cuda.is_available(),
                        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError as e:
        raise SystemExit(f"training dependencies missing ({e}); use .venv-train (scripts/setup_train.sh) or Colab")
    if not env_info["cuda"] and not args.allow_cpu:
        raise SystemExit("no CUDA GPU: a real fine-tune of the 880M model needs one (see scripts/run_colab_train.sh). "
                         "--allow-cpu is for --preprocessing-only / tiny smoke tests.")
    env_info["config_sha256"] = sha256_file(conf_path)
    write_json(rd / "environment.json", env_info)

    out_dir = rd / "output"
    if any(out_dir.glob("checkpoint-*")) and not args.resume:
        raise SystemExit(f"{out_dir} already holds checkpoints: pass --resume to continue that run or --fresh to start over")
    out_dir.mkdir(exist_ok=True)
    stop = threading.Event()

    def watcher():
        while not stop.wait(20):
            try:
                slim_checkpoints(out_dir)
            except OSError:
                pass

    th = threading.Thread(target=watcher, daemon=True)
    th.start()
    t0 = time.time()
    with open(rd / "train.log", "a") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            if EVAL_RE.search(line) or "Step... (" in line or "Error" in line or "Traceback" in line:
                log(line.rstrip())
        code = proc.wait()
    stop.set()
    th.join()
    log(f"upstream exited with {code} after {(time.time() - t0) / 60:.1f} min")
    if code != 0:
        raise SystemExit(f"training failed (see {rd / 'train.log'})")
    slim_checkpoints(out_dir, final=True)
    if args.preprocessing_only or args.max_steps:
        log("preprocessing-only / max-steps smoke run: not exporting")
        return
    cmd_export(cfg, argparse.Namespace(run=args.run, step=None, dest=None))   # the best validation loss decides


# ---------------------------------------------------------------- export
def validation_losses(rd):
    """{step: validation loss} parsed from the upstream log."""
    losses = {}
    for line in (rd / "train.log").read_text().splitlines():
        m = EVAL_RE.search(line)
        if m:
            losses[int(m.group(1))] = float(m.group(3))
    return losses


def cmd_export(cfg, args):
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration

    rd = run_dir(cfg, args)
    out_dir = rd / "output"
    losses = validation_losses(rd)
    ckpts = {int(CKPT_RE.match(p.name).group(1)): p for p in out_dir.iterdir() if p.is_dir() and CKPT_RE.match(p.name)}
    scored = {s: l for s, l in losses.items() if s in ckpts and (ckpts[s] / "pytorch_model.bin").is_file()}
    if not scored:
        raise SystemExit(f"no checkpoint with a validation loss in {out_dir}; was the run completed?")
    step = args.step or min(scored, key=scored.get)
    if step not in ckpts:
        raise SystemExit(f"no checkpoint for step {step}; have {sorted(ckpts)}")
    log("validation loss by step: " + ", ".join(f"{s}: {scored[s]:.4f}" for s in sorted(scored)))
    log(f"selected step {step} (lowest validation loss {scored.get(step, float('nan')):.4f})" if not args.step
        else f"exporting requested step {step}")

    base = model_dir(cfg["training"]["base_model"])
    model = ParlerTTSForConditionalGeneration.from_pretrained(base, torch_dtype=torch.float32)
    state = torch.load(ckpts[step] / "pytorch_model.bin", map_location="cpu", weights_only=True)  # tensors only: no pickled code
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith(("audio_encoder.", "text_encoder.")) for k in missing):
        raise SystemExit(f"checkpoint does not match the base architecture: missing={missing[:3]} unexpected={unexpected[:3]}")
    dest = Path(args.dest) if args.dest else rd / "export"
    remove_tree(dest)
    model.save_pretrained(dest, safe_serialization=True)
    for name in ("preprocessor_config.json", "special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json"):
        if (base / name).is_file():
            shutil.copyfile(base / name, dest / name)
    speaker, styles = cfg.speaker(), cfg["speaker"]["styles"]
    src = read_json(cfg.path("raw") / "SOURCES.json") if (cfg.path("raw") / "SOURCES.json").is_file() else {}
    attribution = cfg["source"]["attribution"]
    (dest / "NONCOMMERCIAL_NOTICE.md").write_text(NOTICE.format(speaker=speaker, styles=", ".join(styles), attribution=attribution))
    info = {
        "exported_at": utc_now(), "run": args.run, "selected_step": step, "selection": "lowest validation loss" if not args.step else "explicit --step",
        "validation_loss_by_step": {str(s): l for s, l in sorted(scored.items())},
        "base_model": read_json(base / "REVISION.json") if (base / "REVISION.json").is_file() else str(base),
        "speaker": speaker, "styles": styles, "license": "CC BY-NC 4.0 (derived from Expresso); noncommercial, not published",
        "dataset_source": src.get("source_url"), "dataset_archive_md5": src.get("archive", {}).get("md5"),
        "train_config_sha256": sha256_file(rd / "train_config.json"),
        "weights_sha256": sha256_file(dest / "model.safetensors"),
    }
    write_json(dest / "export_info.json", info)
    log(f"exported step {step} -> {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", help="path to the experiment config")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build-dataset", "train", "export"):
        s = sub.add_parser(name)
        s.add_argument("--run", default="emotion-v1", help="run name (a directory under <data root>/runs)")
        if name == "train":
            s.add_argument("--fresh", action="store_true", help="delete the run directory first and start over")
            s.add_argument("--resume", action="store_true", help="continue from the newest checkpoint that still has optimizer state")
            s.add_argument("--dry-run", action="store_true", help="write the upstream config and print the command only")
            s.add_argument("--max-steps", type=int, help="smoke test: stop after N optimizer steps (no export)")
            s.add_argument("--preprocessing-only", action="store_true", help="stop after encoding audio (no training)")
            s.add_argument("--allow-cpu", action="store_true", help="permit running without a CUDA GPU (smoke tests only)")
        if name == "export":
            s.add_argument("--step", type=int, help="export this step instead of the best validation loss")
            s.add_argument("--dest", help="output directory (default <run>/export)")
    args = ap.parse_args()
    cfg = Config(args.config)
    {"build-dataset": cmd_build_dataset, "train": cmd_train, "export": cmd_export}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
