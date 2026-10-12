# Expresso emotion fine-tuning (experimental, noncommercial)

**Question.** If `parler-tts/parler-tts-mini-v1` is fine-tuned on one Expresso speaker's *default* and *sad*
read speech, does it produce recognizably different delivery for the two styles on text it has never seen,
while staying natural, intelligible and sounding like the same speaker?

**Status.** The pipeline is built and verified, a checkpoint was trained and evaluated, and one blinded listening set
(set-b) has been rated by one person. **The result is negative on the main question:** the fine-tune moved the voice
toward the target speaker, but the rater could not tell its default from its sad delivery. Set-a is not yet rated. See
[Results](#results).

## License: CC BY-NC 4.0, noncommercial

The Expresso dataset is distributed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Everything derived from it, including prepared audio and any checkpoint trained on it, is **noncommercial
only** and needs attribution:

> Nguyen et al., *EXPRESSO: A Benchmark and Analysis of Discrete Expressive Speech Resynthesis*,
> INTERSPEECH 2023 (<https://arxiv.org/abs/2308.05725>). Dataset by Meta AI (Facebook Research).

The base model, Parler-TTS Mini v1, is Apache-2.0. The fine-tuned weights are not: they carry
`NONCOMMERCIAL_NOTICE.md` and are labelled experimental. **Nothing is published**: no dataset, no
checkpoint, no push. The synthetic voice imitates a real speaker; do not present it as real or use it to
impersonate anyone.

## Provenance

| What | Where | Pinned to |
|---|---|---|
| Dataset | `https://dl.fbaipublicfiles.com/textless_nlp/expresso/data/expresso.tar` (the URL in the [official dataset README](https://github.com/facebookresearch/textlesslib/tree/main/examples/expresso/dataset)) | 38,441,031,680 bytes, md5 `6bf580a4cd4392ae2473626147b5307c` (published in that README, verified), sha256 `9c3592e96f7c688c23f5d593425e48ce963df544872509047200989ea7ebf71b` (ours) |
| Layout and transcript format | read from the real archive, cross-checked with the official processing code | `read/{speaker}/{style}/{base,longform}/{speaker}_{style[_sub]}_{id}.wav`; `read_transcriptions.txt` is `{stem}<TAB>{text}` |
| Training recipe | [`training/README.md`](https://github.com/huggingface/parler-tts/blob/main/training/README.md) and the [single-speaker notebook](https://github.com/ylacombe/scripts_and_notebooks/blob/main/Finetuning_Parler_TTS_v1_on_a_single_speaker_dataset.ipynb) | upstream script at `d108732cd57788ec86bc857d99a6cabd66663d68` (unchanged, `third_party/`) |
| Base model | `parler-tts/parler-tts-mini-v1` | `0392b9451a601e528fd863bbb0598431fee810d9` (unchanged) |

The retrieval date, source URL, license, attribution and checksums are also recorded by the downloader in
`data/expresso/raw/SOURCES.json`. No Chinese-developed or -owned model, service or component is used
(Parler-TTS, DAC, flan-T5, transformers, torch, accelerate, datasets, tensorboard).

## Workflow

Everything reads [`configs/expresso_emotion.toml`](configs/expresso_emotion.toml) and runs from any directory.
`data/expresso` is a gitignored symlink to external storage on this Mac (the archive is 38 GB).

```
scripts/setup.sh                                       # existing: inference env + pinned parler-tts + model
scripts/setup_train.sh                                 # isolated training env (.venv-train)

.venv/bin/python scripts/download_expresso.py          # download (resumable), md5, safe extract, header validation
.venv/bin/python scripts/inventory_expresso.py         # inventory, speaker choice, acoustic checks, listening pack
.venv/bin/python scripts/prepare_expresso.py           # resample, gate, descriptions, grouped splits, verification
.venv/bin/python scripts/evaluate_emotion.py select    # freeze the evaluation set
scripts/run_colab_train.sh                             # Colab GPU: baseline generation, fine-tune, fine-tuned generation
#   (--smoke checks the whole pipeline in a few minutes first; --keep-vm-on-failure keeps the VM if a late step fails)
.venv/bin/python scripts/evaluate_emotion.py measure --runs baseline finetuned     # descriptive acoustics, not ratings
.venv/bin/python scripts/evaluate_emotion.py page --runs baseline finetuned --name set-a
#   listen in the browser, export ratings, then:
.venv/bin/python scripts/evaluate_emotion.py score --key <key.json> --ratings <ratings.json>
```

`scripts/prepare_expresso.py --verify-only` re-checks the saved splits at any time.

## Decisions, and the evidence for them

All numbers are from the real data (`data/expresso/inventory/inventory.md`, `processed/description_checks.json`,
`splits/split_report.md`).

### Dataset and inventory
- 11,615 read-speech WAVs, all 48 kHz mono 24-bit; 11,615 transcripts, one per file. Every file was decoded in full:
  0 corrupt, 0 audio without a transcript, 0 transcripts without audio, 0 files listed in the official split lists but
  missing, 0 byte-identical duplicates. 11 files exist on disk but not in the official lists: all `singing` (excluded).
- Improvised dialogue (about 30 GB) is never extracted. Singing is excluded by style. Longform recordings are excluded
  (2-3 minutes each, no sentence-level transcripts, far above the 30 s training limit).
- Two sub-corpora of `default` are dropped (`exclude_substyles`): **emphasis** (200 clips per speaker; the same sentence
  read repeatedly with a different word stressed, marked `*word*` in the transcript, so it is stressed rather than neutral
  delivery and repeats text) and **essentials** (40 clips of greetings such as "Hello!", 16-20 of them under 1 s). What
  remains of `default` is the plain read corpus, the like-for-like counterpart of `sad`.

### Speaker `ex02`
One rule, applied identically to all four speakers (`scripts/inventory_expresso.py`): (1) at least 100 usable clips in
every target style; (2) every measurable claim in the descriptions must hold for that speaker's recordings; (3) of the
rest, the most usable audio in the scarcer style.

| speaker | default: usable clips / audio | sad: usable clips / audio | noise floor p95 (default / sad) | "sad is subdued" | outcome |
|---|---|---|---|---|---|
| ex02 | 375 / 17:29 | 377 / 20:42 | -57.5 / -56.6 dBFS | supported (4 of 5 indicators) | **selected** |
| ex04 | 374 / 16:54 | 377 / 18:58 | -45.8 / -50.0 dBFS | supported | eligible |
| ex03 | 371 / 15:46 | 378 / 21:36 | -48.7 / -47.2 dBFS | supported | eligible |
| ex01 | 370 / 19:01 | 377 / 19:46 | -49.4 / -51.0 dBFS | **not supported**: sad is not lower in pitch or pace | rejected |

The differences between speakers are small. My first pass leaned towards ex03, which was the rule's pick before the
sub-corpus exclusion above shrank its default audio; I followed the rule rather than my earlier preference. ex03 (the
cleanest median recording) and ex04 are reasonable alternatives: set `speaker.id` in the config and re-run.

### Descriptions and what was verified
- default: *"Expresso speaker ex02 speaks in a neutral delivery. The recording is clear with little background noise."*
- sad: *"Expresso speaker ex02 speaks in a sad, subdued delivery. The recording is clear with little background noise."*

| claim | verdict | evidence (ex02, medians) |
|---|---|---|
| clear, little background noise | supported | noise floor -60.1 dBFS (default), -62.7 dBFS (sad); 95th percentile -57.5 / -56.6 |
| sad is subdued | supported | speech level -20.7 vs -19.7 dBFS; F0 223 vs 248 Hz; F0 range 6.1 vs 11.3 semitones; pace 16.4 vs 19.6 characters per active second (share of time speaking is equal) |
| default is neutral | label only | Expresso's unmarked read style; not measurable |
| "conversational" | not verifiable, **dropped** | the clips are read-aloud sentences |

These are acoustic measurements. They do not show that the delivery *sounds* sad; that needs a listener (see the
listening pack and the listening test below).

### Preparation
- 752 of 760 candidate clips accepted. The 8 exclusions are all under 1 s (`processed/exclusions.jsonl`). No clip was
  corrupt, non-finite, empty, clipped (limit 0.1% of samples at full scale) or without a transcript.
- Target rate 44,100 Hz, read from the model's own config (`audio_encoder.sampling_rate` and `preprocessor_config.json`
  agree). Resampled with soxr (VHQ), not by rewriting a header. Checked on real clips: duration preserved to the sample,
  RMS unchanged (0.000 dB), flat frequency response. Output is 24-bit so quiet speech is not requantized.
- No denoising, silence trimming or loudness normalization: pauses and quiet speech are part of the delivery.
- Transcripts are used exactly as released, with one edit: the `*emphasis*` asterisks (e.g. "I *was* just going to say")
  are markup, not speech, and are dropped from `text`; the original line is kept as `text_original`. No ASR was used.
- A robust characters-per-second check per style would have flagged transcripts attached to the wrong audio; it flagged none.

### Splits
Grouped (union-find) by normalized transcript and by source-audio hash before assignment, so the same words in the other
style or a repeated recording never cross splits. Seed 1234, about 80/10/10, balanced by style:

| split | default | sad | distinct transcripts | audio |
|---|---|---|---|---|
| train | 299 | 301 | 333 | 30.7 min |
| validation | 38 | 38 | 42 | 4.0 min |
| test | 38 | 38 | 42 | 3.5 min |

Most sentences exist in both styles, so validation and test mostly compare identical words in two deliveries.
`prepare_expresso.py` verifies programmatically that no transcript, source file, audio hash or prepared path appears in two
splits, that both styles are in every split, that shares are within tolerance and that the assignment is deterministic.
The test split is frozen (`splits/FROZEN.json`); validation loss selects the checkpoint, the test split is never used to
choose anything.

## Training (`scripts/train_emotion.py`, `scripts/run_colab_train.sh`)

A thin wrapper around the pinned upstream script (`third_party/parler-tts/training/run_parler_tts_training.py`,
unmodified). The wrapper builds the two local datasets the script expects (audio + text, descriptions) from the frozen
train and validation manifests, writes the *complete* upstream JSON config into the run directory (so a run is reproducible
from `train_config.json` + `environment.json`), runs the script, and exports the best checkpoint. It adds only:

- a **data fingerprint**: upstream reuses a non-empty `save_to_disk` directory without checking, so a stale run directory
  could silently train on old audio codes; a changed manifest or description now forces a new run name;
- **checkpoint slimming**: upstream's per-checkpoint optimizer state is about 4 GB. Older checkpoints keep only model
  weights (the newest keeps everything so a crash can `--resume`), so one checkpoint per epoch fits on a Colab disk;
- **telemetry off** (`HF_HUB_DISABLE_TELEMETRY`, `WANDB_MODE=disabled`; upstream sends example telemetry otherwise) and
  tensorboard logging (`wandb` is installed only because upstream imports it);
- **validation-loss checkpoint selection**, then an export with a `NONCOMMERCIAL_NOTICE.md` and `export_info.json` (selected
  step, the validation losses of every step, base-model revision, dataset md5, config and weights hashes).

Two upstream quirks are handled rather than patched: tensorboard creates files in `output_dir` before upstream's own
"directory must be empty" check, so `overwrite_output_dir` is always true (the official commands do the same; it deletes
nothing) and resuming is an explicit `--resume`; and upstream requires a metadata dataset, so descriptions go in a second
local dataset joined on `id`.

### Settings, and where they differ from the official single-speaker notebook

| setting | notebook (Jenny, ~5 h) | this run (ex02, 30.7 min train) | why |
|---|---|---|---|
| init | `parler-tts-mini-v1` | same checkpoint, local pinned copy | as specified |
| learning rate / betas / weight decay | 1e-4, (0.9, 0.99), 0.01 | same | recipe |
| scheduler | constant with warmup, 50 steps | constant with warmup, 20 steps | about 370 total steps instead of about 280 |
| text encoder | frozen | frozen | recipe |
| effective batch | 2 x 18 = 36 | 4 x 4 = 16 | about 10x less data: more optimizer steps per epoch |
| epochs | 2 | 10 (37 steps each) | the notebook's step count is matched, and the validation curve shows where it stops helping |
| precision | fp16 (T4) | bf16 (L4) | Ada GPUs support bf16; T5 does not do fp16 |
| gradient checkpointing, attention | on, sdpa | on, sdpa | recipe |
| duration filter | 2-20 s | 1-30 s | our clips are 1-13 s; 1 s keeps short utterances |
| generation metrics during training | on | off | WER/CLAP need extra model downloads; generation is evaluated afterwards, identically for both models |
| logging | wandb | tensorboard | no account or credentials needed |

Validation loss is computed every epoch with a checkpoint saved at the same step; the test split is not used for any of this.
`preprocessing_num_workers = 1` because `datasets` 3.6's multiprocessing map hangs on Colab's Python 3.13.

## Evaluation design (`scripts/evaluate_emotion.py`)

- **Frozen evaluation set** (`splits/eval_set.json`, created once by `select`): 8 test-split sentences chosen with a fixed
  seed from sentences that exist in both styles (so each has real default and sad recordings), plus 6 new, emotionally
  ambiguous sentences that `select` checks are absent from every transcript in the release.
- **Identical conditions before and after:** the same sentences, the same descriptions (built from the config), the same
  generation seeds (11, 22), the same generation settings (`do_sample`, temperature 1.0, `max_length` 2580,
  `min_new_tokens` 10, float32), the same tokenizer, and the same waveform handling (raw model output saved untouched as
  32-bit float WAV). Both runs execute on the same Colab GPU in one job. 14 sentences x 2 styles x 2 seeds = 56 clips per model.
- **Metadata per clip** (`metadata.jsonl`): checkpoint (path, revision record, weights sha256), description, text, seed,
  duration, generation time, device, dtype, generation kwargs, library versions.
- **Blinded listening page** (`page`): opaque shuffled clip labels; real held-out recordings of the same sentences are mixed
  in as hidden controls; two real reference recordings of the speaker are shown openly for the speaker-consistency question.
  The answer key is written *next to* the page directory, never inside it. For each clip the listener gives: the emotion
  they hear (neutral / sad / other / cannot tell), naturalness (1-5), intelligibility (1-5), and speaker consistency with
  the reference voice (1-5). Ratings stay in the browser and are exported as JSON/CSV; `score` joins them with the key.
- **Nothing is rated automatically.** No claim about emotion, naturalness or identity is made without listener ratings.

## Results

**One blinded listening set (set-b, 72 clips, one rater) has been scored; set-a is unrated.** The listening result does not
show that the model learned recognizably different emotional delivery. It does suggest the voice moved toward the target
speaker and stayed natural and intelligible. The rating is a single person on a small set, so it is indicative only, and
the controls (below) show the rating itself is unreliable. Alongside it are the machine measurements.

### Training run (Colab L4, `runs/emotion-v1`)
Python 3.13.15, torch 2.11.0+cu130, transformers 4.46.1, accelerate 1.15.0, datasets 3.6.0, upstream `d108732cd577`,
config hash in `environment.json`. 370 optimizer steps, about 12 minutes. A first full run (which failed *after* training
at the export step because of a wrapper bug that is now fixed) logged validation losses within 0.001 of these
(4.2541 / 4.0923 / 4.0474 / 4.0287 for the first four epochs against 4.2541 / 4.0925 / 4.0475 / 4.0296 here), so the seeded
run reproduces.

| epoch | 1 | 2 | 3 | **4** | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| step | 37 | 74 | 111 | **148** | 185 | 222 | 259 | 296 | 333 | 370 |
| validation loss | 4.254 | 4.093 | 4.047 | **4.030** | 4.033 | 4.049 | 4.056 | 4.081 | 4.124 | 4.107 |

Validation loss is lowest at epoch 4 and rises slowly afterwards (the last epoch is 1.9% above the best): mild
over-fitting to 30 minutes of audio. The **exported checkpoint is step 148**, chosen by validation loss only
(`export_info.json`). The logged *training* loss shows one inflated value (about 9-11 against a typical 3) every 38 steps:
upstream prints each step's *last micro-batch* scaled by the accumulation count over the whole step's token count, so a long
last micro-batch looks inflated. The loss that is back-propagated is the proper token mean, and the validation loss is
computed separately, so the curve above is unaffected.

### Descriptive measurements of the 56 evaluation clips per model (same GPU, device, seeds, settings)
`evaluate_emotion.py measure` (medians; `outputs/expresso/measurements.md`). These are measurements, **not ratings**.

| | style | duration s | chars / active s | speech level dBFS | F0 median Hz | F0 range st |
|---|---|---|---|---|---|---|
| baseline (unchanged model) | default / sad | 2.37 / 2.30 | 22.6 / 23.3 | -10.8 / -10.7 | 174 / 175 | 7.9 / 7.0 |
| fine-tuned (step 148) | default / sad | 2.48 / 2.72 | 19.6 / 19.1 | -22.0 / -21.6 | 260 / 252 | 8.9 / 7.3 |
| real held-out recordings (n = 8 per style) | default / sad | 2.66 / 3.16 | 19.5 / 16.9 | -20.0 / -21.4 | 263 / 234 | 10.8 / 7.6 |

Same text and seed, sad duration divided by default duration: baseline 0.99 (median), fine-tuned 1.07, real recordings 1.18.

What these numbers do and do not say:
- The unchanged model produces **no measurable difference** between the two descriptions (duration ratio 0.99, same pitch and
  level), so the description alone does not make it speak differently in these respects.
- The fine-tuned model's voice moved to this speaker's range (median F0 about 252-260 Hz and level about -22 dBFS, against 174 Hz
  and -10.8 dBFS for the baseline and 263 Hz and -20 dBFS for the real default recordings). It is quieter and lower-peaked than
  the baseline by about 11 dB, as the real recordings are.
- Its sad version is slower (ratio 1.07), lower in pitch (252 vs 260 Hz) and narrower in pitch range (7.3 vs 8.9 semitones) than
  its default, **in the same direction as the real recordings but much weaker**: the real duration ratio is 1.18 and the real
  pitch drop about 29 Hz. Its level does not drop at all (-21.6 vs -22.0 dBFS; real: -21.4 vs -20.0).
- Sample sizes are small (28 clips per style per model, 8 real clips per style) and pitch tracking on 2-3 second clips is noisy.
  None of this says whether the output sounds sad, natural, intelligible, or like the same person. Only listening can.

### Listening result, set-b (one rater, 72 clips)
`scripts/evaluate_emotion.py score` output, also saved as [`results/score_set-b.md`](results/score_set-b.md) with the raw
ratings in [`results/ratings_set-b.json`](results/ratings_set-b.json). Emotion correct = the rater's choice equals the
intended style; Likert scores are 1-5.

| run | intended style | n | emotion correct | naturalness | intelligibility | speaker consistency |
|---|---|---|---|---|---|---|
| baseline | default | 13 | 62% | 2.62 | 4.38 | 1.00 |
| baseline | sad | 14 | 0% | 2.43 | 4.36 | 1.00 |
| fine-tuned | default | 13 | 15% | 3.08 | 4.69 | 2.92 |
| fine-tuned | sad | 13 | 15% | 3.31 | 5.00 | 4.31 |
| real recordings (hidden controls) | default | 8 | 0% | 3.12 | 5.00 | 4.00 |
| real recordings (hidden controls) | sad | 8 | 62% | 4.50 | 5.00 | 4.50 |

What this does and does not say:
- **Emotion was not recognized in the fine-tuned output.** 15% correct for both styles is at or below chance for a two-way
  choice. The rater did not hear the fine-tuned default and sad as different in the intended way. The machine measurements
  point the same way: the effect is in the right direction but weak (sad is 1.07x longer than default against 1.18x in the
  real recordings, and the level does not drop).
- **The rating itself is not trustworthy yet.** The hidden real recordings should be the easy case, but real default clips
  scored 0% and real sad only 62%. That points to a rater bias (for example hearing everything as sad), a labeling or polarity
  problem in the page or scorer, or both. This has not been diagnosed. Until it is, the 15% and 62% figures should not be read
  as the model's true performance.
- **The voice and quality did improve.** Speaker consistency rises from 1.00 (baseline, which never heard this speaker) to
  2.9-4.3, and naturalness and intelligibility are at or above the baseline. The baseline's 1.00 consistency is not preserved
  identity; see Limitations.
- One rater, 13-14 clips per cell, 8 per control cell: treat differences as indicative only.

### Listen
Two blinded pages are ready (open `index.html` in a browser; ratings save automatically in that browser and export as JSON):

| page | clips | contents | status |
|---|---|---|---|
| `outputs/expresso/listening/set-a/` | 128 | both models, both seeds, plus 16 hidden real-recording controls (about an hour) | not yet rated |
| `outputs/expresso/listening/set-b/` | 72 | seed 11 only (about 40 minutes) | rated by one person (above) |

Before judging anything, listen to `data/expresso/listening_pack/` (original recordings of the speaker in both styles). The
answer keys are `outputs/expresso/listening/set-{a,b}.answer_key.json`; do not open one before rating. Then:

```
.venv/bin/python scripts/evaluate_emotion.py score --key outputs/expresso/listening/set-a.answer_key.json --ratings ~/Downloads/ratings_set-a.json
```

### Next steps
1. Diagnose why the real controls scored 0% / 62% (page labeling, scorer polarity, rater bias), then rate set-a, ideally with
   more than one rater.
2. If emotion separation is still absent, strengthen the signal and retrain: more distinct sad/default descriptions (slower,
   lower, quieter), more style-focused training (epochs, learning rate, more sad data or speakers), and compare against the
   step-370 checkpoint. Use `evaluate_emotion.py measure` as a cheap check before another listening round.

### Artifacts
| what | where | notes |
|---|---|---|
| fine-tuned checkpoint (step 148) | `data/expresso/runs/emotion-v1/export/` | sha256 verified; carries `NONCOMMERCIAL_NOTICE.md`; **not published** |
| training record | `data/expresso/runs/emotion-v1/{train_config.json,environment.json,train.log}` | |
| generated audio + metadata | `outputs/expresso/runs/{baseline,finetuned}/` | raw float32 WAVs + `metadata.jsonl` |
| measurements | `outputs/expresso/measurements.md` | |
| inventory, speaker choice, claim checks | `data/expresso/inventory/` | |
| prepared audio, manifests, splits | `data/expresso/processed/`, `data/expresso/splits/` | test split frozen |
| listening pack of originals | `data/expresso/listening_pack/` | |

## Limitations

- **The unchanged model has not learned this speaker.** The baseline's "ex02" voice is whatever the name token happens to
  evoke and varies from clip to clip. Its speaker-consistency ratings therefore measure how close a generic voice happens to
  be to the reference, not preserved identity. Ordinary adaptation of Parler-TTS to one speaker does not preserve any other
  speaker's identity either; the aim here is to *learn* this speaker.
- About 38 minutes of audio from one speaker, read speech only, two styles. Over-fitting to that speaker's sentences is the
  main risk; validation loss is used for checkpoint selection and the listening test uses sentences never seen in training.
- The acoustic checks on the descriptions are measurements, not perception.
- With a small number of clips and raters, ratings are indicative, not statistically conclusive. So far there is one rater
  on one set, and the hidden controls were misjudged (see Results), so the rating itself needs diagnosing.
- One training run, one seed for training (456) and two for generation: no estimate of run-to-run variance beyond the fact that
  two identical runs agreed on validation loss.
- Same seed does not mean same audio across devices; comparisons are only valid between runs on the same device.
