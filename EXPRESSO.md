# Expresso emotion fine-tuning (experimental, noncommercial)

**Question.** If `parler-tts/parler-tts-mini-v1` is fine-tuned on one Expresso speaker's *default* and *sad*
read speech, does it produce recognizably different delivery for the two styles on text it has never seen,
while staying natural, intelligible and sounding like the same speaker?

**Status.** Two checkpoints have been trained. **emotion-v1** moved the voice to the target speaker but its sad delivery
was too weak to hear (one listener, set-b: 2 of 13 sad clips heard as sad). **emotion-v2** changed only the descriptions, to
words the base model already understands. Its two styles now differ about as much as the real recordings do on every
automatic measure, and **by ear**: in a blinded "which one sounds sadder?" check, one listener picked v2's sad version in
13 of 14 same-sentence pairs (0 the default, 1 "cannot tell"; p = 0.001 against guessing) and the real sad recording in
8 of 8. One listener and 14 pairs: a clear result, not a large study. See [Results](#results). **emotion-v3** adds whisper to the same
speaker: it whispers like the real recordings by every automatic measure, sad and default stay as good as in v2, and a
listener found the whispers good, apart from the clips the automatic check had flagged. **emotion-v4** adds enunciated
(careful, very slow speech): it matches the real recordings' pace (13.6 vs 13.2 characters per second, default 20.6),
the other three styles hold up, and a listener found it really good. `run.sh --style` uses v4.

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
#   adding a style later (emotion-v3: whisper, emotion-v4: enunciated): list it in [speaker] styles, then
#   prepare_expresso.py --extend and evaluate_emotion.py select --extend (existing data, splits and eval items unchanged)
scripts/run_colab_train.sh                             # Colab GPU: baseline generation, fine-tune, fine-tuned generation
#   (--smoke checks the whole pipeline in a few minutes first; --keep-vm-on-failure keeps the VM if a late step fails)
#   --run NAME labels the audio baseline-NAME / finetuned-NAME (emotion-v1 keeps the plain baseline / finetuned)
.venv/bin/python scripts/evaluate_emotion.py measure --runs baseline finetuned     # descriptive acoustics, not ratings

.venv/bin/python scripts/auto_evaluate.py download     # pinned evaluation models (emotion, speaker, Whisper)
.venv/bin/python scripts/auto_evaluate.py calibrate    # check the automatic measures on real recordings, once
.venv/bin/python scripts/auto_evaluate.py score --runs baseline finetuned           # automatic scores, minutes, no listening
#   add --key <key.json> --ratings <ratings.json> to compare with a listener's ratings of the same clips
.venv/bin/python scripts/evaluate_emotion.py generate --set robustness --label robustness-<run> --model <checkpoint dir>
.venv/bin/python scripts/auto_evaluate.py completeness --runs robustness-<run>       # early stops, babbling: flags clips to hear

scripts/run.sh --style default|sad|whisper|enunciated [--check] "Text."  # everyday use of the fine-tune ([speak] run in the config)
.venv/bin/python scripts/evaluate_emotion.py page --runs baseline finetuned --name set-a
#   listen in the browser, export ratings, then:
.venv/bin/python scripts/evaluate_emotion.py score --key <key.json> --ratings <ratings.json>
.venv/bin/python scripts/evaluate_emotion.py pairs --runs <run> --seeds 11 --name <neutral name>
#   a 5-minute "which one sounds sadder?" page (same sentence in both styles); scored with the same `score` command
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
emotion-v1:
- default: *"Expresso speaker ex02 speaks in a neutral delivery. The recording is clear with little background noise."*
- sad: *"Expresso speaker ex02 speaks in a sad, subdued delivery. The recording is clear with little background noise."*

emotion-v2 (current config):
- default: *"Expresso speaker ex02 delivers an expressive and animated speech with a moderate speed and pitch. The recording is clear with little background noise."*
- sad: *"Expresso speaker ex02 delivers a sad, monotone speech, speaking slowly with a low pitch. The recording is clear with little background noise."*

emotion-v3 keeps both texts exactly and adds:
- whisper: *"Expresso speaker ex02 whispers in a soft, breathy voice with no pitch, at a moderate speed. The recording is fairly clear, with some background noise."*

emotion-v4 (current config) keeps all three and adds:
- enunciated: *"Expresso speaker ex02 delivers a careful, clearly enunciated speech, speaking very slowly with a moderate pitch. The recording is clear with little background noise."*

Enunciated claims, from `prepare_expresso.py --extend` (379 of 380 clips accepted, 1 too short): pace 12.6 vs 19.6
characters per active second for default (0.64x) and the slowest style (sad 16.4, whisper 19.3): very slowly; F0 median
232 Hz between sad's 223 and default's 248: moderate pitch; noise floor median -63.1 dBFS, 95th percentile -58.8: the
shared recording sentence holds; "careful, clearly enunciated" is perceptual (Expresso's own style name) and not measured.
Split 303 / 38 / 38; the 1129 earlier manifest rows and every earlier split row are unchanged (compared with a backup).

Why v2: the unchanged model did not respond to "sad" at all (set-b), while "expressive and animated", "monotone", speed
and pitch are the attribute words Parler-TTS Mini was trained on. Every v2 claim holds for ex02, sad relative to default
(table below): F0 range 6.1 vs 11.3 semitones (monotone vs expressive), F0 median 223 vs 248 Hz (low pitch), pace 16.4 vs
19.6 characters per active second (slowly). "Animated" for default is what a listener heard in the real recordings (set-b).
The v1 rows of the table remain as they were.

Whisper claims, from `prepare_expresso.py --extend` (377 of 380 clips accepted, 3 too short): pitch found in 9% of frames
against 67% for default (no pitch); speech level -22.8 vs -19.7 dBFS (soft); pace 19.3 vs 19.6 characters per active second
(moderate speed, as default); noise floor median -49.4 dBFS, above the -55 limit for "little background noise" but 26.6 dB
below the speech (hence "fairly clear, with some background noise" instead of the shared sentence); "breathy" is
perceptual and not measured. Whisper clips joined the split their sentence was already in (301 / 38 / 38, the same shares
as default and sad); the 752 existing examples, their audio and all 76 frozen test rows are unchanged
(`splits/FROZEN.json` records the extension).

| claim | verdict | evidence (ex02, medians) |
|---|---|---|
| clear, little background noise | supported | noise floor -60.1 dBFS (default), -62.7 dBFS (sad); 95th percentile -57.5 / -56.6 |
| sad is subdued | supported | speech level -20.7 vs -19.7 dBFS; F0 223 vs 248 Hz; F0 range 6.1 vs 11.3 semitones; pace 16.4 vs 19.6 characters per active second (share of time speaking is equal) |
| default is neutral | label only, **contradicted by listening** | Expresso's unmarked read style; not measurable. In set-b a listener heard all 8 real default recordings as lively ("upbeat-ish", "passionate - energetic"), not neutral |
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
  whether it sounds sad (not sad / sad / some other emotion / cannot tell; set-b offered "neutral" instead of "not sad"), naturalness (1-5), intelligibility (1-5), and speaker consistency with
  the reference voice (1-5). Ratings stay in the browser and are exported as JSON/CSV; `score` joins them with the key.
- **Nothing is rated automatically.** No claim about emotion, naturalness or identity is made without listener ratings.

## Results

**One blinded listening set (set-b, 72 clips, one rater) has been scored; set-a is unrated.** The real recordings of the
two styles were told apart; the fine-tuned model's were barely told apart, and the baseline's not at all. The fine-tune
did move the voice toward the target speaker. One rater on a small set is indicative, not conclusive. Alongside the
listening result are the machine measurements.

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
`scripts/evaluate_emotion.py score` output, saved as [`results/score_set-b.md`](results/score_set-b.md) with the raw
ratings in [`results/ratings_set-b.json`](results/ratings_set-b.json). 3 of 72 ratings were incomplete and are skipped.
"Not sad" includes "some other emotion"; Likert scores are 1-5.

| run | intended style | n | heard as sad | heard as not sad | cannot tell | naturalness | intelligibility | speaker consistency |
|---|---|---|---|---|---|---|---|---|
| baseline | default | 13 | 8% | 77% | 15% | 2.62 | 4.38 | 1.00 |
| baseline | sad | 14 | 0% | 57% | 43% | 2.43 | 4.36 | 1.00 |
| fine-tuned | default | 13 | 0% | 100% | 0% | 3.08 | 4.69 | 2.92 |
| fine-tuned | sad | 13 | 15% | 62% | 23% | 3.31 | 5.00 | 4.31 |
| real recordings (hidden controls) | default | 8 | 0% | 100% | 0% | 3.12 | 5.00 | 4.00 |
| real recordings (hidden controls) | sad | 8 | 62% | 38% | 0% | 4.50 | 5.00 | 4.50 |

Separation (heard as sad for sad clips minus for default clips): **real recordings +62 points, fine-tuned +15, baseline -8.**

| answers (counts) | neutral | other | sad | cannot tell |
|---|---|---|---|---|
| baseline default / sad | 8 / 6 | 2 / 2 | 1 / 0 | 2 / 6 |
| fine-tuned default / sad | 2 / 4 | 11 / 4 | 0 / 2 | 0 / 3 |
| real default / sad | 0 / 0 | 8 / 3 | 0 / 5 | 0 / 0 |

What this does and does not say:
- **ex02's "default" is not neutral to a listener.** All 8 real default recordings were heard as "some other emotion", with
  notes such as "upbeat-ish" and "passionate - energetic". An earlier version of the scorer counted only "neutral" as
  correct for default, which scored the real default recordings 0% and the fine-tune 15%. That was a mismatch between the
  page's options and this speaker, not a bug in the page or the key (all 72 labels matched). The scorer now measures sad vs
  not sad, and the set-a page asks "Does this clip sound sad?" with "Not sad (neutral, calm or upbeat)" as an option. This
  change was made after seeing set-b, so set-a, with the new wording, is the cleaner test.
- **The fine-tune learned the speaker's default delivery.** Its default clips were heard like the real ones (11 of 13
  "other", against 8 of 8). The baseline's were mostly heard as "neutral".
- **It did not learn a recognizably sad delivery.** Its sad clips were less lively than its default (4 of 13 "other"
  against 11 of 13) but were heard as sad in only 2 of 13, against 5 of 8 real sad recordings. The machine measurements agree:
  the effect is in the right direction but weak (sad is 1.07x longer than default, against 1.18x in the real recordings, and
  the level does not drop).
- **The baseline does not respond to the description at all** (separation -8; many "cannot tell" for its sad clips).
- **The voice and quality improved.** Speaker consistency rises from 1.00 (baseline, which never heard this speaker) to
  2.9-4.3, and naturalness and intelligibility are at or above the baseline. The baseline's 1.00 is not preserved identity;
  see Limitations.
- One rater, 13-14 clips per cell, 8 per control cell: treat differences as indicative only.

### Automatic measures (no listening)
`scripts/auto_evaluate.py` scores generated audio in minutes, so models can be compared without a listening session.
Every measure was checked on real recordings first ([`results/auto_calibration.md`](results/auto_calibration.md)) and
then against the set-b ratings ([`results/auto_score_emotion-v1.md`](results/auto_score_emotion-v1.md)). Models (pinned,
none Chinese-developed): audEERING `wav2vec2-large-robust-12-ft-emotion-msp-dim` (CC BY-NC-SA 4.0), Microsoft
`wavlm-base-plus-sv`, OpenAI `whisper-large-v3-turbo` (MIT).

| measure | what it is | check on real recordings | agreement with the set-b listener |
|---|---|---|---|
| P(sad) | logistic regression on emotion embeddings, trained on ex02's 600 train-split clips | held-out default vs sad: 100% correct | AUC 0.83 for "heard as sad"; **more lenient than the listener** |
| arousal | the emotion model's own output; no training here | AUC 0.95 (ex02), 0.89 (three other speakers) | AUC 0.65 |
| speaker similarity | x-vector cosine similarity to ex02's train-split centroid | ex02 0.93 vs other speakers 0.54 (AUC 0.90) | Spearman +0.74 with speaker consistency |
| WER | Whisper word error rate | 3.1% on real speech | weak (everything is intelligible) |

A naturalness predictor (torchaudio SQUIM MOS) was tried and dropped: it gave 4.43-4.44 for every clip, including the
baseline the listener rated 2.5, so it measured nothing.

What the automatic measures say about emotion-v1 (56 clips per model, both seeds; 8 real recordings per style):

| | classified sad (default / sad) | sad version scores sadder, same sentence | median arousal (default / sad) | speaker similarity |
|---|---|---|---|---|
| baseline | 82% / 89% (not this speaker; meaningless) | 50% | 0.352 / 0.368 | 0.68-0.69 |
| fine-tuned v1 | 11% / 61% | 89% | 0.493 / 0.443 | 0.92-0.95 |
| real recordings | 0% / 100% | 100% | 0.540 / 0.398 | 0.91 |

So v1 learned a measurable part of the sad style (its sad clips usually score sadder than its default clips of the same
sentence) but the effect is about a third of the real one in arousal (a gap of 0.05 against 0.14), and too weak for a
listener to hear as sad (2 of 13). How to use these numbers: compare models of this speaker by the paired "sadder" share
and the arousal gap against the real recordings, then confirm the best one with a short `pairs` page. P(sad) is
meaningless for a voice that is not ex02, and a high P(sad) does not by itself mean a clip sounds sad.

### emotion-v2: descriptions in the base model's vocabulary (automatic measures)
Same data, seed, settings and checkpoint selection as v1 (validation loss lowest at epoch 4 again: 4.022, v1 4.030; the
exported checkpoint is step 148). Only the descriptions changed (see [Descriptions](#descriptions-and-what-was-verified)).
Full output: [`results/auto_score_emotion-v2.md`](results/auto_score_emotion-v2.md); acoustics in `outputs/expresso/measurements.md`.

| | sad version scores sadder, same sentence | sad version has lower arousal | arousal gap (default - sad) | classified sad (default / sad) | speaker similarity |
|---|---|---|---|---|---|
| fine-tuned v1 | 89% | 71% | 0.05 | 11% / 61% | 0.92 / 0.95 |
| **fine-tuned v2** | **100%** | **96%** | **0.13** | **4% / 89%** | 0.90 / 0.95 |
| real recordings | 100% | 100% | 0.14 | 0% / 100% | 0.91 / 0.92 |

| acoustics (medians) | sad / default duration | F0 median default -> sad | F0 range default -> sad | level sad - default |
|---|---|---|---|---|
| fine-tuned v1 | 1.07 | 260 -> 252 Hz | 8.9 -> 7.3 st | +0.4 dB |
| **fine-tuned v2** | **1.14** | **251 -> 218 Hz** | **10.7 -> 6.7 st** | **-1.3 dB** |
| real recordings | 1.18 | 263 -> 234 Hz | 10.8 -> 7.6 st | -1.4 dB |

- **v2 reproduces the real contrast on every measure**, in size as well as direction; v1 had the direction but about a
  third of the size. Its sad arousal (0.379) is close to the real sad recordings (0.398).
- **The words matter more than the training.** The unchanged model, given the v2 descriptions, already speaks the two
  styles differently (sad version lower in arousal for 89% of pairs), unlike with v1's "sad" (39%). But it does so by
  switching voices (sad F0 129 Hz, speaker similarity 0.45): fine-tuning is what keeps it the same speaker.
- **One new failure:** one v2 clip of 56 stops after its first word ("Well," then silence, 0.43 s; `new03_default_s11`),
  which is most of v2's 5.8% default word error rate. v1 had none under 1 s. Too few clips to say whether this is a
  real regression.
- These are automatic measures. They were checked on real recordings and against one listener, and they are more
  lenient than that listener was, so v2 was then checked by ear (below).

### Listening check of emotion-v2 (check-1: 22 pairs, one listener)
Blinded forced choice: the same sentence in both styles, played as A and B in random order, "which one sounds sadder?".
14 pairs from v2 (seed 11) and 8 pairs of real recordings as a check. Answers in
[`results/pairs_check-1.json`](results/pairs_check-1.json), score in [`results/score_check-1.md`](results/score_check-1.md).

| | pairs | picked the sad version | picked the default version | cannot tell |
|---|---|---|---|---|
| **fine-tuned v2** | 14 | **93%** (13) | 0% | 7% (1) |
| real recordings | 8 | 100% (8) | 0% | 0% |

- **The v2 sad delivery is audible.** 13 of 14 is very unlikely by guessing (p = 0.001, one-sided binomial, the
  "cannot tell" counted as a miss). The one tie was *"The package arrived on Tuesday afternoon."*
- This is a relative judgement (which of two is sadder), easier than set-b's absolute one (does this clip sound sad).
  It shows the two styles are told apart in the right direction, not how sad v2's sad clips sound on their own.
  Naturalness and speaker consistency were not re-rated for v2; the automatic speaker similarity is unchanged from v1.
- One listener, who had also rated set-b. Another listener would make it stronger.

### emotion-v2 robustness: early stops (120 new sentences, automatic)
The eval set showed one v2 clip stopping after its first word. To measure how often that happens, v2 spoke 60 new
sentences (20 short of 2-4 words, 20 medium of 8-11, 20 long of 23-31; several starting "Well," / "Oh," / "So,") in both
styles on this Mac, with the evaluated checkpoint, descriptions and settings. `auto_evaluate.py completeness` flags a clip
as incomplete if Whisper's transcript lacks the end of the text, the clip ends mid-sound (its last 100 ms less than 15 dB
below its own speech level), or it is under half the time ex02 needs for the text. The same rule flags 0 of 152 real
held-out recordings. Full output: [`results/completeness_emotion-v2.md`](results/completeness_emotion-v2.md).

| | clips | incomplete (stopped early) | complete but worth a listen |
|---|---|---|---|
| **v2, new sentences** | 120 | **1** (long, default) | 4 (all short) |
| v2, eval set | 56 | 1 (the known "Well,") | 0 |
| v1, eval set | 56 | 0 | 0 |
| real recordings | 152 | 0 | 6 (Whisper spelling, e.g. "6.45 a.m.") |

- **v2 stops early in about 1% of clips** (2 of 176; 95% CI roughly 0.3-4%), both in the default style (2 of 88) and
  none of 88 sad. Both end abruptly mid-word (3-5 dB below the speech level, where every complete clip is 21 dB or more).
  The "Well," start is not the cause: all six sentences beginning "Well," / "Oh," / "So," were complete.
- **Very short sentences get padding:** 3 of 40 short clips run several seconds long: *"I know."* (both styles: about a
  second of speech, then seconds of extra sound) and *"Nobody called back."* (2 s of silence first). Medium and long
  sentences had no such problem.
- v1 had none in 56 clips, too few to say whether v2 is worse.

### emotion-v3: adding whisper (automatic measures and a listen)
v3 is v2's recipe on default + sad + whisper: the same data and descriptions for default and sad, whisper added with
`prepare_expresso.py --extend` (see [Descriptions](#descriptions-and-what-was-verified)), and the same seed and settings.
Validation loss was lowest at step 224 (epoch 4, as for v1 and v2), the exported checkpoint. The 3-style classifier and
the voicing cue were checked on real recordings first ([`results/auto_calibration.md`](results/auto_calibration.md):
100% of held-out real clips classified correctly; little voicing separates real whisper from default perfectly, for ex02
and the three other speakers). Full output: [`results/auto_score_emotion-v3.md`](results/auto_score_emotion-v3.md),
[`results/completeness_emotion-v3.md`](results/completeness_emotion-v3.md).

| 28 clips per style (14 sentences x 2 seeds) | classified as intended | voicing | same-sentence pairs: higher P / stronger cue than default | WER | speaker similarity |
|---|---|---|---|---|---|
| **v3 whisper** | **96%** | **10%** | **100% / 100%** | 4.8% | 0.86 |
| real whisper (8 clips) | 100% | 9% | 100% / 100% | 5.6% | 0.92 |
| v3 sad | 89% | 69% | 100% / 96% | 1.4% | 0.92 |
| v2 sad (same classifier) | 93% | 66% | 100% / 96% | 1.4% | 0.90 |
| v3 default | 96% | 70% | | 1.9% | 0.91 |

- **Whisper is learned**: nearly unvoiced like the real thing and as intelligible to Whisper ASR. The unchanged model,
  given the same description, does not whisper at all (67% voiced): the word alone is not enough, the fine-tune is.
- **Sad and default are unchanged** from v2 within noise; adding a style cost nothing measurable.
- **Whispers are a little less like the speaker's voice** to the x-vector model (0.86 vs 0.92 for her real whispers).
- **Early stops**: none in 56 default and sad clips. In whisper, 3 of 28 clips were flagged by the transcript check
  (garbled words, or the last words not heard); a listener confirmed those three sound wonky and found the rest of the
  whispers good (4 sentences heard next to the real recordings). The audio-only checks that `run.sh --style` retries on
  do not catch these; the robustness batch below shows that they depend on sentence length.

### emotion-v4: adding enunciated (automatic measures and a listen)
The same recipe as v3, run `runs/emotion-v4` (same seed and settings, 1204 training clips). Validation loss was lowest at
step 450 (epoch 6; v1-v3 chose epoch 4 of smaller data), the exported checkpoint. Calibration first, with enunciated as a
fourth class: 100% of held-out real clips classified correctly, P(enunciated) separates real enunciated from default
perfectly ([`results/auto_calibration.md`](results/auto_calibration.md)). Full output:
[`results/auto_score_emotion-v4.md`](results/auto_score_emotion-v4.md), [`results/completeness_emotion-v4.md`](results/completeness_emotion-v4.md).

| 28 clips per style | classified as intended | pace (chars / active s) | pairs: higher P(style) than default | WER | speaker similarity |
|---|---|---|---|---|---|
| **v4 enunciated** | **75%** (11% default, 14% sad) | **13.6** | **100%** | 1.0% | 0.95 |
| real enunciated (8 clips) | 100% | 13.2 | 100% | 1.9% | 0.95 |
| unchanged model, same description | 0% | | 54% | 4.8% | 0.63 |
| v4 default | 100% | 20.6 | | 1.0% | 0.94 |
| v4 sad (v3) | 100% (93%) | | 100% (100%) | 2.9% (1.4%) | 0.90 (0.93) |
| v4 whisper (v3) | 100% (96%) | | 100% (96%) | 13.0% (4.8%) | 0.87 (0.84) |

- **Enunciated is learned**: as slow as the real recordings and told apart from default in every pair. The unchanged
  model does not slow down for the description at all. It is less distinct than whisper: a quarter of the clips are
  classified as default or sad.
- **Default, sad and whisper hold up.** Whisper's higher WER comes from two garbled clips ("put it in the coat. Put it in
  the coat."); 4 of 28 whispers were flagged against 3 of 28 for v3. No early stops in 112 clips.
- **Listening**: a listener heard enunciated and default versions of the same sentences and found enunciated really good.

Robustness batch for v4, the 60 `[robustness]` sentences spoken whole in every style
([`results/completeness_emotion-v4-robustness.md`](results/completeness_emotion-v4-robustness.md)); cells incomplete / to check:

| style | short | medium | long | v3 long |
|---|---|---|---|---|
| default | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| sad | 0 / 2 | 0 / 0 | 0 / 0 | 0 / 0 |
| whisper | 0 / 1 | 0 / 2 | **3 / 1** | 8 / 4 |
| enunciated | 0 / 0 | 0 / 0 | 0 / 0 | |

Enunciated is as robust as default (0 of 60 flagged, even though its long sentences run 15-20 s). Long whispers spoken
whole improved from 12 of 20 flagged (v3) to 4 of 20; why is not tested (one seed, and v4 trained two epochs longer on
more data). `run.sh --style whisper` still speaks long text in pieces, which flagged 0 of 20 with `--check` on v3.

### Robustness batch (v3)
The 60 `[robustness]` sentences (20 short, 20 medium, 20 long of 23-31 words) in every style, seed 11, judged by the same
rule ([`results/completeness_emotion-v3-robustness.md`](results/completeness_emotion-v3-robustness.md)). Cells:
incomplete / complete but flagged to check (garbled).

| style | short | medium | long | incomplete overall |
|---|---|---|---|---|
| default | 0 / 1 | 0 / 0 | 0 / 0 | 0/60 (95% CI 0-6%) |
| sad | 0 / 2 | 0 / 0 | 0 / 0 | 0/60 (0-6%) |
| **whisper** | 0 / 0 | 0 / 2 | **8 / 4** | **8/60 (7-24%)** |

- **Default and sad are as robust as in v2** (0 of 120 incomplete, as v2's batch). The 3 short clips flagged for a word
  ("Come here, fleece") were clean when generated again.
- **Whisper fails on long sentences**: 12 of 20 long whispers stop early (8) or garble words (4), against 2 of 20 medium
  and none of the short ones. Every style was trained on equally short sentences (median 7 words, at most 25), so length
  alone does not explain it. A likely reason, not tested: the unchanged model cannot whisper, so whisper rests on about
  300 short clips, while default and sad build on the base model's own long-sentence ability.
- **The audio-only checks catch none of the 17 flagged clips**: they end in silence at a plausible length. Whisper's
  transcript is needed, hence `--check` below.

### Everyday use
`scripts/run.sh --style default|sad|whisper|enunciated "Text."` speaks with the checkpoint named in `[speak]` of the config (emotion-v4; v3 and v2 before),
using the descriptions stored with that checkpoint (`export_info.json`) and the evaluated generation settings. A check of
3 new sentences per style gave P(sad) 0.00-0.06 for default and 0.94-1.00 for sad, lower arousal for sad in all 3 pairs,
speaker similarity 0.87-0.97 and exact transcripts. Without `--style`, `run.sh` is unchanged.

With `--style`, a clip that is cut off mid-sound, under half or over 2.5x (plus 1 s) the time ex02 needs for its text is
generated again, up to twice, and the reason is printed. These are the audio-only checks of `completeness`
(`expresso_common.audio_problem`, no Whisper): they flag none of the 152 real recordings and none of v1's 56 clips, and
they catch all five problem clips above (both cut-offs and the three padded short ones). In a first try, 1 of 8 short
clips was regenerated (it ended in a burst 15 dB louder than its speech) and the second attempt was clean. A wrong
word in the middle of a clip is not caught: that would need Whisper.

`--check` (with `--style`) adds Whisper: each clip is transcribed and generated again if it fails `completeness`'s rule
(last words missing, or 25%+ of the words wrong). If all three tries fail, the least bad is kept (complete but garbled
over incomplete). Tested on the 17 sentences the robustness batch flagged, generated again with `--check`: the 2 medium
whispers and the 3 default and sad clips were clean on the first try; of the 12 long whispers, 9 ended clean (3 at once,
4 after one retry, 2 after two) and 3 still failed after two retries (9 of 26 tries clean, 35%). Cost: a few seconds of
Whisper per clip, plus about a minute per retry of a long whisper. The rule flags 9 of 228 real recordings for Whisper's
spelling ("straight jacket"), so it occasionally regenerates a good clip too.

Whisper text is spoken in pieces of at most 14 words (`speak.py`'s `pieces`: cut at sentence ends, then commas, then
before a conjunction or preposition), each generated and checked on its own and joined with short pauses (0.25 s after a
comma, 0.45 s after a sentence). Default and sad are spoken whole. On the 20 long robustness sentences, the joined clips
judged against the full text by the same rule:

| long whispers (20) | flagged |
|---|---|
| whole (the batch above) | 12 |
| in pieces | 1 (the last 2 words missing) |
| in pieces, with `--check` | 0 (11 retries over about 57 pieces; every piece ended clean) |

About a minute per long whisper on MPS when no piece needs a retry. A listener heard 5 of these (including the awkward
cut "her bed was | so full") and found the joins and pauses pretty good.

The local page, `scripts/serve.py`, does the same through a browser: it loads the checkpoint once (so only generation
time is paid per clip, not the 10-20 s model load), with the style buttons and "check words" (`--check`) on by default.
Both use `scripts/styled_voice.py`. When Whisper hears a complete clip exactly as it heard the try before, the clip is kept:
the text is spelled differently from how it is heard ("can not" / "cannot"), and more tries would only repeat it.

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
1. More styles on the same recipe, each with `--extend`, a new run, automatic scoring and a short listen. Measured on 60
   real clips per style, how far each is from ex02's default (best single measure, AUC): confused 0.94 (pace), laughing
   0.87-0.91 (arousal, valence), happy only 0.69 (arousal; her default already sounds upbeat), so happy is the hardest to
   learn and to confirm. Whisper (1.00) and enunciated (0.97) are done.
2. Optional: a second listener, absolute naturalness ratings, a listen to the 17 clips the v3 batch flagged (does each
   really sound wrong?).

### Artifacts
| what | where | notes |
|---|---|---|
| emotion-v1 checkpoint (step 148) | `data/expresso/runs/emotion-v1/export/` | weights deleted 2026-10-16 to free 3.3 GB (sha256 in `export_info.json`; retrain with the same seed); records kept |
| emotion-v2 checkpoint (step 148) | `data/expresso/runs/emotion-v2/export/` | sha256 verified; carries `NONCOMMERCIAL_NOTICE.md`; **not published**; v2 descriptions |
| emotion-v3 checkpoint (step 224) | `data/expresso/runs/emotion-v3/export/` | same; default, sad, whisper |
| emotion-v4 checkpoint (step 450) | `data/expresso/runs/emotion-v4/export/` | same; default, sad, whisper, enunciated; used by `run.sh --style` |
| automatic scores | `outputs/expresso/auto/` (per-clip JSON, cache), summaries in `results/auto_*.md`, `results/completeness_*.md` | |
| robustness batch (v2) | `outputs/expresso/runs/robustness-finetuned-emotion-v2/` | 120 clips |
| training record | `data/expresso/runs/emotion-v1/{train_config.json,environment.json,train.log}` | |
| generated audio + metadata | `outputs/expresso/runs/{baseline,finetuned}/` (v1), `{baseline,finetuned}-emotion-v2/` | raw float32 WAVs + `metadata.jsonl` |
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
  on one set.
- One training run, one seed for training (456) and two for generation: no estimate of run-to-run variance beyond the fact that
  two identical runs agreed on validation loss.
- Same seed does not mean same audio across devices; comparisons are only valid between runs on the same device.
