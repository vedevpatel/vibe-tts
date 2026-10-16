# Automatic measures on the evaluation set

Classifier and speaker centroid from `calibrate` (2026-10-16T19:49:45Z; styles default, enunciated, sad, whisper; held-out real recordings classified correctly: 100%).
'real recordings' are the speaker's own held-out recordings of the same test sentences (no new sentences exist for them).

| run | style | n | classified default | classified enunciated | classified sad | classified whisper | median arousal | median voicing | WER | median speaker similarity |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline-emotion-v4 | default | 28 | 36% | 0% | 64% | 0% | 0.414 | 71% | 2.4% | 0.805 |
| baseline-emotion-v4 | sad | 28 | 4% | 4% | 93% | 0% | 0.317 | 70% | 1.9% | 0.441 |
| baseline-emotion-v4 | whisper | 28 | 14% | 0% | 82% | 4% | 0.349 | 67% | 2.9% | 0.712 |
| baseline-emotion-v4 | enunciated | 28 | 11% | 0% | 89% | 0% | 0.329 | 68% | 4.8% | 0.634 |
| finetuned-emotion-v3 | default | 28 | 96% | 0% | 0% | 4% | 0.507 | 70% | 1.9% | 0.904 |
| finetuned-emotion-v3 | sad | 28 | 0% | 4% | 93% | 4% | 0.374 | 69% | 1.4% | 0.928 |
| finetuned-emotion-v3 | whisper | 28 | 0% | 0% | 4% | 96% | 0.563 | 10% | 4.8% | 0.841 |
| finetuned-emotion-v4 | default | 28 | 100% | 0% | 0% | 0% | 0.501 | 72% | 1.0% | 0.941 |
| finetuned-emotion-v4 | sad | 28 | 0% | 0% | 100% | 0% | 0.368 | 66% | 2.9% | 0.900 |
| finetuned-emotion-v4 | whisper | 28 | 0% | 0% | 0% | 100% | 0.546 | 15% | 13.0% | 0.866 |
| finetuned-emotion-v4 | enunciated | 28 | 11% | 75% | 14% | 0% | 0.438 | 69% | 1.0% | 0.946 |
| real recordings | default | 8 | 100% | 0% | 0% | 0% | 0.540 | 68% | 1.9% | 0.940 |
| real recordings | sad | 8 | 0% | 0% | 100% | 0% | 0.398 | 67% | 1.9% | 0.866 |
| real recordings | whisper | 8 | 0% | 0% | 0% | 100% | 0.527 | 9% | 5.6% | 0.903 |
| real recordings | enunciated | 8 | 0% | 100% | 0% | 0% | 0.473 | 72% | 1.9% | 0.947 |

Paired comparison with default, the automatic version of "which of these two is more <style>?": for each sentence (and seed), the share of pairs where the style's version scores higher on P(style) or on the style's own cue (low arousal for sad, little voicing for whisper) than the default version. 50% = not told apart.

| run | style | pairs | higher P(style) | stronger cue | separation in classified style (points) |
|---|---|---|---|---|---|
| baseline-emotion-v4 | sad | 28 | 82% | 89% | +29 |
| baseline-emotion-v4 | whisper | 28 | 29% | 50% | +4 |
| baseline-emotion-v4 | enunciated | 28 | 54% | - | +0 |
| finetuned-emotion-v3 | sad | 28 | 100% | 96% | +93 |
| finetuned-emotion-v3 | whisper | 28 | 96% | 100% | +93 |
| finetuned-emotion-v4 | sad | 28 | 100% | 100% | +100 |
| finetuned-emotion-v4 | whisper | 28 | 100% | 100% | +100 |
| finetuned-emotion-v4 | enunciated | 28 | 100% | - | +75 |
| real recordings | sad | 8 | 100% | 100% | +100 |
| real recordings | whisper | 8 | 100% | 100% | +100 |
| real recordings | enunciated | 8 | 100% | - | +100 |

These are automatic proxies, not ratings. The classifier is trained on this speaker's real recordings: it is meaningless for a voice that is not this speaker (the unchanged baseline), and for sad it was more lenient than a listener (set-b), so use it to compare models of this voice, not as proof of how a clip sounds. The cues (arousal, voicing) need no training and work across speakers.
