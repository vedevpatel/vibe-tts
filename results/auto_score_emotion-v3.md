# Automatic measures on the evaluation set

Classifier and speaker centroid from `calibrate` (2026-10-14T22:32:58Z; styles default, sad, whisper; held-out real recordings classified correctly: 100%).
'real recordings' are the speaker's own held-out recordings of the same test sentences (no new sentences exist for them).

| run | style | n | classified default | classified sad | classified whisper | median arousal | median voicing | WER | median speaker similarity |
|---|---|---|---|---|---|---|---|---|---|
| baseline-emotion-v3 | default | 28 | 29% | 71% | 0% | 0.414 | 71% | 2.4% | 0.791 |
| baseline-emotion-v3 | sad | 28 | 0% | 100% | 0% | 0.317 | 70% | 1.9% | 0.432 |
| baseline-emotion-v3 | whisper | 28 | 14% | 86% | 0% | 0.349 | 67% | 2.9% | 0.693 |
| finetuned-emotion-v2 | default | 28 | 96% | 4% | 0% | 0.509 | 68% | 5.8% | 0.919 |
| finetuned-emotion-v2 | sad | 28 | 7% | 93% | 0% | 0.379 | 66% | 1.4% | 0.903 |
| finetuned-emotion-v3 | default | 28 | 96% | 0% | 4% | 0.507 | 70% | 1.9% | 0.912 |
| finetuned-emotion-v3 | sad | 28 | 7% | 89% | 4% | 0.374 | 69% | 1.4% | 0.923 |
| finetuned-emotion-v3 | whisper | 28 | 0% | 4% | 96% | 0.563 | 10% | 4.8% | 0.855 |
| real recordings | default | 8 | 100% | 0% | 0% | 0.540 | 68% | 1.9% | 0.941 |
| real recordings | sad | 8 | 0% | 100% | 0% | 0.398 | 67% | 1.9% | 0.857 |
| real recordings | whisper | 8 | 0% | 0% | 100% | 0.527 | 9% | 5.6% | 0.915 |

Paired comparison with default, the automatic version of "which of these two is more <style>?": for each sentence (and seed), the share of pairs where the style's version scores higher on P(style) or on the style's own cue (low arousal for sad, little voicing for whisper) than the default version. 50% = not told apart.

| run | style | pairs | higher P(style) | stronger cue | separation in classified style (points) |
|---|---|---|---|---|---|
| baseline-emotion-v3 | sad | 28 | 93% | 89% | +29 |
| baseline-emotion-v3 | whisper | 28 | 39% | 50% | +0 |
| finetuned-emotion-v2 | sad | 28 | 100% | 96% | +89 |
| finetuned-emotion-v3 | sad | 28 | 100% | 96% | +89 |
| finetuned-emotion-v3 | whisper | 28 | 100% | 100% | +93 |
| real recordings | sad | 8 | 100% | 100% | +100 |
| real recordings | whisper | 8 | 100% | 100% | +100 |

These are automatic proxies, not ratings. The classifier is trained on this speaker's real recordings: it is meaningless for a voice that is not this speaker (the unchanged baseline), and for sad it was more lenient than a listener (set-b), so use it to compare models of this voice, not as proof of how a clip sounds. The cues (arousal, voicing) need no training and work across speakers.
