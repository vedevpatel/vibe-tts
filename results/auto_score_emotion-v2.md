# Automatic measures on the evaluation set

Classifier and speaker centroid from `calibrate` (2026-10-12T22:07:00Z): held-out real default vs sad accuracy 100%, AUC 1.000.
'real recordings' are the speaker's own held-out recordings of the same test sentences (no new sentences exist for them).

| run | style | n | classified sad | mean P(sad) | median arousal | median valence | WER | median speaker similarity |
|---|---|---|---|---|---|---|---|---|
| baseline-emotion-v2 | default | 28 | 71% | 0.64 | 0.414 | 0.386 | 2.4% | 0.801 |
| baseline-emotion-v2 | sad | 28 | 96% | 0.97 | 0.317 | 0.345 | 1.9% | 0.454 |
| finetuned-emotion-v2 | default | 28 | 4% | 0.05 | 0.509 | 0.399 | 5.8% | 0.895 |
| finetuned-emotion-v2 | sad | 28 | 89% | 0.91 | 0.379 | 0.346 | 1.4% | 0.945 |
| real recordings | default | 8 | 0% | 0.01 | 0.540 | 0.378 | 1.9% | 0.910 |
| real recordings | sad | 8 | 100% | 0.99 | 0.398 | 0.307 | 1.9% | 0.915 |

Paired comparison, the automatic version of "which of these two sounds sadder?": for each sentence (and seed), the share of pairs where the sad version scores sadder than the default version. 50% = the styles are not told apart.

| run | pairs | sad version has higher P(sad) | sad version has lower arousal | separation in classified sad (points) |
|---|---|---|---|---|
| baseline-emotion-v2 | 28 | 89% | 89% | +25 |
| finetuned-emotion-v2 | 28 | 100% | 96% | +86 |
| real recordings | 8 | 100% | 100% | +100 |

These are automatic proxies, not ratings. 'Classified sad' and P(sad) come from a classifier trained on this speaker's real recordings: it is meaningless for a voice that is not this speaker (the unchanged baseline is classified sad in both styles), and it is more lenient than a listener (set-b), so use it to compare models of this voice, not as proof that a clip sounds sad. Arousal needs no training and works across speakers, but agrees less with a listener.
