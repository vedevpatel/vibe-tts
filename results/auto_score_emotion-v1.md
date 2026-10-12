# Automatic measures on the evaluation set

Classifier and speaker centroid from `calibrate` (2026-10-12T22:07:00Z): held-out real default vs sad accuracy 100%, AUC 1.000.
'real recordings' are the speaker's own held-out recordings of the same test sentences (no new sentences exist for them).

| run | style | n | classified sad | mean P(sad) | median arousal | median valence | WER | median speaker similarity |
|---|---|---|---|---|---|---|---|---|
| baseline | default | 28 | 82% | 0.81 | 0.352 | 0.389 | 2.4% | 0.680 |
| baseline | sad | 28 | 89% | 0.84 | 0.368 | 0.363 | 3.4% | 0.692 |
| finetuned | default | 28 | 11% | 0.12 | 0.493 | 0.399 | 1.0% | 0.915 |
| finetuned | sad | 28 | 61% | 0.62 | 0.443 | 0.374 | 1.0% | 0.945 |
| real recordings | default | 8 | 0% | 0.01 | 0.540 | 0.378 | 1.9% | 0.910 |
| real recordings | sad | 8 | 100% | 0.99 | 0.398 | 0.307 | 1.9% | 0.915 |

Paired comparison, the automatic version of "which of these two sounds sadder?": for each sentence (and seed), the share of pairs where the sad version scores sadder than the default version. 50% = the styles are not told apart.

| run | pairs | sad version has higher P(sad) | sad version has lower arousal | separation in classified sad (points) |
|---|---|---|---|---|
| baseline | 28 | 50% | 39% | +7 |
| finetuned | 28 | 89% | 71% | +50 |
| real recordings | 8 | 100% | 100% | +100 |

## Agreement with listener ratings (69 rated clips from 1 file(s))

| automatic measure | listener rating | agreement |
|---|---|---|
| P(sad) | heard as sad (8 clips) vs not | AUC 0.826 |
| low arousal | heard as sad vs not | AUC 0.654 |
| speaker similarity | speaker consistency (1-5) | Spearman +0.74 |
| word errors (negated) | intelligibility (1-5) | Spearman +0.19 |

| run | style | n | listener: heard as sad | automatic: classified sad |
|---|---|---|---|---|
| baseline | default | 13 | 8% | 92% |
| baseline | sad | 14 | 0% | 93% |
| finetuned | default | 13 | 0% | 15% |
| finetuned | sad | 13 | 15% | 46% |
| real recordings | default | 8 | 0% | 0% |
| real recordings | sad | 8 | 62% | 100% |

AUC 0.5 = no agreement, 1.0 = perfect; Spearman 0 = no agreement, +1 = same ranking.

These are automatic proxies, not ratings. 'Classified sad' and P(sad) come from a classifier trained on this speaker's real recordings: it is meaningless for a voice that is not this speaker (the unchanged baseline is classified sad in both styles), and it is more lenient than a listener (set-b), so use it to compare models of this voice, not as proof that a clip sounds sad. Arousal needs no training and works across speakers, but agrees less with a listener.
