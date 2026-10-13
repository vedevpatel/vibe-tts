# Automatic measures: calibration on real recordings (ex02)

Train split 600 clips; held-out (validation + test) 152 clips; other Expresso speakers 90 clips.
AUC: 0.5 = no better than chance, 1.0 = perfect separation.

| measure | check | result |
|---|---|---|
| sadness classifier | held-out default vs sad, accuracy | 100% |
| sadness classifier | held-out default vs sad, AUC | 1.000 |
| low arousal | held-out ex02 default vs sad, AUC | 0.949 |
| low arousal | other speakers default vs sad, AUC | 0.886 |
| low valence | held-out ex02 default vs sad, AUC | 0.692 |
| low valence | other speakers default vs sad, AUC | 0.632 |
| low dominance | held-out ex02 default vs sad, AUC | 0.952 |
| low dominance | other speakers default vs sad, AUC | 0.901 |
| speaker similarity | held-out ex02 vs other speakers, AUC | 0.895 |
| speaker similarity | median: held-out ex02 / other speakers | 0.925 / 0.544 |
| word error rate | held-out real recordings | 3.1% |

Held-out real recordings by style:

| style | mean P(sad) | median arousal | median valence | median dominance |
|---|---|---|---|---|
| default | 0.01 | 0.530 | 0.499 | 0.588 |
| sad | 1.00 | 0.385 | 0.410 | 0.487 |
