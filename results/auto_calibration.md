# Automatic measures: calibration on real recordings (ex02)

Train split 901 clips; held-out (validation + test) 228 clips; other Expresso speakers 135 clips. Styles: default, sad, whisper.
AUC: 0.5 = no better than chance, 1.0 = perfect separation.

| measure | check | result |
|---|---|---|
| style classifier (3 styles) | held-out, share classified as the right style | 100% |
| P(sad) | held-out default vs sad, AUC | 1.000 |
| low arousal (no training) | held-out ex02 default vs sad, AUC | 0.949 |
| low arousal (no training) | other speakers default vs sad, AUC | 0.874 |
| P(whisper) | held-out default vs whisper, AUC | 1.000 |
| low voicing (no training) | held-out ex02 default vs whisper, AUC | 1.000 |
| low voicing (no training) | other speakers default vs whisper, AUC | 1.000 |
| speaker similarity | held-out ex02 vs other speakers, AUC | 0.836 |
| speaker similarity | median: held-out ex02 / other speakers | 0.892 / 0.594 |
| word error rate | held-out real default recordings | 2.7% |
| word error rate | held-out real sad recordings | 3.4% |
| word error rate | held-out real whisper recordings | 2.5% |

Held-out real recordings by style (classified as: share of clips the classifier puts in each style):

| style | classified default | classified sad | classified whisper | median arousal | median valence | median voicing |
|---|---|---|---|---|---|---|
| default | 100% | 0% | 0% | 0.530 | 0.499 | 67% |
| sad | 0% | 100% | 0% | 0.385 | 0.410 | 64% |
| whisper | 0% | 0% | 100% | 0.546 | 0.438 | 9% |
