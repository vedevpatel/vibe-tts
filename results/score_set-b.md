Ratings from 1 file(s); 3 incomplete rating(s) skipped.
Heard as sad / not sad / cannot tell = share of clips given that answer; 'not sad' includes 'some other emotion'.

| run | intended style | n | heard as sad | heard as not sad | cannot tell | naturalness | intelligibility | speaker consistency |
|---|---|---|---|---|---|---|---|---|
| baseline | default | 13 | 8% | 77% | 15% | 2.62 | 4.38 | 1.00 |
| baseline | sad | 14 | 0% | 57% | 43% | 2.43 | 4.36 | 1.00 |
| finetuned | default | 13 | 0% | 100% | 0% | 3.08 | 4.69 | 2.92 |
| finetuned | sad | 13 | 15% | 62% | 23% | 3.31 | 5.00 | 4.31 |
| reference | default | 8 | 0% | 100% | 0% | 3.12 | 5.00 | 4.00 |
| reference | sad | 8 | 62% | 38% | 0% | 4.50 | 5.00 | 4.50 |

Separation = heard as sad for sad clips minus heard as sad for default clips (percentage points; 0 means the two styles were not told apart):

| run | separation |
|---|---|
| baseline | -8 |
| finetuned | +15 |
| reference | +62 |

All answers (counts):

| run | intended style | neutral | other | sad | unsure |
|---|---|---|---|---|---|
| baseline | default | 8 | 2 | 1 | 2 |
| baseline | sad | 6 | 2 | 0 | 6 |
| finetuned | default | 2 | 11 | 0 | 0 |
| finetuned | sad | 4 | 4 | 2 | 3 |
| reference | default | 0 | 8 | 0 | 0 |
| reference | sad | 0 | 3 | 5 | 0 |

Likert scales are 1-5 (higher is better). Means over clips and raters; with few clips treat differences as indicative only.
Baseline caveat: the unchanged model has never heard this Expresso speaker, so its 'speaker consistency' against the reference voice measures how close a generic voice happens to be, not preserved identity. Adaptation to this speaker is the point of the fine-tune; it does not preserve any other speaker's identity.
