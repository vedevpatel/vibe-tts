Ratings from 1 file(s). Emotion accuracy = rater's choice equals the intended style.

| run | intended style | n | emotion correct | naturalness | intelligibility | speaker consistency |
|---|---|---|---|---|---|---|
| baseline | default | 13 | 62% | 2.62 | 4.38 | 1.00 |
| baseline | sad | 14 | 0% | 2.43 | 4.36 | 1.00 |
| finetuned | default | 13 | 15% | 3.08 | 4.69 | 2.92 |
| finetuned | sad | 13 | 15% | 3.31 | 5.00 | 4.31 |
| reference | default | 8 | 0% | 3.12 | 5.00 | 4.00 |
| reference | sad | 8 | 62% | 4.50 | 5.00 | 4.50 |

Likert scales are 1-5 (higher is better). Means over clips and raters; with few clips treat differences as indicative only.
Baseline caveat: the unchanged model has never heard this Expresso speaker, so its 'speaker consistency' against the reference voice measures how close a generic voice happens to be, not preserved identity. Adaptation to this speaker is the point of the fine-tune; it does not preserve any other speaker's identity.
