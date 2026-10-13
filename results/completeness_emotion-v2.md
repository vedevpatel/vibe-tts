# Early stops and other incomplete speech

Rule: incomplete if Whisper's transcript lacks the last 2+ words (or the last third) of the text, the clip ends mid-sound (last 100 ms less than 15 dB below its speech level), or it is under 0.5x the time ex02 needs for the text (characters / ex02's speaking pace for that style). 'Check' = complete but over 2.5x that time, or a word error rate of 25% or more.

**The rule on 152 real held-out recordings:** 0/152 (0%; 95% CI 0%-2%) flagged incomplete, 6 flagged to check.

- real `ex02_default_00087` (check: word error rate 29%): "Me in a straitjacket, heading for Ravenwood." heard as "Me in a straight jacket, heading for Ravenwood."
- real `ex02_default_00088` (check: word error rate 25%): "How you holding up?" heard as "How are you holding up?"
- real `ex02_default_00147` (check: word error rate 29%): "Jacks's always pay off their debts." heard as "JAXs always pay off their debts."
- real `ex02_sad_00087` (check: word error rate 29%): "Me in a straitjacket, heading for Ravenwood." heard as ""'Me in a straight jacket, heading for Ravenwood.'"
- real `ex02_sad_00147` (check: word error rate 29%): "Jacks's always pay off their debts." heard as "Jackses always pay off their debts."
- real `ex02_sad_00260` (check: word error rate 33%): "Okay, alarm set for this Saturday at six forty five AM." heard as "Okay. Alarm set for this Saturday at 6.45 a.m."

## robustness-finetuned-emotion-v2

| length | style | clips | incomplete | to check |
|---|---|---|---|---|
| short | default | 20 | 0 | 2 |
| short | sad | 20 | 0 | 2 |
| medium | default | 20 | 0 | 0 |
| medium | sad | 20 | 0 | 0 |
| long | default | 20 | 1 | 0 |
| long | sad | 20 | 0 | 0 |

**Incomplete overall: 1/120 (1%; 95% CI 0%-5%).**
- default: 1/60 (2%; 95% CI 0%-9%)
- sad: 0/60 (0%; 95% CI 0%-6%)

Flagged clips (listen to these):

- `short00_default_s11` check: 5.13 s, more than 2.5x the 0.36 s this text needs. "I know." heard as "I know... Heeeeeeem..." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v2/raw/short00_default_s11.wav`)
- `short00_sad_s11` check: 7.13 s, more than 2.5x the 0.43 s this text needs. "I know." heard as "I know." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v2/raw/short00_sad_s11.wav`)
- `short03_default_s11` check: word error rate 33%. "Come here, please." heard as "Come here, please, Ys." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v2/raw/short03_default_s11.wav`)
- `short11_sad_s11` check: 4.47 s, more than 2.5x the 1.16 s this text needs. "Nobody called back." heard as "Nobody called back." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v2/raw/short11_sad_s11.wav`)
- `long07_default_s11` incomplete: last 6 of 30 words missing. "We spent most of the summer by the lake, swimming in the mornings and reading on the porch in the afternoons, and nobody ever asked us to be anywhere else." heard as "We spent most of the summer by the lake, swimming in the mornings and reading on the porch in the afternoons, and no one" (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v2/raw/long07_default_s11.wav`)

## finetuned-emotion-v2

| length | style | clips | incomplete | to check |
|---|---|---|---|---|
| test | default | 16 | 0 | 0 |
| test | sad | 16 | 0 | 0 |
| new | default | 12 | 1 | 0 |
| new | sad | 12 | 0 | 0 |

**Incomplete overall: 1/56 (2%; 95% CI 0%-9%).**
- default: 1/28 (4%; 95% CI 1%-18%)
- sad: 0/28 (0%; 95% CI 0%-12%)

Flagged clips (listen to these):

- `new03_default_s11` incomplete: last 8 of 9 words missing. "Well, that is one way to look at it." heard as "Well," (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/finetuned-emotion-v2/raw/new03_default_s11.wav`)

## finetuned

| length | style | clips | incomplete | to check |
|---|---|---|---|---|
| test | default | 16 | 0 | 0 |
| test | sad | 16 | 0 | 0 |
| new | default | 12 | 0 | 0 |
| new | sad | 12 | 0 | 0 |

**Incomplete overall: 0/56 (0%; 95% CI 0%-6%).**
- default: 0/28 (0%; 95% CI 0%-12%)
- sad: 0/28 (0%; 95% CI 0%-12%)
