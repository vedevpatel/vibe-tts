# Early stops and other incomplete speech

Rule: incomplete if Whisper's transcript lacks the last 2+ words (or the last third) of the text, the clip ends mid-sound (last 100 ms less than 15 dB below its speech level), or it is under 0.5x the time ex02 needs for the text (characters / ex02's speaking pace for that style). 'Check' = complete but over 2.5x that time, or a word error rate of 25% or more.

**The rule on 304 real held-out recordings:** 0/304 (0%; 95% CI 0%-1%) flagged incomplete, 13 flagged to check.

- real `ex02_default_00087` (check: word error rate 29%): "Me in a straitjacket, heading for Ravenwood." heard as "Me in a straight jacket, heading for Ravenwood."
- real `ex02_default_00088` (check: word error rate 25%): "How you holding up?" heard as "How are you holding up?"
- real `ex02_default_00147` (check: word error rate 29%): "Jacks's always pay off their debts." heard as "JAXs always pay off their debts."
- real `ex02_enunciated_00087` (check: word error rate 29%): "Me in a straitjacket, heading for Ravenwood." heard as "Me, in a straight jacket, heading for Ravenwood."
- real `ex02_enunciated_00088` (check: word error rate 25%): "How you holding up?" heard as "How are you holding up?"
- real `ex02_enunciated_00147` (check: word error rate 29%): "Jacks's always pay off their debts." heard as "Jaxes always pay off their debts."
- real `ex02_enunciated_00260` (check: word error rate 33%): "Okay, alarm set for this Saturday at six forty five AM." heard as "Okay, alarm set for this Saturday at 6.45 a.m."
- real `ex02_sad_00087` (check: word error rate 29%): "Me in a straitjacket, heading for Ravenwood." heard as ""'Me in a straight jacket, heading for Ravenwood.'"
- real `ex02_sad_00147` (check: word error rate 29%): "Jacks's always pay off their debts." heard as "Jackses always pay off their debts."
- real `ex02_sad_00260` (check: word error rate 33%): "Okay, alarm set for this Saturday at six forty five AM." heard as "Okay. Alarm set for this Saturday at 6.45 a.m."
- real `ex02_whisper_00147` (check: word error rate 29%): "Jacks's always pay off their debts." heard as "JAXs always pay off their debts."
- real `ex02_whisper_00112` (check: word error rate 25%): "What's good, Bev?" heard as "What's good, Beth?"
- real `ex02_whisper_00252` (check: word error rate 29%): "Hang gliding and bungee jumping sound scary." heard as "Hand gliding and bungee jumping sounds scary."

## finetuned-emotion-v4

| length | style | clips | incomplete | to check |
|---|---|---|---|---|
| new | default | 12 | 0 | 0 |
| new | enunciated | 12 | 0 | 0 |
| new | sad | 12 | 0 | 0 |
| new | whisper | 12 | 0 | 2 |
| test | default | 16 | 0 | 0 |
| test | enunciated | 16 | 0 | 0 |
| test | sad | 16 | 0 | 0 |
| test | whisper | 16 | 0 | 2 |

**Incomplete overall: 0/112 (0%; 95% CI 0%-3%).**
- default: 0/28 (0%; 95% CI 0%-12%)
- enunciated: 0/28 (0%; 95% CI 0%-12%)
- sad: 0/28 (0%; 95% CI 0%-12%)
- whisper: 0/28 (0%; 95% CI 0%-12%)

Flagged clips (listen to these):

- `test06_whisper_s11` check: word error rate 25%. "Tugging at his collar, Tim was noticeably warm." heard as "Tucking at his collar, Tim was noticeably warm. Warm." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/finetuned-emotion-v4/raw/test06_whisper_s11.wav`)
- `test06_whisper_s22` check: word error rate 88%. "Tugging at his collar, Tim was noticeably warm." heard as "Tugging at his collar, Tim was noticed there was no tape for him." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/finetuned-emotion-v4/raw/test06_whisper_s22.wav`)
- `new04_whisper_s22` check: word error rate 25%. "We will find out what happened soon enough." heard as "We will find out what happens soon." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/finetuned-emotion-v4/raw/new04_whisper_s22.wav`)
- `new05_whisper_s22` check: word error rate 89%. "He read the letter twice and put it away." heard as "He read the letter twice and put it in the coat. Put it in the coat." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/finetuned-emotion-v4/raw/new05_whisper_s22.wav`)
