# Early stops and other incomplete speech

Rule: incomplete if Whisper's transcript lacks the last 2+ words (or the last third) of the text, the clip ends mid-sound (last 100 ms less than 15 dB below its speech level), or it is under 0.5x the time ex02 needs for the text (characters / ex02's speaking pace for that style). 'Check' = complete but over 2.5x that time, or a word error rate of 25% or more.

**The rule on 228 real held-out recordings:** 0/228 (0%; 95% CI 0%-2%) flagged incomplete, 9 flagged to check.

- real `ex02_default_00087` (check: word error rate 29%): "Me in a straitjacket, heading for Ravenwood." heard as "Me in a straight jacket, heading for Ravenwood."
- real `ex02_default_00088` (check: word error rate 25%): "How you holding up?" heard as "How are you holding up?"
- real `ex02_default_00147` (check: word error rate 29%): "Jacks's always pay off their debts." heard as "JAXs always pay off their debts."
- real `ex02_sad_00087` (check: word error rate 29%): "Me in a straitjacket, heading for Ravenwood." heard as ""'Me in a straight jacket, heading for Ravenwood.'"
- real `ex02_sad_00147` (check: word error rate 29%): "Jacks's always pay off their debts." heard as "Jackses always pay off their debts."
- real `ex02_sad_00260` (check: word error rate 33%): "Okay, alarm set for this Saturday at six forty five AM." heard as "Okay. Alarm set for this Saturday at 6.45 a.m."
- real `ex02_whisper_00147` (check: word error rate 29%): "Jacks's always pay off their debts." heard as "JAXs always pay off their debts."
- real `ex02_whisper_00112` (check: word error rate 25%): "What's good, Bev?" heard as "What's good, Beth?"
- real `ex02_whisper_00252` (check: word error rate 29%): "Hang gliding and bungee jumping sound scary." heard as "Hand gliding and bungee jumping sounds scary."

## finetuned-emotion-v3

| length | style | clips | incomplete | to check |
|---|---|---|---|---|
| new | default | 12 | 0 | 0 |
| new | sad | 12 | 0 | 0 |
| new | whisper | 12 | 1 | 0 |
| test | default | 16 | 0 | 0 |
| test | sad | 16 | 0 | 0 |
| test | whisper | 16 | 0 | 2 |

**Incomplete overall: 1/84 (1%; 95% CI 0%-6%).**
- default: 0/28 (0%; 95% CI 0%-12%)
- sad: 0/28 (0%; 95% CI 0%-12%)
- whisper: 1/28 (4%; 95% CI 1%-18%)

Flagged clips (listen to these):

- `test03_whisper_s22` check: word error rate 40%. "Is he masquerading as a butler in the main household?" heard as "Is he masquerading? He is a battler in the main scene." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/finetuned-emotion-v3/raw/test03_whisper_s22.wav`)
- `test06_whisper_s22` check: word error rate 25%. "Tugging at his collar, Tim was noticeably warm." heard as "Tucking at his collar, Tim was noticeably worn." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/finetuned-emotion-v3/raw/test06_whisper_s22.wav`)
- `new00_whisper_s22` incomplete: last 2 of 9 words missing. "I did not expect to see you here today." heard as "I did not expect to see you." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/finetuned-emotion-v3/raw/new00_whisper_s22.wav`)
