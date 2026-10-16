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

## robustness-finetuned-emotion-v4

| length | style | clips | incomplete | to check |
|---|---|---|---|---|
| short | default | 20 | 0 | 0 |
| short | enunciated | 20 | 0 | 0 |
| short | sad | 20 | 0 | 2 |
| short | whisper | 20 | 0 | 1 |
| medium | default | 20 | 0 | 0 |
| medium | enunciated | 20 | 0 | 0 |
| medium | sad | 20 | 0 | 0 |
| medium | whisper | 20 | 0 | 2 |
| long | default | 20 | 0 | 0 |
| long | enunciated | 20 | 0 | 0 |
| long | sad | 20 | 0 | 0 |
| long | whisper | 20 | 3 | 1 |

**Incomplete overall: 3/240 (1%; 95% CI 0%-4%).**
- default: 0/60 (0%; 95% CI 0%-6%)
- enunciated: 0/60 (0%; 95% CI 0%-6%)
- sad: 0/60 (0%; 95% CI 0%-6%)
- whisper: 3/60 (5%; 95% CI 2%-14%)

Flagged clips (listen to these):

- `short02_sad_s11` check: word error rate 33%. "Maybe next week." heard as "Maybe be next week." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v4/raw/short02_sad_s11.wav`)
- `short10_whisper_s11` check: word error rate 33%. "He left early." heard as "You left early." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v4/raw/short10_whisper_s11.wav`)
- `short15_sad_s11` check: word error rate 50%. "The lights went out." heard as "The Lice One Out!" (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v4/raw/short15_sad_s11.wav`)
- `medium12_whisper_s11` check: word error rate 70%. "The garden looks different now that the trees are gone." heard as "The garden looks different now than the trees are gar-gar-gar-gar. Perfect cake." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v4/raw/medium12_whisper_s11.wav`)
- `medium19_whisper_s11` check: word error rate 27%. "Everyone agreed to meet again at the end of the month." heard as "Everyone agreed to meet again at the end of the end of the month." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v4/raw/medium19_whisper_s11.wav`)
- `long01_whisper_s11` incomplete: last 2 of 29 words missing. "The museum was almost empty that afternoon, so we walked slowly through every room, reading each small card beside the paintings until the guard told us they were closing." heard as "The moon was almost empty that afternoon, so we walked slowly through every room, reading each small card beside the painting. Until the card told us, see?" (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v4/raw/long01_whisper_s11.wav`)
- `long03_whisper_s11` check: word error rate 31%. "He packed the last of the boxes into the car, took one more look at the empty rooms, and then drove away without telling anyone where he was going." heard as "He packed the last of the boxes into the car, took one more or more look at the empty rooms, and then drove away without fending. Tell me now, why are you?" (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v4/raw/long03_whisper_s11.wav`)
- `long04_whisper_s11` incomplete: last 3 of 30 words missing. "Oh, I almost forgot to mention that the meeting has been moved to Thursday, and the room on the second floor will not be available until later in the day." heard as "Oh, I almost forgot to mention that the meeting has been moved to Thursday, and the room on the second floor will not be a vividly feeling." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v4/raw/long04_whisper_s11.wav`)
- `long11_whisper_s11` incomplete: last 3 of 30 words missing. "The small bakery on the corner has been closed for years, but every time I walk past it, I can still remember the smell of fresh bread in the morning." heard as "The small bakery on the corner has been closed for years, but every time I walk past it, I can still remember the smell of fresh bread." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v4/raw/long11_whisper_s11.wav`)
