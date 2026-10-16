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

## robustness-finetuned-emotion-v3

| length | style | clips | incomplete | to check |
|---|---|---|---|---|
| short | default | 20 | 0 | 1 |
| short | sad | 20 | 0 | 2 |
| short | whisper | 20 | 0 | 0 |
| medium | default | 20 | 0 | 0 |
| medium | sad | 20 | 0 | 0 |
| medium | whisper | 20 | 0 | 2 |
| long | default | 20 | 0 | 0 |
| long | sad | 20 | 0 | 0 |
| long | whisper | 20 | 8 | 4 |

**Incomplete overall: 8/180 (4%; 95% CI 2%-9%).**
- default: 0/60 (0%; 95% CI 0%-6%)
- sad: 0/60 (0%; 95% CI 0%-6%)
- whisper: 8/60 (13%; 95% CI 7%-24%)

Flagged clips (listen to these):

- `short03_default_s11` check: word error rate 33%. "Come here, please." heard as "Come here, fleece." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v3/raw/short03_default_s11.wav`)
- `short06_sad_s11` check: word error rate 33%. "Oh, I see." heard as "All I see..." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v3/raw/short06_sad_s11.wav`)
- `short13_sad_s11` check: word error rate 33%. "Is this yours?" heard as "Is th-this yours?" (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v3/raw/short13_sad_s11.wav`)
- `medium11_whisper_s11` check: word error rate 30%. "Please make sure the door is locked before you leave." heard as "Please make sure the door is locked. Pew pew." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v3/raw/medium11_whisper_s11.wav`)
- `medium19_whisper_s11` check: word error rate 45%. "Everyone agreed to meet again at the end of the month." heard as "The Ephraim agreed to meet again at the end of the end of the month." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v3/raw/medium19_whisper_s11.wav`)
- `long00_whisper_s11` incomplete: last 3 of 28 words missing. "Well, after everything that happened last winter, I never thought we would be sitting here together again, talking about the old house and the people who lived there." heard as "Well, after everything that happened last winter, I never thought we would be sitting here together again, talking about the old house and the people." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v3/raw/long00_whisper_s11.wav`)
- `long01_whisper_s11` incomplete: last 8 of 29 words missing. "The museum was almost empty that afternoon, so we walked slowly through every room, reading each small card beside the paintings until the guard told us they were closing." heard as "The museum was almost empty that afternoon, so we walked slowly through every room reading each small card beside the paintings." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v3/raw/long01_whisper_s11.wav`)
- `long02_whisper_s11` incomplete: last 2 of 28 words missing. "When the phone finally rang, she let it ring three more times before answering, because she already knew who was calling and what they were going to say." heard as "When the phone finally rang, she let it ring three more times before answering because she already knew who was she offering? P-I-O Medicaid." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v3/raw/long02_whisper_s11.wav`)
- `long04_whisper_s11` incomplete: last 7 of 30 words missing. "Oh, I almost forgot to mention that the meeting has been moved to Thursday, and the room on the second floor will not be available until later in the day." heard as "Oh, I almost forgot to mention that the meeting has been moved to Thursday. And the room on the second floor will not..." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v3/raw/long04_whisper_s11.wav`)
- `long05_whisper_s11` incomplete: last 3 of 31 words missing. "So, if you follow the river for about a mile and then turn left at the old stone bridge, you will see the cottage just behind the row of tall trees." heard as "So if you follow the river for about a mile and then turn left at the old stone bridge, you will see, you will see the bottom." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v3/raw/long05_whisper_s11.wav`)
- `long06_whisper_s11` incomplete: last 2 of 30 words missing. "The letters were tied together with a faded blue ribbon, and some of them had been read so many times that the paper had started to tear along the folds." heard as "The letters were tied together with a faded blue ribbon, and some of them had been read so many times that the paper had started to tear along." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v3/raw/long06_whisper_s11.wav`)
- `long08_whisper_s11` check: word error rate 43%. "I told them that the instructions were clear enough, but somehow the whole thing still ended up in pieces on the floor of the living room by midnight." heard as "I told them that the instructions were clear enough, but somehow the whole thing still ended up in pieces on the floor of the left-the left-the left-the left-the left-the left-the" (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v3/raw/long08_whisper_s11.wav`)
- `long09_whisper_s11` check: word error rate 29%. "After the storm passed, the streets were covered with branches and leaves, and the neighbors came outside one by one to see how much damage had been done." heard as "After the storm passed, the streets were covered with branches and leaves, and the neighbors came outside one by by one to see how long to stay out on Jeffing. Damn." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v3/raw/long09_whisper_s11.wav`)
- `long12_whisper_s11` incomplete: last 7 of 27 words missing. "She kept every ticket from every concert she had ever attended, and the shoebox under her bed was so full that the lid no longer closed properly." heard as "She kept every ticket from every concert she'd have ever attended, and the shoebox under her bed was so full." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v3/raw/long12_whisper_s11.wav`)
- `long15_whisper_s11` incomplete: last 6 of 23 words missing. "Nobody could explain why the clock in the hallway had stopped at exactly the same minute every night for the past three weeks." heard as "Nobody could explain why the clock in the hallway had stopped at exactly the same noon means." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v3/raw/long15_whisper_s11.wav`)
- `long16_whisper_s11` check: word error rate 31%. "He spoke very quietly, as if he was afraid that someone in the next room might hear, and then he handed me an envelope with my name on it." heard as "He poked verily, as if he was afraid that someone in the next room might hear, and then he handed me an M-m-m with my main name." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v3/raw/long16_whisper_s11.wav`)
- `long17_whisper_s11` check: word error rate 37%. "Oh, and before you go, could you please take these books back to the library, because they were due last Friday and I keep forgetting about them." heard as "Oh, and before you go, could you please, take you please, please take these books back to the library because they would keep, we had to keep forgetting about them." (`/Users/vedpatel/vibe-tts/outputs/expresso/runs/robustness-finetuned-emotion-v3/raw/long17_whisper_s11.wav`)
