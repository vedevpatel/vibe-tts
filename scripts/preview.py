import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COSYVOICE_DIR = PROJECT_ROOT / "CosyVoice"
MODEL_DIR = (
    PROJECT_ROOT
    / "pretrained_models"
    / "Fun-CosyVoice3-0.5B-2512"
)
PROMPT_WAV = COSYVOICE_DIR / "asset" / "zero_shot_prompt.wav"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Upstream imports (cosyvoice, Matcha-TTS) live inside the cloned repo.
sys.path.insert(0, str(COSYVOICE_DIR))
sys.path.insert(0, str(COSYVOICE_DIR / "third_party" / "Matcha-TTS"))

import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    cosyvoice = AutoModel(model_dir=str(MODEL_DIR))

    for i, j in enumerate(cosyvoice.inference_zero_shot(
        "Testing the zero-shot TTS functionality.",
        "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
        str(PROMPT_WAV),
        stream=False,
    )):
        out = OUTPUT_DIR / f"zero_shot_{i}.wav"
        torchaudio.save(str(out), j["tts_speech"], cosyvoice.sample_rate)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
