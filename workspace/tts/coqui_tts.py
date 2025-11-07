#!/usr/bin/env python3
# Coqui TTS script: read text from a file and synthesize to WAV
# Usage: python3 tts/coqui_tts.py input_text.txt output_voice.wav [voice_id]

import sys, os
from TTS.api import TTS

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 tts/coqui_tts.py input_text.txt output_voice.wav [voice_id]")
        sys.exit(1)

    in_txt = sys.argv[1]
    out_wav = sys.argv[2]

    if not os.path.exists(in_txt):
        raise FileNotFoundError(in_txt)

    with open(in_txt, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # English glow-tts single speaker model
    tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False, gpu=False)
    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)
    tts.tts_to_file(text=text, file_path=out_wav)
    print(f"[OK] Synthesized voice -> {os.path.abspath(out_wav)}")

if __name__ == "__main__":
    main()