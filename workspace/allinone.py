#!/usr/bin/env python3
"""
allinone.py — FULL GEMINI ONLY
Gemini làm hết:
- Sinh truyện
- Sinh title
- Description
- Tags
- Thumbnail (ảnh base64)
- Tool render audio/video
"""

import os
import json
import base64
import requests
import datetime
import uuid
import random
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from moviepy.editor import ImageClip, AudioFileClip

try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except:
    COQUI_AVAILABLE = False


# PATHS
ROOT = Path.cwd()
WORKSPACE = ROOT / "workspace"
OUTPUT = WORKSPACE / "output"
ASSETS = ROOT / "assets"

OUTPUT.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Gemini API
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"  # đổi tùy mày


# -------------------------------------
# 🔥 Gemini — TEXT GEN (story + SEO)
# -------------------------------------
def gemini_text(prompt: str) -> str:
    url = "https://generativelanguage.googleapis.com/v1beta/models/" + GEMINI_MODEL + ":generateContent"
    r = requests.post(
        url,
        params={"key": GEMINI_KEY},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


# -------------------------------------
# 🔥 Gemini — IMAGE GEN (thumbnail)
# -------------------------------------
def gemini_image(prompt: str) -> bytes:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateImage"
    r = requests.post(
        url,
        params={"key": GEMINI_KEY},
        json={"prompt": {"text": prompt}},
        timeout=60,
    )
    r.raise_for_status()
    js = r.json()
    b64 = js["images"][0]["imageBytes"]
    return base64.b64decode(b64)


# -------------------------------------
# TTS fallback
# -------------------------------------
def make_silent(seconds: int, out: Path):
    sr = 22050
    data = np.zeros(int(sr * seconds), dtype=np.float32)
    sf.write(out, data, sr)


def coqui_tts(text: str, out: Path):
    if not COQUI_AVAILABLE:
        raise RuntimeError("Coqui not installed")
    tts = TTS("tts_models/en/vctk/vits")
    tts.tts_to_file(text=text, file_path=str(out))


# -------------------------------------
# MIX + RENDER
# -------------------------------------
def mix_voice_rain(voice: Path, rain: Path, duration: int, out: Path):
    v, sr1 = sf.read(str(voice), dtype="float32")
    r, sr2 = sf.read(str(rain), dtype="float32")

    if v.ndim > 1:
        v = v.mean(axis=1)
    if r.ndim > 1:
        r = r.mean(axis=1)

    sr = sr1
    need = duration * sr

    # loop rain
    if len(r) < need:
        rep = (need // len(r)) + 1
        r = np.tile(r, rep)
    r = r[:need]

    # pad voice
    if len(v) < need:
        pad = np.zeros(need - len(v), dtype=np.float32)
        v = np.concatenate([v, pad])
    else:
        v = v[:need]

    # mix
    mix = v * 1.0 + r * 0.25
    mx = np.max(np.abs(mix))
    if mx > 1:
        mix = mix / mx

    sf.write(out, mix, sr)


def render_video(audio_path: Path, image_path: Path, out_path: Path, duration: int):
    clip = ImageClip(str(image_path)).set_duration(duration)
    clip = clip.set_audio(AudioFileClip(str(audio_path)))
    clip.write_videofile(str(out_path), fps=1, codec="libx264", audio_codec="aac", threads=2,
                         verbose=False, logger=None)


# -------------------------------------
# MAIN
# -------------------------------------
def main():
    seed = "cozy rainy night"
    duration = 7200

    logging.info("🔥 Gemini generating story...")
    story = gemini_text(
        "Viết một câu chuyện dài, chill, kể chậm, nhẹ nhàng, phù hợp để nghe lúc ngủ. "
        "Chủ đề: " + seed +
        ". Viết rất dài, đọc hết sẽ gần 1 tiếng."
    )

    (OUTPUT / "story.txt").write_text(story, encoding="utf-8")

    logging.info("🔥 Gemini generating SEO...")
    seo = gemini_text(
        "Tạo SEO YouTube cho video kể chuyện nền tiếng mưa dựa trên nội dung sau:\n\n"
        + story[:2000] +
        "\n\nTrả về JSON: {title:'', description:'', tags:[]}"
    )
    try:
        js = json.loads(seo)
    except:
        js = {
            "title": "Relaxing Rainy Story",
            "description": "Rainy night story auto generated.",
            "tags": ["rain", "sleep", "asmr"]
        }

    (OUTPUT / "title.txt").write_text(js["title"], encoding="utf-8")
    (OUTPUT / "description.txt").write_text(js["description"], encoding="utf-8")
    (OUTPUT / "tags.txt").write_text(",".join(js.get("tags", [])), encoding="utf-8")

    logging.info("🔥 Gemini generating thumbnail...")
    img_bytes = gemini_image("cinematic cozy rainy night, warm lights, soft, anime style, no text")
    thumb = OUTPUT / "thumbnail.jpg"
    thumb.write_bytes(img_bytes)

    logging.info("🔥 TTS...")
    voice = OUTPUT / "voice.wav"
    try:
        coqui_tts(story, voice)
    except:
        make_silent(1800, voice)

    logging.info("🔥 Rain loop...")
    rain = ASSETS / "rain.mp3"
    rain_loop = OUTPUT / "rain_loop.wav"
    if rain.exists():
        r, sr = sf.read(str(rain), dtype="float32")
        total = duration * sr
        reps = int(total // len(r)) + 1
        full = np.tile(r, reps)[:total]
        sf.write(rain_loop, full, sr)
    else:
        make_silent(duration, rain_loop)

    logging.info("🔥 Mix...")
    final_audio = OUTPUT / "final_audio.wav"
    mix_voice_rain(voice, rain_loop, duration, final_audio)

    logging.info("🔥 Render video...")
    final_video = OUTPUT / "final_video.mp4"
    render_video(final_audio, thumb, final_video, duration)

    logging.info("🔥 DONE — final_video.mp4 created.")


if __name__ == "__main__":
    main()
