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
    du
