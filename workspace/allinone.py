#!/usr/bin/env python3
import os
import json
import random
import requests
from datetime import datetime
from moviepy.editor import AudioFileClip, ImageClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont

# === CONFIG ===
OUTPUT_DIR = "workspace/output"
TTS_MODEL = "tts_models/en/vctk/vits"
RAIN_SOUND = "workspace/assets/rain.mp3"
FONT_PATH = "workspace/assets/font.ttf"
DURATION_MINUTES = int(os.getenv("TARGET_DURATION", 10))

# === KEYS ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
YT_UPLOAD = os.getenv("YT_UPLOAD", "false").lower() == "true"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# === STORY GENERATION ===
def generate_story(seed):
    prompt = f"Write a calm, soothing bedtime story in English about: {seed}. Tone: peaceful, emotional, cinematic."
    print(f"🪶 Generating story with prompt: {prompt}")

    res = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8,
        },
    )

    try:
        data = res.json()
        if "choices" in data:
            story = data["choices"][0]["message"]["content"].strip()
        else:
            print("⚠️ API response error:", data)
            story = "Once upon a time, on a rainy evening, everything felt calm and warm..."
    except Exception as e:
        print("❌ Error parsing GPT response:", e)
        print("Response text:", res.text)
        story = "A soft rain fell outside as the city lights flickered gently..."

    return story


# === SEO GENERATION ===
def generate_seo(story):
    prompt = f"""
    Based on this chill bedtime story:
    ---
    {story[:800]}
    ---
    Write:
    1. A YouTube title (max 80 chars, highly clickable, chill tone)
    2. A YouTube description (2 paragraphs)
    3. 10 SEO tags separated by commas
    """

    try:
        res = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            },
        )
        data = res.json()
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        print("⚠️ SEO generation failed:", e)
        content = f"""
Title: Chill Rainy Night Story 🌧️ Peaceful Ambience
Description: A calm storytelling video with gentle rain sounds — perfect for sleep, relaxation, or study.
Tags: chill, rain sounds, relaxation, sleep, ambient, storytelling, calm music, ASMR, cozy night, peaceful vibes
        """

    lines = content.strip().split("\n")
    title = lines[0].replace("Title:", "").strip()
    description = "\n".join(lines[1:]).replace("Description:", "").strip()
    tags = ",".join([x.strip() for x in description.split(",")[-10:]])

    return title, description, tags


# === THUMBNAIL AI ===
def generate_thumbnail(title):
    print(f"🖼️ Generating AI thumbnail for: {title}")
    prompt = f"cinematic cozy rainy night scene with soft lighting, for a chill story titled: {title}"

    try:
        img_url = "https://image.pollinations.ai/prompt/" + requests.utils.quote(prompt)
        thumb_path = os.path.join(OUTPUT_DIR, "thumbnail.jpg")
        img_data = requests.get(img_url).content
        with open(thumb_path, "wb") as f:
            f.write(img_data)
        return thumb_path
    except Exception as e:
        print("⚠️ Thumbnail generation failed:", e)
        fallback = Image.new("RGB", (1280, 720), color=(30, 30, 30))
        draw = ImageDraw.Draw(fallback)
        draw.text((50, 300), title, fill="white", font=ImageFont.truetype(FONT_PATH, 48))
        fallback.save(os.path.join(OUTPUT_DIR, "thumbnail.jpg"))
        return os.path.join(OUTPUT_DIR, "thumbnail.jpg")


# === AUDIO SYNTHESIS ===
def synthesize_audio(text, out_path="voice.wav"):
    pri
