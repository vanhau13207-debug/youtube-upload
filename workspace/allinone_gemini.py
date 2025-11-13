#!/usr/bin/env python3
import os
import random
import base64
import requests
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
from TTS.api import TTS

# ======================== CONFIG =========================
OUTPUT_DIR = "workspace/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
FONT_PATH = "workspace/assets/font.ttf"

RAIN_BG = "workspace/assets/rain_bg.mp4"   # video nền động
RAIN_SOUND = "workspace/assets/rain_bg.mp4"  # dùng audio của video nền luôn

TARGET_DURATION = int(os.getenv("TARGET_DURATION", "7200"))  # default 2h

# =========================================================


# 1️⃣ SINH TRUYỆN BẰNG GEMINI
def generate_story(seed):
    prompt = f"""
Write a long, cinematic English bedtime story (30 paragraphs) about:
{seed}

The tone should be emotional, cozy, rainy-night, peaceful.
    """

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}"
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:
        res = requests.post(url, json=payload)
        data = res.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text.strip()
    except Exception as e:
        print("⚠️ Gemini story failed:", e)
        return "On a rainy night, the world softened into quiet reflections..."


# 2️⃣ SEO: TITLE + DESCRIPTION + TAGS
def generate_seo(story):
    prompt = f"""
Generate:
1. YouTube title (max 80 chars, emotional, rain, cozy, cinematic)
2. 2-paragraph description
3. 12 SEO tags separated by commas

Story:
{story[:1200]}
"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        res = requests.post(url, json=payload)
        content = res.json()["candidates"][0]["content"]["parts"][0]["text"]
    except:
        print("⚠️ SEO fallback")
        return (
            "Rainy Night Story 🌧️ Cozy Ambience for Sleep",
            "Relaxing story with rain ambience.",
            "rain,relaxing,sleep,story,rain sounds,chill,night,cozy,asmr,ambience"
        )

    lines = content.split("\n")
    title = lines[0].replace("Title:", "").strip()
    description = "\n".join(lines[1:]).replace("Description:", "").strip()

    tags = ",".join([x.strip() for x in description.split(",")[-12:]])

    return title, description, tags


# 3️⃣ TRÍCH FRAME → GIAO CHO GEMINI FLASH TẠO THUMBNAIL
def generate_thumbnail(title, story):
    frame_path = os.path.join(OUTPUT_DIR, "frame.jpg")
    thumb_path = os.path.join(OUTPUT_DIR, "thumbnail.jpg")

    # lấy frame ngẫu nhiên (giữa video)
    ts = random.randint(20, 150)
    os.system(f"ffmpeg -ss {ts} -i {RAIN_BG} -frames:v 1 {frame_path} -y")

    with open(frame_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text":
f"""
Enhance this frame into a cinematic YouTube thumbnail.
Mood: cozy rainy night, emotional, soft lighting.
Include depth, contrast, slight glow, FILM look.
Story title: {title}
"""
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_b64
                        }
                    }
                ]
            }
        ]
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}"

    try:
        res = requests.post(url, json=payload)
        data = res.json()["candidates"][0]["content"]["parts"][0]["inline_data"]["data"]
        img_bytes = base64.b64decode(data)
        with open(thumb_path, "wb") as f:
            f.write(img_bytes)
        return thumb_path

    except Exception as e:
        print("⚠️ Gemini thumbnail fail:", e)
        return frame_path


# 4️⃣ COQUI TTS → GIỌNG DÀI 2 TIẾNG
def generate_voice(text):
    print("🎙️ Synthesizing with Coqui TTS...")
    tts = TTS(model_name="tts_models/en/vctk/vits", gpu=False)
    out_path = os.path.join(OUTPUT_DIR, "voice.wav")
    tts.tts_to_file(text=text, file_path=out_path, speaker="p315")
    return out_path


# 5️⃣ RENDER VIDEO DÀI 2H (NỀN MƯA + TTS)
def render_video(voice_path, thumb_path):
    print("🎬 Rendering full HD 2-hour video...")

    # nền động (mưa)
    bg = VideoFileClip(RAIN_BG).loop(duration=TARGET_DURATION).resize((1280, 720))

    audio = AudioFileClip(voice_path)

    final = bg.set_audio(audio)
    out_path = os.path.join(OUTPUT_DIR, "final_video.mp4")

    final.write_videofile(
        out_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        bitrate="4000k"
    )

    return out_path


# ================== MAIN ==========================
if __name__ == "__main__":
    seed = "A cozy rainy night in a lakeside cabin"

    print("📘 Generating story...")
    story = generate_story(seed)

    print("🔍 Generating SEO...")
    title, description, tags = generate_seo(story)

    print("🖼️ Creating AI thumbnail...")
    thumb = generate_thumbnail(title, story)

    print("🎙️ Generating voice...")
    voice = generate_voice(story)

    print("🎬 Rendering final video...")
    video = render_video(voice, thumb)

    # Lưu SEO vào file
    with open(os.path.join(OUTPUT_DIR, "seo.txt"), "w") as f:
        f.write("TITLE:\n" + title + "\n\nDESCRIPTION:\n" + description + "\n\nTAGS:\n" + tags)

    print("✅ DONE: Video + Thumbnail + SEO ready.")
