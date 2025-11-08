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
ASSETS_DIR = "workspace/assets"
RAIN_SOUND = os.path.join(ASSETS_DIR, "rain.mp3")
FONT_PATH = os.path.join(ASSETS_DIR, "font.ttf")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

DURATION_MINUTES = int(os.getenv("TARGET_DURATION", 10))
DURATION_SECONDS = DURATION_MINUTES * 60
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
YT_UPLOAD = os.getenv("YT_UPLOAD", "false").lower() == "true"

# === AUTO DOWNLOAD RAIN SOUND ===
if not os.path.exists(RAIN_SOUND) or os.path.getsize(RAIN_SOUND) < 1000:
    print("🌧️ Rain sound not found, downloading fallback...")
    try:
        os.system(
            "curl -L -o workspace/assets/rain.mp3 "
            "https://huggingface.co/datasets/hauntedai/audio-sfx/resolve/main/rain_soft.mp3?download=true"
        )
        print("✅ Rain sound downloaded successfully.")
    except Exception as e:
        print("⚠️ Failed to download rain.mp3:", e)


# === STORY GENERATION ===
def generate_story(seed):
    prompt = f"Write a calm, soothing bedtime story in English about: {seed}. Tone: peaceful, emotional, cinematic."
    print(f"🪶 Generating story with prompt: {prompt}")

    try:
        res = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
            },
        )
        data = res.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"].strip()
        else:
            print("⚠️ GPT returned:", data)
            return "Once upon a calm night, gentle rain whispered outside the window..."
    except Exception as e:
        print("❌ GPT Error:", e)
        return "A soft rain fell outside as the city lights flickered gently..."


# === SEO + THUMBNAIL ===
def generate_seo(story):
    prompt = f"""
    Based on this chill bedtime story:
    ---
    {story[:800]}
    ---
    Write:
    1. YouTube title (max 80 chars)
    2. Description (2 short paragraphs)
    3. 10 SEO tags separated by commas
    """
    try:
        res = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            },
        )
        data = res.json()
        text = data["choices"][0]["message"]["content"]
    except:
        text = """Title: Rainy Night Story
Description: Calm storytelling with gentle rain — perfect for relaxation or sleep.
Tags: chill, rain, relax, sleep, study, calm, asmr, peaceful, bedtime, ambience"""

    lines = text.split("\n")
    title = lines[0].replace("Title:", "").strip()
    desc = "\n".join(lines[1:]).replace("Description:", "").strip()
    tags = ",".join([x.strip() for x in desc.split(",")[-10:]])
    return title, desc, tags


def generate_thumbnail(title, thumb_path):
    print(f"🖼️ Generating thumbnail for: {title}")
    prompt = f"cinematic cozy rainy night scene, soft lighting, titled: {title}"
    try:
        url = "https://image.pollinations.ai/prompt/" + requests.utils.quote(prompt)
        img = requests.get(url).content
        with open(thumb_path, "wb") as f:
            f.write(img)
        return thumb_path
    except:
        thumb = Image.new("RGB", (1280, 720), (20, 20, 20))
        draw = ImageDraw.Draw(thumb)
        draw.text((50, 350), title, fill="white", font=ImageFont.truetype(FONT_PATH, 50))
        thumb.save(thumb_path)
        return thumb_path


# === AUDIO (placeholder) ===
def synthesize_audio(story_text, out_path):
    print("🎙️ Creating fake audio track (placeholder)...")
    os.system(f"cp {RAIN_SOUND} {out_path}")
    return out_path


# === VIDEO RENDER ===
def render_video(seed):
    print(f"🎬 Rendering video for seed: {seed}")
    story = generate_story(seed)
    title, desc, tags = generate_seo(story)

    safe_name = seed.replace(" ", "_").replace(",", "_")
    story_audio = os.path.join(OUTPUT_DIR, f"{safe_name}_voice.wav")
    thumb_path = os.path.join(OUTPUT_DIR, f"{safe_name}_thumb.jpg")
    output_path = os.path.join(OUTPUT_DIR, f"{safe_name}.mp4")

    synthesize_audio(story, story_audio)
    generate_thumbnail(title, thumb_path)

    try:
        # Load ảnh thumbnail và tiếng mưa
        img_clip = ImageClip(thumb_path).set_duration(DURATION_SECONDS)
        voice_clip = AudioFileClip(story_audio).volumex(0.6)
        rain_clip = AudioFileClip(RAIN_SOUND).volumex(0.3)

        # Kết hợp âm thanh
        final_audio = voice_clip.set_duration(DURATION_SECONDS)
        img_clip = img_clip.set_audio(final_audio)

        # Xuất video MP4
        print(f"💾 Saving video to {output_path} ...")
        img_clip.write_videofile(
            output_path,
            fps=30,
            codec="libx264",
            audio_codec="aac",
            bitrate="2000k",
            threads=2
        )

        print(f"✅ Rendered: {output_path}")
        return output_path
    except Exception as e:
        print("❌ Render failed:", e)
        return None


# === MAIN ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, required=False, default="rainy night story")
    args = parser.parse_args()

    try:
        render_video(args.seed)
    except Exception as e:
        print("❌ Critical error:", e)

    print("🎉 Done.")
