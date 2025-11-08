#!/usr/bin/env python3
import os, random, requests
from datetime import datetime
from moviepy.editor import AudioFileClip, ImageClip, CompositeVideoClip
from TTS.api import TTS
from PIL import Image, ImageDraw, ImageFont

# === CONFIG ===
OUTPUT_DIR = "workspace/output"
FONT_PATH = "workspace/assets/font.ttf"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DURATION_SECONDS = int(os.getenv("TARGET_DURATION", "7200"))

# === STORY GENERATION ===
def generate_story(seed):
    prompt = f"Write a calm, cinematic, relaxing English bedtime story about {seed}. Tone: soft, cozy, emotional."
    print("🪶 Generating story via OpenAI...")
    res = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8,
        },
    )
    try:
        data = res.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("⚠️ GPT fallback text used:", e)
        return "On a quiet rainy night, the world outside was calm..."

# === TTS LOCAL (Coqui offline, free) ===
def synthesize_audio(text, out_path="workspace/output/voice.wav"):
    try:
        print("🎙️ Generating voice with Coqui TTS (offline)...")
        tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)

        # Chọn giọng mặc định (bắt buộc với multi-speaker model)
        speaker = random.choice(["p315", "p270", "p233", "p340", "p362"])
        print(f"🗣️ Using speaker voice: {speaker}")

        tts.tts_to_file(text=text, file_path=out_path, speaker=speaker)
        return out_path
    except Exception as e:
        print("⚠️ Local TTS failed:", e)
        return None

# === THUMBNAIL AI ===
def generate_thumbnail(title):
    try:
        prompt = f"cinematic cozy rainy night scene, title: {title}, soft lighting, 4k, realistic"
        img_url = "https://image.pollinations.ai/prompt/" + requests.utils.quote(prompt)
        img_data = requests.get(img_url).content
        thumb_path = os.path.join(OUTPUT_DIR, "thumbnail.jpg")
        with open(thumb_path, "wb") as f:
            f.write(img_data)
        return thumb_path
    except Exception as e:
        print("⚠️ Thumbnail generation failed:", e)
        img = Image.new("RGB", (1280, 720), (20, 20, 20))
        draw = ImageDraw.Draw(img)
        draw.text((50, 300), title, fill="white", font=ImageFont.truetype(FONT_PATH, 48))
        thumb_path = os.path.join(OUTPUT_DIR, "thumbnail.jpg")
        img.save(thumb_path)
        return thumb_path

# === RENDER VIDEO ===
def render_video(audio_path, thumb_path, out_path="workspace/output/final_video.mp4"):
    print("🎬 Rendering video (2h, 720p, 24fps)...")
    bg = ImageClip(thumb_path).resize(height=720)
    audio = AudioFileClip(audio_path)
    video = bg.set_duration(audio.duration).set_audio(audio)
    video.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac", bitrate="2000k")
    print(f"✅ Render complete: {out_path}")
    return out_path

# === MAIN ===
if __name__ == "__main__":
    seed = "A cozy night by the lake with rain sounds"
    story = generate_story(seed)
    title = "Relaxing Rainy Story 🌧️ " + seed
    thumb = generate_thumbnail(title)
    voice = synthesize_audio(story)
    if not voice:
        print("⚠️ Falling back to ambient rain sound...")
        voice = "workspace/assets/rain.mp3"
    render_video(voice, thumb)
