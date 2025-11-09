#!/usr/bin/env python3
import os, requests, shutil
from moviepy.editor import *
from TTS.api import TTS
from PIL import Image, ImageDraw, ImageFont

# === CONFIG ===
OUTPUT_DIR = "workspace/output"
ASSETS_DIR = "workspace/assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FONT_PATH = os.path.join(ASSETS_DIR, "font.ttf")
RAIN_SOUND = os.path.join(ASSETS_DIR, "rain.mp3")
RAIN_VIDEO = os.path.join(ASSETS_DIR, "rain_bg.mp4")
TARGET_DURATION = int(os.getenv("TARGET_DURATION", "7200"))

# === STORY GENERATION (Gemini API miễn phí) ===
def generate_story(seed):
    print(f"🪶 Generating story using Gemini for seed: {seed}")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Missing GEMINI_API_KEY. Please add it in GitHub Secrets.")
        return "On a quiet rainy night, soft raindrops touched the window, and peace filled the air..."

    prompt = f"""
    Write a calm, emotional, and cinematic English bedtime story about {seed}.
    Make it relaxing, detailed, and soothing — perfect for listening while it rains.
    """

    try:
        res = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30
        )

        data = res.json()
        story = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return story
    except Exception as e:
        print("⚠️ Gemini API failed:", e)
        return "The rain whispered softly against the glass, carrying memories of warmth and calm nights..."


# === TTS ===
def synthesize_audio(text, out_path):
    print("🎙️ Generating voice with Coqui TTS...")
    try:
        tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)
        tts.tts_to_file(text=text, file_path=out_path, speaker="p273")
    except Exception as e:
        print("⚠️ TTS failed:", e)
        shutil.copy(RAIN_SOUND, out_path)
    return out_path

# === THUMBNAIL ===
def generate_thumbnail(title):
    try:
        prompt = f"cinematic cozy rainy night scene, title: {title}, soft lighting, 4k"
        img_url = "https://image.pollinations.ai/prompt/" + requests.utils.quote(prompt)
        img_data = requests.get(img_url).content
        thumb_path = os.path.join(OUTPUT_DIR, "thumbnail.jpg")
        with open(thumb_path, "wb") as f:
            f.write(img_data)

        img = Image.open(thumb_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(FONT_PATH, 64)
        draw.text((50, 620), title, fill="white", font=font)
        img.save(thumb_path)
        return thumb_path
    except Exception as e:
        print("⚠️ Thumbnail fallback:", e)
        return None

# === AUDIO MIX ===
def mix_audio(voice_path, rain_path, out_path):
    print("🎧 Mixing voice + rain sound...")
    os.system(
        f'ffmpeg -y -i "{voice_path}" -i "{rain_path}" '
        f'-filter_complex "[1:a]volume=0.25[a1];[0:a][a1]amix=inputs=2:duration=first" "{out_path}"'
    )
    return out_path

# === RENDER VIDEO ===
def render_video(audio_path, bg_video_path, thumb_path, out_path):
    print("🎬 Rendering 2-hour chill video...")
    os.system(f'ffmpeg -y -stream_loop -1 -i "{bg_video_path}" -t {TARGET_DURATION} -c copy "{OUTPUT_DIR}/temp_bg.mp4"')
    bg_video_path = f"{OUTPUT_DIR}/temp_bg.mp4"
    audio = AudioFileClip(audio_path)
    bg = VideoFileClip(bg_video_path).loop(duration=audio.duration)
    final = bg.set_audio(audio).resize(height=720)
    final.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac", bitrate="2500k")
    print(f"✅ Render complete: {out_path}")
    return out_path

if __name__ == "__main__":
    seed = "A cozy night by the lake with rain sounds"
    title = f"🌧️ {seed.title()} | 2H Chill Story for Sleep & Relaxation"
    story = generate_story(seed)
    voice_path = os.path.join(OUTPUT_DIR, "voice.wav")
    mixed_audio = os.path.join(OUTPUT_DIR, "mixed.wav")
    final_path = os.path.join(OUTPUT_DIR, "final_video.mp4")
    synthesize_audio(story, voice_path)
    mix_audio(voice_path, RAIN_SOUND, mixed_audio)
    thumb = generate_thumbnail(title)
    render_video(mixed_audio, RAIN_VIDEO, thumb, final_path)

