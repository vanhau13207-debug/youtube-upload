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

# === SEO GENERATION (Gemini API) ===
def generate_seo(story):
    print("🧠 Generating SEO title, description, and tags with Gemini...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Missing GEMINI_API_KEY, using fallback SEO...")
        return (
            "Rainy Night Chill Story 🌧️ Peaceful Sleep & Relaxation",
            "A calming bedtime story with rain sounds and cinematic vibes — perfect for sleep, relaxation, or focus.",
            "rain sounds,chill,relaxing music,sleep,study,storytelling,asmr,calm night,bedtime story,peaceful"
        )

    prompt = f"""
    You are a YouTube SEO expert.
    Based on this chill English bedtime story:
    ---
    {story[:1200]}
    ---
    Write optimized metadata for YouTube:
    1️⃣ A catchy, emotional Title (max 80 chars, includes “Rain”, “Chill”, or “Relax”)
    2️⃣ A human-like Description (2 paragraphs, natural tone)
    3️⃣ 15 SEO tags (comma-separated, lowercase, no '#', focused on chill, sleep, and rain)
    """

    try:
        res = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=45
        )
        data = res.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        # === parse nội dung trả về ===
        lines = content.splitlines()
        title = ""
        description = ""
        tags = ""
        for line in lines:
            if not title and ("title:" in line.lower() or line.strip().startswith("1")):
                title = line.split(":", 1)[-1].strip()
            elif "description" in line.lower() or line.strip().startswith("2"):
                description = "\n".join(lines[lines.index(line)+1:]).strip()
                break

        # tìm tags ở cuối văn bản
        if "tags" in content.lower():
            tags_part = content.lower().split("tags")[-1]
            tags = tags_part.replace(":", "").replace("\n", ",").strip()

        if not tags:
            tags = "chill,relaxing,rain sounds,sleep,study,storytelling,asmr,calm night,peaceful,cozy"
        return title[:80], description[:2000], tags[:500]
    except Exception as e:
        print("⚠️ SEO generation failed:", e)
        return (
            "Rain Ambience Story 🌧️ Chill Night Vibes for Sleep & Study",
            "A 2-hour peaceful story with soft rain ambience, relaxing voice, and cinematic visuals.",
            "rain sounds,chill,relaxing,asmr,bedtime story,calm night,peaceful,vibes,sleep,study,cozy"
        )


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

# === THUMBNAIL GEMINI + FRAME FROM VIDEO ===
def generate_thumbnail(title, story):
    print("🖼️ Generating SEO thumbnail from rain_bg.mp4 via Gemini...")

    import base64, cv2, numpy as np
    from PIL import Image, ImageDraw, ImageFont

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Missing GEMINI_API_KEY — fallback thumbnail.")
        return "workspace/assets/default_thumb.jpg"

    # === 1️⃣ Lấy 1 frame giữa video nền ===
    bg_video = "workspace/assets/rain_bg.mp4"
    cap = cv2.VideoCapture(bg_video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("⚠️ Fallback frame (empty image)")
        frame = np.zeros((720, 1280, 3), np.uint8)

    # Lưu frame tạm
    frame_path = os.path.join(OUTPUT_DIR, "frame_base.jpg")
    cv2.imwrite(frame_path, frame)

    # === 2️⃣ Encode frame để gửi Gemini ===
    with open(frame_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    # === 3️⃣ Gọi Gemini sinh prompt mô tả thumbnail ===
    prompt = f"""
    You are a YouTube thumbnail designer & SEO expert.
    Based on this image (rain scene) and the video title:
    "{title}"
    Write a short English description (1 sentence) for generating a cinematic, aesthetic thumbnail image for YouTube.
    Make it optimized for emotional engagement and high CTR.
    """

    try:
        res = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": img_base64}}
                    ]
                }]
            },
            timeout=40
        )

        data = res.json()
        desc = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        print(f"🎨 Gemini thumbnail prompt: {desc}")
    except Exception as e:
        print("⚠️ Gemini vision failed:", e)
        desc = f"cinematic rainy night, cozy lights, aesthetic 4k scene, for: {title}"

    # === 4️⃣ Gọi Pollinations để tạo ảnh từ prompt Gemini ===
    try:
        img_url = "https://image.pollinations.ai/prompt/" + requests.utils.quote(desc)
        img_data = requests.get(img_url).content
        ai_thumb_path = os.path.join(OUTPUT_DIR, "ai_thumb.jpg")
        with open(ai_thumb_path, "wb") as f:
            f.write(img_data)
    except Exception as e:
        print("⚠️ Pollinations download failed:", e)
        ai_thumb_path = frame_path

    # === 5️⃣ Overlay SEO text lên thumbnail ===
    img = Image.open(ai_thumb_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 100))
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype(FONT_PATH, 70)
        font_sub = ImageFont.truetype(FONT_PATH, 36)
    except:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    draw.text((60, 500), title[:60], fill=(255, 255, 255, 255), font=font_title)
    draw.text((60, 650), "Relax • Sleep • Rain • Chill", fill=(200, 200, 200, 230), font=font_sub)

    final_path = os.path.join(OUTPUT_DIR, "thumbnail.jpg")
    img.convert("RGB").save(final_path, quality=95)
    print(f"✅ Thumbnail saved: {final_path}")
    return final_path


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


