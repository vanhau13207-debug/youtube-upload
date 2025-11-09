#!/usr/bin/env python3
import os, sys, time, json, base64, requests, shutil
from datetime import datetime
OUTPUT_DIR = "workspace/output"
ASSETS_DIR = "workspace/assets"
FONT_PATH = os.path.join(ASSETS_DIR, "font.ttf")
RAIN_SOUND = os.path.join(ASSETS_DIR, "rain.mp3")
RAIN_VIDEO = os.path.join(ASSETS_DIR, "rain_bg.mp4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper: ensure assets exist (download lightweight fallbacks if missing)
def ensure_assets():
    if not os.path.exists(RAIN_SOUND):
        print("⚠️ rain.mp3 missing, downloading fallback...")
        try:
            url = "https://cdn.pixabay.com/download/audio/2022/03/15/audio_f8f4d05db0.mp3?filename=rain-ambient-1141.mp3"
            r = requests.get(url, timeout=20)
            open(RAIN_SOUND, "wb").write(r.content)
            print("✅ rain.mp3 downloaded")
        except Exception as e:
            print("❌ failed download rain:", e)
    if not os.path.exists(RAIN_VIDEO):
        print("⚠️ rain_bg.mp4 missing, downloading small fallback (loopable)...")
        try:
            url = "https://samplelib.com/lib/preview/mp4/sample-5s.mp4"
            r = requests.get(url, timeout=20)
            open(RAIN_VIDEO, "wb").write(r.content)
            print("✅ rain_bg.mp4 downloaded (short sample).")
        except Exception as e:
            print("❌ failed download rain video:", e)

ensure_assets()

# === Gemini story generator ===
def generate_story(seed):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Missing GEMINI_API_KEY, falling back to template story.")
        return "On a quiet rainy night, the world softened. The rain sang a lullaby..."
    prompt = f"Write a calm, cinematic English bedtime story about: {seed}. Tone: soothing, detailed, perfect for a 2-hour narrated video."
    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}",
            headers={"Content-Type":"application/json"},
            json={"contents":[{"parts":[{"text":prompt}]}]},
            timeout=30
        )
        j = resp.json()
        text = j["candidates"][0]["content"]["parts"][0]["text"].strip()
        print("✅ Gemini story generated.")
        return text
    except Exception as e:
        print("⚠️ Gemini story failed:", e)
        return "On a quiet rainy night, the world softened. The rain sang a lullaby..."

# === Gemini SEO generator ===
def generate_seo(story):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ("Rainy Night Chill Story 🌧️", "A calm story with rain sounds for sleep.", "rain,sleep,chill")
    prompt = f"""You are a YouTube SEO expert. Based on this story (first 1000 chars):\n\n{story[:1000]}\n\nReturn:
1) Title (one line, <=80 chars)
2) Description (2 short paragraphs)
3) 12 comma-separated tags (lowercase)."""
    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}",
            headers={"Content-Type":"application/json"},
            json={"contents":[{"parts":[{"text":prompt}]}]},
            timeout=25
        )
        j = resp.json()
        content = j["candidates"][0]["content"]["parts"][0]["text"].strip()
        # try parsing
        parts = [p.strip() for p in content.split("\n") if p.strip()]
        title = parts[0][:80] if parts else "Rainy Night Chill Story 🌧️"
        description = "\n".join(parts[1:3]) if len(parts) > 1 else "A calm story with rain."
        tags = ""
        # detect tags line
        for p in parts[::-1]:
            if "," in p and len(p.split(",")) >= 3:
                tags = p
                break
        if not tags:
            tags = "rain,chill,sleep,relax,asmr,story,ambient,study,cozy"
        print("✅ Gemini SEO generated.")
        return title, description, tags
    except Exception as e:
        print("⚠️ Gemini SEO failed:", e)
        return ("Rainy Night Chill Story 🌧️","A calm story with rain sounds for sleep.","rain,chill,sleep")

# === Thumbnail: extract frame -> Gemini Vision -> Pollinations -> overlay text ===
def generate_thumbnail(title, story):
    print("🖼️ Generating thumbnail (frame -> Gemini Vision -> Pollinations -> overlay)")
    try:
        import cv2, numpy as np
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        print("Installing deps for thumbnail...")
        os.system("pip install opencv-python-headless pillow numpy")
        import cv2, numpy as np
        from PIL import Image, ImageDraw, ImageFont

    # extract mid frame
    cap = cv2.VideoCapture(RAIN_VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total//2))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        frame = 255 * np.ones((720,1280,3), np.uint8)
    frame_path = os.path.join(OUTPUT_DIR, "frame_base.jpg")
    cv2.imwrite(frame_path, frame)

    # encode for vision
    api_key = os.getenv("GEMINI_API_KEY")
    desc_prompt = f"""
    You are a top YouTube thumbnail designer. Analyze the provided rainy-frame image and this title:
    "{title}"
    Suggest a concise image prompt (one sentence) optimized for high CTR thumbnails: cinematic, cozy, warm lights, clear center subject, space for bold text.
    """
    try:
        b64 = base64.b64encode(open(frame_path,"rb").read()).decode()
        payload = {"contents":[{"parts":[{"text":desc_prompt},{"inline_data":{"mime_type":"image/jpeg","data":b64}}]}]}
        resp = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent?key={api_key}",
                             headers={"Content-Type":"application/json"}, json=payload, timeout=40)
        j = resp.json()
        prompt_text = j["candidates"][0]["content"]["parts"][0]["text"].strip()
        print("🎯 Gemini vision prompt:", prompt_text)
    except Exception as e:
        print("⚠️ Gemini vision failed:", e)
        prompt_text = f"cinematic rainy night, cozy warm lights, soft bokeh, space for bold text, ultra-realistic, 4k, centered subject"

    # call Pollinations to generate image from prompt
    try:
        img_url = "https://image.pollinations.ai/prompt/" + requests.utils.quote(prompt_text)
        img_data = requests.get(img_url, timeout=30).content
        ai_thumb = os.path.join(OUTPUT_DIR, "ai_thumb.jpg")
        open(ai_thumb,"wb").write(img_data)
        base_img_path = ai_thumb
    except Exception as e:
        print("⚠️ Pollinations failed:", e)
        base_img_path = frame_path

    # overlay text
    from PIL import Image as PILImage, ImageDraw, ImageFont as PILFont
    img = PILImage.open(base_img_path).convert("RGBA")
    overlay = PILImage.new("RGBA", img.size, (0,0,0,110))
    img = PILImage.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)
    try:
        font_title = PILFont.truetype(FONT_PATH, 64)
        font_sub = PILFont.truetype(FONT_PATH, 36)
    except Exception:
        font_title = PILFont.load_default()
        font_sub = PILFont.load_default()
    # wrap title
    def wrap_text(s, width=28):
        words = s.split()
        lines = []
        cur = ""
        for w in words:
            if len(cur + " " + w) <= width:
                cur = (cur + " " + w).strip()
            else:
                lines.append(cur); cur = w
        if cur: lines.append(cur)
        return "\n".join(lines)
    twrap = wrap_text(title, 28)
    draw.text((60, img.size[1]-220), twrap, font=font_title, fill=(255,255,255,255))
    draw.text((60, img.size[1]-80), "Relax • Sleep • Rain • Chill", font=font_sub, fill=(200,200,200,230))
    final_thumb = os.path.join(OUTPUT_DIR, "thumbnail.jpg")
    img.convert("RGB").save(final_thumb, quality=95)
    print("✅ Thumbnail ready:", final_thumb)
    return final_thumb

# === TTS via Coqui (offline) ===
def synthesize_audio(text, out_path):
    print("🎙️ Synthesizing audio (Coqui TTS)...")
    try:
        from TTS.api import TTS
        tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)
        # choose a speaker id if multi-speaker model; change if necessary
        try:
            tts.tts_to_file(text=text, file_path=out_path, speaker="p273")
        except TypeError:
            # model may not accept speaker param
            tts.tts_to_file(text=text, file_path=out_path)
        print("✅ Voice saved:", out_path)
        return out_path
    except Exception as e:
        print("⚠️ Coqui TTS failed:", e)
        # fallback: create a silent WAV or copy rain sound
        fallback = os.path.join(OUTPUT_DIR, "voice_fallback.wav")
        shutil.copy(RAIN_SOUND, fallback)
        return fallback

# === Mix voice + rain => final audio
def mix_audio(voice, rain, out_mixed):
    print("🎧 Mixing voice and rain via ffmpeg...")
    cmd = f'ffmpeg -y -i "{voice}" -i "{rain}" -filter_complex "[1:a]volume=0.20[a1];[0:a][a1]amix=inputs=2:duration=first:dropout_transition=2" -c:a pcm_s16le "{out_mixed}"'
    os.system(cmd)
    return out_mixed

# === Render video using ffmpeg (loop bg video) ===
def render_video(audio_path, bg_video_path, out_path, duration_seconds=7200):
    print("🎬 Rendering final video (ffmpeg)...")
    tmp_bg = os.path.join(OUTPUT_DIR, "tmp_bg_loop.mp4")
    # create looped bg of exact duration
    os.system(f'ffmpeg -y -stream_loop -1 -i "{bg_video_path}" -t {duration_seconds} -c copy "{tmp_bg}"')
    # mux audio and looped bg into final
    os.system(f'ffmpeg -y -i "{tmp_bg}" -i "{audio_path}" -c:v libx264 -preset veryfast -crf 23 -c:a aac -b:a 160k -shortest "{out_path}"')
    print("✅ Rendered video:", out_path)
    return out_path

# === Main flow ===
def build_one(seed, duration_seconds):
    print("=== Start build:", seed)
    story = generate_story(seed)
    title, desc, tags = generate_seo(story)
    voice = os.path.join(OUTPUT_DIR, "voice.wav")
    synthesize_audio(story, voice)
    mixed = os.path.join(OUTPUT_DIR, "mixed.wav")
    mix_audio(voice, RAIN_SOUND, mixed)
    thumb = generate_thumbnail(title, story)
    final = os.path.join(OUTPUT_DIR, "final_video.mp4")
    render_video(mixed, RAIN_VIDEO, final, duration_seconds)
    # save metadata
    meta = {"title": title, "description": desc, "tags": tags}
    open(os.path.join(OUTPUT_DIR,"meta.json"), "w").write(json.dumps(meta, ensure_ascii=False, indent=2))
    print("=== Build complete:", final)
    return final

if __name__ == "__main__":
    # default seeds (can pass --seed)
    seeds = [
        "a cozy night by the lake with rain sounds",
        "a calm story in a mountain cabin during a storm",
        "relaxing rainy afternoon in a city cafe"
    ]
    # for GitHub Actions we will render one video per run; choose first or override with env/arg
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default=None)
    p.add_argument("--duration", default=os.getenv("TARGET_DURATION","7200"))
    args = p.parse_args()
    seed = args.seed or seeds[0]
    dur = int(args.duration)
    build_one(seed, dur)
