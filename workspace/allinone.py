import os, random, json, requests, time
from TTS.api import TTS
from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont

# === CONFIG ===
OUTPUT_DIR = "workspace/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
YT_UPLOAD = os.getenv("YT_UPLOAD", "false").lower() == "true"
TARGET_DURATION = int(os.getenv("TARGET_DURATION", "600"))

# === 1. Generate Story ===
def generate_story(seed):
    prompt = f"Write a relaxing English bedtime story based on: {seed}. Make it gentle, cozy, and vivid for YouTube narration."
    headers = {"Authorization": f"Bearer {OPENAI_KEY}"}
    res = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": prompt}]}
    )
    story = res.json()["choices"][0]["message"]["content"]
    return story.strip()

# === 2. Text to Speech (Coqui) ===
def story_to_audio(text, out_path="output.wav"):
    tts = TTS("tts_models/en/vctk/vits")
    tts.tts_to_file(text=text, file_path=out_path)

# === 3. Generate thumbnail (AI) ===
def generate_thumbnail(title):
    img_path = os.path.join(OUTPUT_DIR, f"{title[:40].replace(' ','_')}.jpg")
    try:
        res = requests.post(
            "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2",
            headers={"Authorization": "Bearer hf_jGQiExampleFakeKey"},
            json={"inputs": f"A cozy chill YouTube thumbnail for {title}, cinematic lighting, rain, storytime, peaceful vibe"}
        )
        with open(img_path, "wb") as f:
            f.write(res.content)
    except Exception:
        Image.new("RGB", (1280, 720), (30, 30, 30)).save(img_path)
    return img_path

# === 4. Generate SEO metadata ===
def auto_seo(title):
    prompt = f"Generate a YouTube title, description, and tags for: {title}. Focus on SEO, English, cozy, rain, chill storytelling niche."
    headers = {"Authorization": f"Bearer {OPENAI_KEY}"}
    res = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": prompt}]}
    )
    data = res.json()["choices"][0]["message"]["content"]
    return data

# === 5. Render Video ===
def render_video(audio_path, thumbnail_path, title):
    audio = AudioFileClip(audio_path)
    duration = audio.duration
    bg = ImageClip(thumbnail_path).set_duration(duration).resize((1280, 720))
    video = bg.set_audio(audio)
    out_path = os.path.join(OUTPUT_DIR, f"{title[:50].replace(' ','_')}.mp4")
    video.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac")
    return out_path

# === MAIN ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, default="rainy night story")
    args = parser.parse_args()

    story = generate_story(args.seed)
    title = args.seed.title()
    audio_path = os.path.join(OUTPUT_DIR, "voice.wav")
    story_to_audio(story, audio_path)

    thumb = generate_thumbnail(title)
    meta = auto_seo(title)
    video = render_video(audio_path, thumb, title)

    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
        json.dump({"title": title, "seo": meta, "video": video}, f, indent=2)
    print("✅ Done:", title)
