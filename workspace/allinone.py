import os, json, requests, time
from TTS.api import TTS
from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "workspace/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TARGET_DURATION = int(os.getenv("TARGET_DURATION", "600"))
YT_UPLOAD = os.getenv("YT_UPLOAD", "false").lower() == "true"

# === 1. Sinh truyện ===
def generate_story(seed):
    prompt = f"Write a relaxing English bedtime story about {seed}. Make it cozy, imaginative, and peaceful, perfect for YouTube narration."
    res = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_KEY}"},
        json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": prompt}]}
    )
    return res.json()["choices"][0]["message"]["content"].strip()

# === 2. Text-to-speech ===
def story_to_audio(text, out_path="voice.wav"):
    tts = TTS("tts_models/en/vctk/vits")
    tts.tts_to_file(text=text, file_path=out_path)

# === 3. Thumbnail AI ===
def generate_thumbnail(title):
    img_path = os.path.join(OUTPUT_DIR, f"{title[:40].replace(' ', '_')}.jpg")
    try:
        r = requests.post(
            "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2",
            headers={"Authorization": "Bearer hf_fakeapikey"},
            json={"inputs": f"A cinematic chill YouTube thumbnail for '{title}', cozy rain, ambient light, peaceful mood"}
        )
        with open(img_path, "wb") as f:
            f.write(r.content)
    except:
        Image.new("RGB", (1280, 720), (40, 40, 40)).save(img_path)
    return img_path

# === 4. SEO metadata ===
def auto_seo(title):
    prompt = f"Generate a YouTube title, description, and tags in English for: {title}. Focus on rain sounds, storytelling, relaxing, and SEO optimization."
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_KEY}"},
        json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": prompt}]}
    )
    return r.json()["choices"][0]["message"]["content"]

# === 5. Render video ===
def render_video(audio_path, thumb_path, title):
    audio = AudioFileClip(audio_path)
    bg = ImageClip(thumb_path).set_duration(audio.duration).resize((1280, 720))
    video = bg.set_audio(audio)
    out = os.path.join(OUTPUT_DIR, f"{title[:50].replace(' ','_')}.mp4")
    video.write_videofile(out, fps=24, codec="libx264", audio_codec="aac")
    return out

# === MAIN ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, default="rainy night")
    args = parser.parse_args()

    story = generate_story(args.seed)
    title = args.seed.title()

    audio = os.path.join(OUTPUT_DIR, "voice.wav")
    story_to_audio(story, audio)

    thumb = generate_thumbnail(title)
    seo_data = auto_seo(title)
    video = render_video(audio, thumb, title)

    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
        json.dump({"title": title, "seo": seo_data, "video": video}, f, indent=2)
    print("✅ Done:", title)
