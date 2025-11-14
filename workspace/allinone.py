#!/usr/bin/env python3
import os, json, time, base64, logging, datetime, subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import numpy as np
import soundfile as sf

from PIL import Image, ImageDraw, ImageFont

# ======== PATHS ========
ROOT = Path.cwd()
WORKSPACE = ROOT / "workspace"
OUTPUT = WORKSPACE / "output"
ASSETS = WORKSPACE / "assets"

OUTPUT.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

# ======== ENV ========
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_TEXT_MODEL = "gemini-2.0-flash"
GEMINI_IMAGE_MODEL = "gemini-2.0-flash"

COQUI_MODEL = os.getenv("COQUI_MODEL", "tts_models/en/vctk/vits")

RAIN_MP3 = Path(os.getenv("RAIN_FILE", ASSETS/"rain.mp3"))
RAIN_BG = Path(os.getenv("RAIN_BG", ASSETS/"rain_bg.mp4"))

MIN_DURATION = int(os.getenv("TARGET_DURATION", "7200"))
AUDIO_SR = 22050
TTS_WPM = 150
THUMB_AT = 10.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ====================================================================
# GEMINI TEXT API
# ====================================================================
def gemini_text(prompt: str):
    if not GEMINI_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY")

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    r = requests.post(url, params={"key": GEMINI_KEY}, json=payload, timeout=60)
    r.raise_for_status()
    js = r.json()

    try:
        return js["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return str(js)
# === FAST GEMINI IMAGE (Flash Image Model, super fast) ===
def gemini_image(ref_b64: str):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-image:generateImage"


    payload = {
        "prompt": {
            "text": "Enhance into a high CTR cinematic rain ambience thumbnail. No text. Keep composition."
        },
        "image_context": {
            "reference_images": [
                {"image_bytes": ref_b64}
            ]
        },
        "format": "JPEG"
    }

    try:
        r = requests.post(url, params={"key": GEMINI_KEY}, json=payload, timeout=20)
        r.raise_for_status()
        js = r.json()
        return base64.b64decode(js["images"][0]["imageBytes"])
    except Exception as e:
        logging.warning(f"Thumbnail generation failed: {e}")
        return None


# ====================================================================
# STORY GENERATOR (multi-batch until long enough)
# ====================================================================
def estimate_secs(text: str):
    return int((len(text.split()) / TTS_WPM) * 60)

def generate_story(seed: str, min_s: int):
    full = ""
    tries = 0
    prompt = (
        f"Write a long, slow atmospheric English story for sleep/study.\n"
        f"Topic: {seed}.\n"
        f"Calm, descriptive, 1400+ words.\n"
        f"No titles. Only story paragraphs.\n"
    )

    while True:
        tries += 1
        try:
            part = gemini_text(prompt)
        except:
            part = "The rain whispered softly as the quiet room embraced the night."

        full += "\n\n" + part.strip()

        secs = estimate_secs(full)
        logging.info(f"Batch {tries} → estimated {secs}s")

        if secs >= min_s:
            break
        if tries >= 12:
            logging.info("Reached max batches.")
            break

        time.sleep(0.5)

    return full.strip()

# ====================================================================
# INTRO (5–7 seconds)
# ====================================================================
def generate_intro(title: str, seed: str):
    prompt = (
        "Write a VERY SHORT spoken intro for a relaxing rain storytelling video. "
        "Length: 1–2 sentences, about 5–7 seconds when spoken. "
        "Tone: warm, calm, soft. No plot spoilers.\n\n"
        f"Title: {title}\nTopic: {seed}"
    )

    try:
        return gemini_text(prompt).strip()
    except:
        return "Welcome. Tonight’s rain story begins now."

# ====================================================================
# COQUI TTS
# ====================================================================
try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except:
    COQUI_AVAILABLE = False

def tts_generate(text: str, out: Path):
    if not COQUI_AVAILABLE:
        raise RuntimeError("Coqui missing")

    tts = TTS(model_name=COQUI_MODEL)

    maxc = 2500
    parts = []
    buf = ""

    for p in text.split("\n"):
        if len(buf) + len(p) > maxc:
            parts.append(buf)
            buf = p
        else:
            buf = (buf + "\n" + p).strip()

    if buf:
        parts.append(buf)

    segs = []
    sr = None

    for i, p in enumerate(parts):
        tmp = out.parent / f"tts_{i}.wav"
        tts.tts_to_file(text=p, file_path=str(tmp))
        segs.append(tmp)

    arrays = []

    for s in segs:
        a, r = sf.read(s, dtype="float32")
        sr = r if sr is None else sr
        arrays.append(a)

    full = np.concatenate(arrays)
    sf.write(out, full, sr)

    for s in segs:
        s.unlink(missing_ok=True)

def wav_len(wav: Path):
    try:
        d, sr = sf.read(wav, dtype="float32")
        if d.ndim > 1:
            d = d.mean(axis=1)
        return len(d) / sr
    except:
        return 0

# ====================================================================
# AUDIO HELPERS
# ====================================================================
def silent(sec: int, out: Path):
    data = np.zeros(int(sec * AUDIO_SR), dtype="float32")
    sf.write(out, data, AUDIO_SR)

def loop_audio(src: Path, dur: int, out: Path):
    d, sr = sf.read(src, dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)

    need = int(dur * sr)
    rep = (need // len(d)) + 1
    full = np.tile(d, rep)[:need]
    sf.write(out, full, sr)

def mix_audio(vw: Path, rw: Path, dur: int, out: Path):
    v, sv = sf.read(vw, dtype="float32")
    r, sr = sf.read(rw, dtype="float32")

    if v.ndim > 1: v = v.mean(axis=1)
    if r.ndim > 1: r = r.mean(axis=1)

    if sv != sr:
        v = np.interp(
            np.linspace(0, len(v), int(len(v) * sr / sv), endpoint=False),
            np.arange(len(v)),
            v
        )

    need = int(dur * sr)
    if len(r) < need:
        r = np.tile(r, (need // len(r)) + 1)[:need]
    else:
        r = r[:need]

    if len(v) < need:
        v = np.concatenate([v, np.zeros(need - len(v))])
    else:
        v = v[:need]

    r = r * (10 ** (-12 / 20))  # rain quieter
    mix = v + r

    m = np.max(np.abs(mix))
    if m > 1: mix /= m

    sf.write(out, mix, sr)

# ====================================================================
# FRAME EXTRACT
# ====================================================================
def extract_frame(video: Path, sec: float, out: Path):
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(sec),
        "-i", str(video),
        "-vframes", "1",
        str(out)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logging.info(f"Thumbnail frame extracted → {out}")

# ====================================================================
# THUMBNAIL FALLBACK
# ====================================================================
def fallback_thumb(text: str, out: Path):
    img = Image.new("RGB", (1280, 720), (20, 20, 30))
    d = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 60)
    except:
        font = ImageFont.load_default()

    d.text((60, 260), text[:70], font=font, fill=(240, 240, 240))
    img.save(out)

# ====================================================================
# RENDER VIDEO (FFmpeg)
# ====================================================================
def render_video(bg: Path, audio: Path, out: Path, dur: int):
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", str(bg),
        "-i", str(audio),
        "-t", str(dur),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-r", "24",
        "-c:a", "aac",
        "-shortest",
        str(out)
]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.info(f"Rendered final video → {out}")

# ====================================================================
# MAIN
# ====================================================================
def main():
    logging.info("=== allinone FFmpeg pipeline start ===")

    seed = os.getenv("VIDEO_SEED", "Cozy rainy focus")
    logging.info(f"Seed: {seed}")

    # STORY
    story = generate_story(seed, MIN_DURATION)
    (OUTPUT/"story.txt").write_text(story, encoding="utf-8")

    # SEO
    try:
        seo_prompt = (
            "Give JSON: {title:'',description:'',tags:[...]}. "
            "Make SEO for rain ambience storytelling. "
            f"Story: {story[:1500]}"
        )
        raw = gemini_text(seo_prompt)
        js = json.loads(raw)
        title = js.get("title", f"Rain Story — {seed}")
        desc = js.get("description", "")
        tags = js.get("tags", ["rain","sleep","study"])
    except:
        title = f"Rain Story — {seed}"
        desc = "Relaxing rain ambience with storytelling."
        tags = ["rain","sleep","study"]

    (OUTPUT/"title.txt").write_text(title, encoding="utf-8")
    (OUTPUT/"description.txt").write_text(desc, encoding="utf-8")
    (OUTPUT/"tags.txt").write_text(",".join(tags), encoding="utf-8")

    # INTRO
    intro = generate_intro(title, seed)
    combined = intro + "\n\n" + story

    # TTS
    voice_wav = OUTPUT/"voice.wav"
    try:
        tts_generate(combined, voice_wav)
    except:
        silent(10, voice_wav)

    voice_len = int(wav_len(voice_wav))
    final_duration = max(MIN_DURATION, voice_len)

    # RAIN LOOP
    rain_loop = OUTPUT/"rain_loop.wav"
    try:
        loop_audio(RAIN_MP3, final_duration, rain_loop)
    except:
        silent(final_duration, rain_loop)

    # MIX
    final_audio = OUTPUT/"final_audio.wav"
    mix_audio(voice_wav, rain_loop, final_duration, final_audio)

    # === THUMBNAIL ===
    frame_png = OUTPUT / "thumb_frame.png"
    thumb_jpg = OUTPUT / "thumbnail.jpg"

    try:
        extract_frame(RAIN_BG, THUMB_AT, frame_png)

        with open(frame_png, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        enhanced = gemini_image(b64)

        # Check enhanced image
        if enhanced and isinstance(enhanced, (bytes, bytearray)) and len(enhanced) > 1000:
            thumb_jpg.write_bytes(enhanced)
            logging.info("Thumbnail generated by Gemini")
        else:
            logging.warning("Gemini returned invalid image → using fallback thumbnail")
            fallback_thumb(title, thumb_jpg)

    except Exception as e:
        logging.warning(f"Thumbnail failed: {e}")
        fallback_thumb(title, thumb_jpg)


    # RENDER VIDEO
    final_video = OUTPUT/"final_video.mp4"
    render_video(RAIN_BG, final_audio, final_video, final_duration)

    logging.info("=== DONE ===")


if __name__ == "__main__":
    main()




