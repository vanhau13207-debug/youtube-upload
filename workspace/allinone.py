#!/usr/bin/env python3
"""
allinone.py — FULL pipeline (auto story -> TTS -> mix -> video -> SEO -> thumbnail)
Drop-in for workspace/allinone.py in repo.

Features:
- Generate story via API: GEMINI_API_KEY / OPENAI_API_KEY or fallback to local files
- Generate SEO (title/description/tags) via API if available
- Generate thumbnail image via IMAGE_API_ENDPOINT or Pollinations if configured
- Coqui TTS integration (uses env COQUI_MODEL) when available; fallback to silent voice
- Mix voice + rain ambience, render MP4 via moviepy
- Saves outputs in workspace/output/
- Configurable via environment variables

ENV VARs used (optional):
- GEMINI_API_KEY or OPENAI_API_KEY
- GENERATOR_ENDPOINT (custom generator endpoint)
- SEO_API_KEY / SEO_ENDPOINT (optional)
- IMAGE_API_KEY / IMAGE_ENDPOINT (optional)
- COQUI_MODEL (e.g. tts_models/en/vctk/vits)
- RAIN_FILE (path to rain file; default assets/rain.mp3)
- BG_IMAGE (path to background image; default assets/bg.jpg)
- TARGET_DURATION (seconds) or --duration
"""

import os
import sys
import argparse
import uuid
import json
import random
import datetime
import logging
from pathlib import Path
from typing import Optional, Tuple, List

# optional Coqui TTS
try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except Exception:
    COQUI_AVAILABLE = False

import numpy as np
import soundfile as sf

from moviepy.editor import ImageClip, AudioFileClip

import requests

# Paths
ROOT = Path.cwd()
WORKSPACE = ROOT / "workspace"
STORIES = WORKSPACE / "stories"
ASSETS = ROOT / "assets"
OUTPUT = WORKSPACE / "output"

# Ensure directories
OUTPUT.mkdir(parents=True, exist_ok=True)
STORIES.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

# Defaults
DEFAULT_RAIN = ASSETS / "rain.mp3"
DEFAULT_BG = ASSETS / "bg.jpg"
DEFAULT_COQUI_MODEL = os.getenv("COQUI_MODEL", "tts_models/en/vctk/vits")
DEFAULT_AUDIO_SR = int(os.getenv("AUDIO_SR", "22050"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------------------
#  API: Story / SEO / Image
# ---------------------------
def call_text_generator(seed: Optional[str], max_tokens: int = 2500) -> str:
    """Try GEMINI/OPENAI or GENERATOR_ENDPOINT. Returns text or raise."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    endpoint = os.getenv("GENERATOR_ENDPOINT")
    prompt = (seed or "A calm cozy rainy night story") + "\n\nWrite a long, slow, soothing narrated story suitable for a multi-hour ambient video. Use gentle language."
    headers = {}
    if endpoint:
        headers["Authorization"] = f"Bearer {os.getenv('GENERATOR_KEY','')}" if os.getenv('GENERATOR_KEY') else ""
        try:
            r = requests.post(endpoint, json={"prompt": prompt, "max_tokens": max_tokens}, timeout=60)
            r.raise_for_status()
            js = r.json()
            # support multiple shapes
            return js.get("text") or (js.get("choices") and js["choices"][0].get("text")) or js.get("output") or json.dumps(js)
        except Exception as e:
            raise RuntimeError(f"Generator endpoint failed: {e}")

    if key:
        # Try simple OpenAI-compatible REST call (user must set API-compatible endpoint via GENERATOR_ENDPOINT ideally)
        # We'll attempt basic OpenAI chat completions if OPENAI_API_KEY present and no endpoint configured
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and not os.getenv("GENERATOR_ENDPOINT"):
            # call OpenAI chat completions (simple)
            url = "https://api.openai.com/v1/chat/completions"
            hdr = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
            payload = {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            try:
                r = requests.post(url, headers=hdr, json=payload, timeout=60)
                r.raise_for_status()
                data = r.json()
                txt = None
                if "choices" in data and data["choices"]:
                    txt = data["choices"][0].get("message", {}).get("content")
                if not txt:
                    txt = data.get("choices", [{}])[0].get("text")
                if txt:
                    return txt
            except Exception as e:
                raise RuntimeError(f"OpenAI call failed: {e}")

    raise RuntimeError("No generator available (set GENERATOR_ENDPOINT or OPENAI_API_KEY/GEMINI_API_KEY)")


def call_seo_generator(seed: str, story_excerpt: str) -> Tuple[str, str, List[str]]:
    """Generate title, description, tags via API if configured; otherwise simple heuristic."""
    seo_endpoint = os.getenv("SEO_ENDPOINT")
    seo_key = os.getenv("SEO_API_KEY") or os.getenv("OPENAI_API_KEY")
    if seo_endpoint:
        try:
            r = requests.post(seo_endpoint, json={"seed": seed, "excerpt": story_excerpt}, headers={"Authorization": f"Bearer {seo_key}"}, timeout=30)
            r.raise_for_status()
            js = r.json()
            return js.get("title",""), js.get("description",""), js.get("tags",[] or [])
        except Exception:
            pass
    # fallback simple generator
    title = f"Relaxing Rainy Story — {seed or 'Cozy Night'} • Sleep & Focus"
    description = ("A relaxing rainy story with ambient rain sounds to help you sleep, relax, or focus. "
                   "Automatically generated.")
    tags = ["rain", "sleep", "relax", "asmr", "story", "ambient", "chill"]
    return title, description, tags


def call_image_generator(title: str, seed: Optional[str]) -> Optional[bytes]:
    """Generate thumbnail image bytes via IMAGE_ENDPOINT or Pollinations (if configured). Return image bytes or None."""
    image_endpoint = os.getenv("IMAGE_ENDPOINT")
    image_key = os.getenv("IMAGE_API_KEY")
    prompt = f"cozy rainy night, warm lights, cinematic, soft, cozy, text: {title}"
    if image_endpoint:
        headers = {"Authorization": f"Bearer {image_key}"} if image_key else {}
        try:
            r = requests.post(image_endpoint, json={"prompt": prompt, "size": "1024x576"}, headers=headers, timeout=60)
            r.raise_for_status()
            # accept either direct bytes or JSON with base64
            if r.headers.get("content-type","").startswith("image"):
                return r.content
            js = r.json()
            if "image_base64" in js:
                import base64
                return base64.b64decode(js["image_base64"])
            if "images" in js and js["images"]:
                import base64
                return base64.b64decode(js["images"][0])
        except Exception:
            logging.warning("Image endpoint failed")
            return None
    # Pollinations fallback (simple POST)
    poll_url = "https://image.pollinations.ai/prompt/" + requests.utils.quote(prompt)
    try:
        r = requests.get(poll_url, timeout=30)
        if r.status_code == 200 and r.headers.get("content-type","").startswith("image"):
            return r.content
    except Exception:
        pass
    return None


# ---------------------------
#  Local helpers: stories
# ---------------------------
def pick_local_story(seed: Optional[str]) -> str:
    files = list(STORIES.glob("*.txt"))
    if files:
        chosen = random.choice(files)
        logging.info(f"Using local story file: {chosen}")
        return chosen.read_text(encoding="utf-8")
    # fallback base
    base = (
        "The rain whispered against the window as the town settled into a slow, measured breath. "
        "Tonight, the lamps glowed softly and the streets remembered the footsteps of those who passed. "
        "You breathe in time with the rain and let the world blur at the edges..."
    )
    return "\n\n".join([base for _ in range(60)])


# ---------------------------
#  TTS / audio utilities
# ---------------------------
def tts_generate_coqui(text: str, out_path: Path, model_name: str = DEFAULT_COQUI_MODEL):
    if not COQUI_AVAILABLE:
        raise RuntimeError("Coqui TTS not available (install TTS)")
    logging.info(f"Generating TTS with model {model_name}")
    tts = TTS(model_name)
    # chunk text into parts to avoid OOM
    max_chars = 2500
    parts = []
    cur = ""
    for para in text.split("\n"):
        if len(cur) + len(para) + 1 > max_chars:
            parts.append(cur)
            cur = para
        else:
            cur = (cur + "\n" + para).strip()
    if cur:
        parts.append(cur)
    tmp_files = []
    for i, p in enumerate(parts):
        tmp = out_path.parent / f"voice_part_{i}.wav"
        logging.info(f"TTS chunk {i+1}/{len(parts)} size {len(p)} chars")
        tts.tts_to_file(text=p, file_path=str(tmp))
        tmp_files.append(tmp)
    # concatenate parts
    arrays = []
    sr = None
    for t in tmp_files:
        a, s = sf.read(str(t), dtype='float32')
        if sr is None:
            sr = s
        if s != sr:
            logging.warning("Resampling; sample rates differ")
        arrays.append(a)
    if not arrays:
        raise RuntimeError("No TTS output")
    full = np.concatenate(arrays, axis=0)
    sf.write(str(out_path), full, sr or DEFAULT_AUDIO_SR)
    # cleanup
    for t in tmp_files:
        try:
            t.unlink()
        except Exception:
            pass
    logging.info(f"Coqui TTS written: {out_path}")


def make_silent_wave(seconds: int, out_path: Path, sr: int = DEFAULT_AUDIO_SR):
    total = int(seconds * sr)
    data = np.zeros((total,), dtype='float32')
    sf.write(str(out_path), data, sr)
    logging.info(f"Created silent wave {out_path} ({seconds}s)")


def loop_audio_to_duration(src: Path, duration_s: int, out_path: Path):
    data, sr = sf.read(str(src), dtype='float32')
    if data.ndim > 1:
        data = data.mean(axis=1)
    needed = int(duration_s * sr)
    chunks = []
    idx = 0
    while idx < needed:
        take = min(len(data), needed - idx)
        chunks.append(data[:take])
        idx += take
    full = np.concatenate(chunks, axis=0)
    sf.write(str(out_path), full, sr)
    logging.info(f"Looped audio to {duration_s}s -> {out_path}")


def mix_voice_and_rain(voice_path: Path, rain_path: Path, duration_s: int, out_path: Path, voice_db=0.0, rain_db=-12.0):
    v, sr_v = sf.read(str(voice_path), dtype='float32')
    r, sr_r = sf.read(str(rain_path), dtype='float32')
    if v.ndim > 1:
        v = v.mean(axis=1)
    if r.ndim > 1:
        r = r.mean(axis=1)
    if sr_v != sr_r:
        # naive resample v -> sr_r
        factor = sr_r / sr_v
        v = np.interp(np.linspace(0, len(v), int(len(v)*factor), endpoint=False), np.arange(len(v)), v)
        sr = sr_r
    else:
        sr = sr_v
    needed = int(duration_s * sr)
    if len(r) < needed:
        reps = (needed // len(r)) + 1
        r = np.tile(r, reps)[:needed]
    else:
        r = r[:needed]
    if len(v) < needed:
        v = np.concatenate([v, np.zeros(needed - len(v), dtype='float32')])
    else:
        v = v[:needed]
    def dbm(db): return 10 ** (db/20.0)
    v = v * dbm(voice_db)
    r = r * dbm(rain_db)
    mixed = v + r
    mx = np.max(np.abs(mixed)) if mixed.size else 1.0
    if mx > 1.0:
        mixed = mixed / mx
    sf.write(str(out_path), mixed, sr)
    logging.info(f"Mixed final audio to {out_path}")


# ---------------------------
#  Video rendering
# ---------------------------
def render_video(audio_path: Path, out_video: Path, duration_s: int, bg_image: Optional[Path] = None):
    if bg_image and bg_image.exists():
        img = str(bg_image)
    else:
        tmp = OUTPUT / "bg_auto.jpg"
        if not tmp.exists():
            from PIL import Image, ImageDraw
            im = Image.new("RGB", (1280,720), (10,10,20))
            d = ImageDraw.Draw(im)
            d.text((50,300), "Relaxing Rainy Story", fill=(200,200,220))
            im.save(tmp)
        img = str(tmp)
    clip = ImageClip(img).set_duration(duration_s)
    audio = AudioFileClip(str(audio_path))
    clip = clip.set_audio(audio)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(str(out_video), fps=1, codec="libx264", audio_codec="aac", threads=2, verbose=False, logger=None)
    logging.info(f"Rendered video {out_video}")


# ---------------------------
#  IO helpers: save SEO + thumbnail
# ---------------------------
def save_seo_files(title: str, description: str, tags: List[str]):
    (OUTPUT / "title.txt").write_text(title, encoding="utf-8")
    (OUTPUT / "description.txt").write_text(description, encoding="utf-8")
    (OUTPUT / "tags.txt").write_text(",".join(tags), encoding="utf-8")
    logging.info("Saved SEO files")


def save_thumbnail_bytes(b: bytes, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(b)
    logging.info(f"Saved thumbnail {out_path}")


# ---------------------------
#  Main pipeline
# ---------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=str, default=None)
    p.add_argument("--duration", type=int, default=int(os.getenv("TARGET_DURATION", "7200")))
    p.add_argument("--no-tts", action="store_true")
    p.add_argument("--keep-intermediate", action="store_true")
    args = p.parse_args()

    seed = args.seed or os.getenv("VIDEO_SEED", "Cozy Rainy Night")
    duration = args.duration
    start = datetime.datetime.utcnow().isoformat()
    logging.info("Starting full allinone pipeline")

    # 1) Generate story
    story_text = None
    try:
        story_text = call_text_generator(seed)
        logging.info("Generated story via API")
    except Exception as e:
        logging.info(f"Generator unavailable: {e}. Using local story.")
        story_text = pick_local_story(seed)

    # save story
    (OUTPUT / "story.txt").write_text(story_text, encoding="utf-8")

    # 2) SEO generation
    try:
        snippet = story_text[:1500]
        title, description, tags = call_seo_generator(seed, snippet)
    except Exception as e:
        logging.warning(f"SEO generator failed: {e}")
        title = f"Relaxing Rainy Story — {seed}"
        description = "A relaxing rainy story with ambient rain sounds."
        tags = ["rain","sleep","relax","asmr","story"]
    save_seo_files(title, description, tags)

    # 3) Thumbnail generation
    thumb_bytes = None
    try:
        thumb_bytes = call_image_generator(title, seed)
        if thumb_bytes:
            save_thumbnail_bytes(thumb_bytes, OUTPUT / "thumbnail.jpg")
    except Exception as e:
        logging.warning(f"Thumbnail generation failed: {e}")

    # 4) TTS generation (Coqui) or fallback
    voice_path = OUTPUT / "voice.wav"
    if args.no_tts:
        logging.info("no-tts requested, creating silent placeholder voice")
        make_silent_wave(min(1800, max(300, duration//3)), voice_path)
    else:
        if COQUI_AVAILABLE:
            try:
                tts_generate_coqui(story_text, voice_path, model_name=os.getenv("COQUI_MODEL", DEFAULT_COQUI_MODEL))
            except Exception as e:
                logging.warning(f"Coqui generation failed: {e}; falling back to silent voice")
                make_silent_wave(min(1800, max(300, duration//3)), voice_path)
        else:
            logging.warning("Coqui not available; creating silent voice placeholder")
            make_silent_wave(min(1800, max(300, duration//3)), voice_path)

    # 5) Prepare rain loop
    rain_src = Path(os.getenv("RAIN_FILE", str(DEFAULT_RAIN)))
    rain_loop = OUTPUT / "rain_loop.wav"
    if rain_src.exists():
        try:
            loop_audio_to_duration(rain_src, duration, rain_loop)
        except Exception as e:
            logging.warning(f"Looping rain failed: {e}; creating silent rain")
            make_silent_wave(duration, rain_loop)
    else:
        logging.warning("No rain asset found; creating silent rain")
        make_silent_wave(duration, rain_loop)

    # 6) Mix voice + rain -> final audio
    final_audio = OUTPUT / "final_audio.wav"
    mix_voice_and_rain(voice_path, rain_loop, duration, final_audio, voice_db=0.0, rain_db=-12.0)

    # 7) Render video
    out_video = OUTPUT / f"final_video_{uuid.uuid4().hex[:8]}.mp4"
    bg = Path(os.getenv("BG_IMAGE", str(DEFAULT_BG)))
    render_video(final_audio, out_video, duration, bg_image=bg if bg.exists() else None)

    # create stable final path
    final_link = OUTPUT / "final_video.mp4"
    if final_link.exists():
        final_link.unlink()
    out_video.rename(final_link)

    logging.info(f"Final video saved: {final_link}")
    logging.info(f"Started at {start}; finished at {datetime.datetime.utcnow().isoformat()}")

    # cleanup unless keep intermediate
    if not args.keep_intermediate:
        for f in OUTPUT.glob("voice_part_*.wav"):
            try:
                f.unlink()
            except Exception:
                pass

if __name__ == "__main__":
    main()
