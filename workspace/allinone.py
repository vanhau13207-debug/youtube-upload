#!/usr/bin/env python3
# allinone.py — FULL pipeline: Gemini Flash -> TTS -> mix -> render -> thumbnail from frame
import os
import sys
import json
import time
import base64
import uuid
import logging
import datetime
from pathlib import Path
from typing import List, Optional

import requests
import numpy as np
import soundfile as sf

from moviepy.editor import VideoFileClip, ImageClip, AudioFileClip, concatenate_videoclips

# Try Coqui TTS
try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except Exception:
    COQUI_AVAILABLE = False

# Paths
ROOT = Path.cwd()
WORKSPACE = ROOT / "workspace"
OUTPUT = WORKSPACE / "output"
ASSETS = ROOT / "assets"

OUTPUT.mkdir(parents=True, exist_ok=True)
WORKSPACE.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

# Config / env
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.0-flash")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.0-flash")
COQUI_MODEL = os.getenv("COQUI_MODEL", "tts_models/en/vctk/vits")
RAIN_MP3 = Path(os.getenv("RAIN_FILE", str(ASSETS / "rain.mp3")))
RAIN_BG_MP4 = Path(os.getenv("RAIN_BG", str(ASSETS / "rain_bg.mp4")))
DURATION = int(os.getenv("TARGET_DURATION", "7200"))  # seconds
THUMB_FRAME_AT = float(os.getenv("THUMB_FRAME_AT", "10.0"))  # seconds into video to grab frame
AUDIO_SR = int(os.getenv("AUDIO_SR", "22050"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------- Gemini text generation ----------
def gemini_text(prompt: str, model: str = None, key: Optional[str] = None, max_tokens: int = 2048) -> str:
    """Call Google Generative Language API (simple wrapper). Returns text."""
    key = key or GEMINI_KEY
    model = model or GEMINI_TEXT_MODEL
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateText"
    payload = {
        "prompt": {
            "text": prompt
        },
        "maxOutputTokens": max_tokens
    }
    r = requests.post(url, params={"key": key}, json=payload, timeout=60)
    r.raise_for_status()
    js = r.json()
    # parse common shape
    try:
        # Many responses: candidates -> content -> parts -> text OR candidates[0].content[0].text
        if "candidates" in js and js["candidates"]:
            cand = js["candidates"][0]
            # nested content parts
            content = cand.get("content")
            if isinstance(content, list) and content:
                # try to glue text parts
                texts = []
                for c in content:
                    if isinstance(c, dict) and "text" in c:
                        texts.append(c["text"])
                    elif isinstance(c, dict) and "type" in c and c["type"] == "output_text":
                        # alternate formats
                        if "text" in c:
                            texts.append(c["text"])
                if texts:
                    return "\n".join(texts)
        # fallback common field
        if "candidates" in js and js["candidates"]:
            cand = js["candidates"][0]
            if "content" in cand and isinstance(cand["content"], list) and cand["content"]:
                part = cand["content"][0]
                if isinstance(part, dict) and "text" in part:
                    return part["text"]
        # last resort: stringify
        return json.dumps(js)
    except Exception:
        return json.dumps(js)


# ---------- Gemini image generation from reference image (base64) ----------
def gemini_image_from_reference(img_b64: str, prompt_extra: str = "", model: str = None, key: Optional[str] = None) -> bytes:
    """Send reference image base64 to Gemini image endpoint to create thumbnail-like image.
       Returns raw image bytes (PNG/JPEG)."""
    key = key or GEMINI_KEY
    model = model or GEMINI_IMAGE_MODEL
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateImage"
    payload = {
        "prompt": {
            "text": (
                "Create a highly-clickable YouTube thumbnail for a chill/study/sleep rain ambience video. "
                "Enhance colors, contrast, composition; keep the original composition and subject recognizable; emphasize calm, cozy, cinematic vibes. "
                + prompt_extra
            )
        },
        # include reference image bytes
        "image_context": {
            "reference_images": [
                {
                    "image_bytes": img_b64
                }
            ]
        },
        "format": "PNG"
    }
    r = requests.post(url, params={"key": key}, json=payload, timeout=90)
    r.raise_for_status()
    js = r.json()
    # expect images[0].imageBytes as base64
    try:
        b64 = js["images"][0]["imageBytes"]
        return base64.b64decode(b64)
    except Exception as e:
        raise RuntimeError(f"Unexpected Gemini image response: {e} -- {js}")


# ---------- Helpers for SEO files ----------
def save_seo_files(title: str, description: str, tags: List[str]):
    (OUTPUT / "title.txt").write_text(title or "", encoding="utf-8")
    (OUTPUT / "description.txt").write_text(description or "", encoding="utf-8")
    (OUTPUT / "tags.txt").write_text(",".join(tags or []), encoding="utf-8")


# ---------- Coqui TTS generation ----------
def generate_tts_coqui(text: str, out_wav: Path, model: str = COQUI_MODEL):
    if not COQUI_AVAILABLE:
        raise RuntimeError("Coqui TTS not available (install TTS)")
    logging.info(f"Generating TTS with model {model}")
    tts = TTS(model)
    # chunk text into reasonable parts to avoid OOM
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
        tmp = out_wav.parent / f"voice_part_{i}.wav"
        logging.info(f"TTS chunk {i+1}/{len(parts)} ({len(p)} chars)")
        tts.tts_to_file(text=p, file_path=str(tmp))
        tmp_files.append(tmp)
    # concatenate parts
    arrays = []
    sr = None
    for f in tmp_files:
        a, s = sf.read(str(f), dtype="float32")
        if sr is None:
            sr = s
        arrays.append(a)
    if not arrays:
        raise RuntimeError("No TTS output")
    full = np.concatenate(arrays, axis=0)
    sf.write(str(out_wav), full, int(sr or AUDIO_SR))
    # cleanup parts
    for f in tmp_files:
        try:
            f.unlink()
        except Exception:
            pass
    logging.info(f"TTS saved to {out_wav}")


# ---------- silent fallback ----------
def make_silent_wave(seconds: int, out_wav: Path, sr: int = AUDIO_SR):
    total = int(seconds * sr)
    data = np.zeros((total,), dtype='float32')
    sf.write(str(out_wav), data, sr)
    logging.info(f"Created silent wave: {out_wav} ({seconds}s)")


# ---------- extract frame from mp4 ----------
def extract_frame(mp4_path: Path, at_time: float, out_png: Path):
    if not mp4_path.exists():
        raise RuntimeError(f"Video background {mp4_path} not found")
    clip = VideoFileClip(str(mp4_path))
    # clamp at_time into duration
    t = min(max(0.0, at_time), clip.duration - 0.001)
    frame = clip.get_frame(t)
    try:
        from PIL import Image
        im = Image.fromarray(frame)
        im.save(out_png)
    finally:
        try:
            clip.reader.close()
            clip.audio = None
        except Exception:
            pass
    logging.info(f"Extracted frame at {t}s -> {out_png}")


# ---------- loop audio to duration ----------
def loop_audio(src_audio: Path, duration_s: int, out_wav: Path):
    if not src_audio.exists():
        raise RuntimeError(f"Source audio {src_audio} missing")
    data, sr = sf.read(str(src_audio), dtype='float32')
    if data.ndim > 1:
        data = data.mean(axis=1)
    need = int(duration_s * sr)
    if len(data) == 0:
        raise RuntimeError("Source audio empty")
    reps = (need // len(data)) + 1
    full = np.tile(data, reps)[:need]
    sf.write(str(out_wav), full, sr)
    logging.info(f"Looped {src_audio} to {duration_s}s -> {out_wav}")


# ---------- mix voice + rain ----------
def mix_voice_and_rain(voice_wav: Path, rain_wav: Path, duration_s: int, out_wav: Path, voice_db=0.0, rain_db=-12.0):
    v, sr_v = sf.read(str(voice_wav), dtype='float32')
    r, sr_r = sf.read(str(rain_wav), dtype='float32')
    if v.ndim > 1:
        v = v.mean(axis=1)
    if r.ndim > 1:
        r = r.mean(axis=1)
    sr = sr_r
    if sr_v != sr_r:
        # resample v -> sr_r (naive)
        v = np.interp(np.linspace(0, len(v), int(len(v) * sr_r / sr_v), endpoint=False), np.arange(len(v)), v)
    need = int(duration_s * sr)
    if len(r) < need:
        reps = (need // len(r)) + 1
        r = np.tile(r, reps)[:need]
    else:
        r = r[:need]
    if len(v) < need:
        v = np.concatenate([v, np.zeros(need - len(v), dtype='float32')])
    else:
        v = v[:need]
    def dbmul(db): return 10 ** (db / 20.0)
    v = v * dbmul(voice_db)
    r = r * dbmul(rain_db)
    mixed = v + r
    mx = np.max(np.abs(mixed)) if mixed.size else 1.0
    if mx > 1.0:
        mixed = mixed / mx
    sf.write(str(out_wav), mixed, sr)
    logging.info(f"Mixed audio saved to {out_wav}")


# ---------- render final video using rain_bg.mp4 as visual ----------
def render_with_background(video_bg: Path, audio_wav: Path, out_mp4: Path, duration_s: int):
    bg = VideoFileClip(str(video_bg))
    # loop or trim bg to duration_s
    if bg.duration < duration_s:
        cnt = int(np.ceil(duration_s / bg.duration))
        clips = [bg.copy() for _ in range(cnt)]
        bg_final = concatenate_videoclips(clips).subclip(0, duration_s)
    else:
        bg_final = bg.subclip(0, duration_s)
    audio = AudioFileClip(str(audio_wav))
    bg_final = bg_final.set_audio(audio)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # write with decent settings
    bg_final.write_videofile(str(out_mp4), codec="libx264", audio_codec="aac", fps=24, threads=2, verbose=False, logger=None)
    try:
        bg.reader.close()
    except Exception:
        pass
    logging.info(f"Rendered final video to {out_mp4}")


# ---------- helper: safe JSON parse ----------
def try_parse_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        # sometimes Gemini returns trailing text, try to find first { ... }
        try:
            start = s.index("{")
            end = s.rindex("}") + 1
            return json.loads(s[start:end])
        except Exception:
            return None


# ---------- Main pipeline ----------
def main():
    logging.info("=== allinone pipeline start ===")
    start_time = datetime.datetime.utcnow().isoformat()

    # 1) Generate story via Gemini (English, chill)
    seed = os.getenv("VIDEO_SEED", "Cozy rainy focus")
    story_prompt = (
        f"Write a long, slow, soothing English story in a calm, chill tone suitable for listening while studying, "
        f"working, or falling asleep. Topic: {seed}. Use gentle language, clear paragraphs, and create about 40-60 minutes of natural-sounding narration. "
        "Output plain text."
    )
    try:
        story_text = gemini_text(story_prompt)
        logging.info("Gemini story generated")
    except Exception as e:
        logging.warning(f"Gemini story failed: {e}; using fallback local story")
        # fallback: simple repeated base
        base = ("The rain whispered against the window as the town settled into a slow, measured breath. "
                "Tonight, the lamps glowed softly and the streets remembered the footsteps of those who passed. "
                "You breathe in time with the rain and let the world blur at the edges...")
        story_text = "\n\n".join([base for _ in range(60)])

    (OUTPUT / "story.txt").write_text(story_text, encoding="utf-8")

    # 2) Generate SEO (title, description, tags) via Gemini
    seo_prompt = (
        "Create a YouTube SEO-optimized title (short & compelling), a long description (~300-500 words) with keywords, "
        "and an array of 15-25 concise tags relevant to a chill/study/sleep rain ambience storytelling video. "
        "Return strictly JSON: {\"title\":\"...\",\"description\":\"...\",\"tags\":[\"tag1\",\"tag2\",...]} .\n\n"
        f"Input story excerpt:\n{story_text[:1500]}"
    )
    title = ""
    description = ""
    tags: List[str] = []
    try:
        seo_raw = gemini_text(seo_prompt)
        parsed = try_parse_json(seo_raw)
        if parsed:
            title = parsed.get("title", "")
            description = parsed.get("description", "")
            tags = parsed.get("tags", []) or []
        else:
            logging.warning("Could not parse SEO JSON; using fallback")
    except Exception as e:
        logging.warning(f"Gemini SEO failed: {e}")

    if not title:
        title = f"Relaxing Rain Story — {seed}"
    if not description:
        description = "Relaxing rainy ambience with a soothing narrated story to help you focus, relax, or sleep."
    if not tags:
        tags = ["rain", "relax", "sleep", "study", "ambient", "story"]

    save_seo_files(title, description, tags)
    logging.info("SEO saved")

    # 3) Extract frame from rain_bg.mp4 and create thumbnail via Gemini image (base on frame)
    frame_png = OUTPUT / "thumb_frame.png"
    thumb_out = OUTPUT / "thumbnail.jpg"
    try:
        extract_frame(RAIN_BG_MP4, THUMB_FRAME_AT, frame_png)
        with open(frame_png, "rb") as f:
            frame_b64 = base64.b64encode(f.read()).decode("utf-8")
        try:
            thumb_bytes = gemini_image_from_reference(frame_b64, prompt_extra="Make it a high-CTR YouTube thumbnail, enhance but keep original composition.")
            thumb_out.write_bytes(thumb_bytes)
            logging.info("Thumbnail generated by Gemini and saved")
        except Exception as e:
            logging.warning(f"Gemini thumbnail failed: {e}; using frame as thumbnail")
            thumb_out.write_bytes(frame_png.read_bytes())
    except Exception as e:
        logging.warning(f"Frame extraction or thumbnail generation failed: {e}")
        # if anything fails, ensure a thumbnail placeholder exists (use black image)
        try:
            from PIL import Image, ImageDraw
            im = Image.new("RGB", (1280, 720), (20, 20, 30))
            d = ImageDraw.Draw(im)
            d.text((50, 300), title[:60], fill=(220, 220, 220))
            im.save(thumb_out)
            logging.info("Created fallback thumbnail")
        except Exception:
            logging.error("Failed to create fallback thumbnail")

    # 4) TTS generation (Coqui) or fallback silent wave
    voice_wav = OUTPUT / "voice.wav"
    try:
        if COQUI_AVAILABLE:
            generate_tts_coqui(story_text, voice_wav, model=COQUI_MODEL)
        else:
            raise RuntimeError("Coqui not available")
    except Exception as e:
        logging.warning(f"TTS failed: {e}; creating silent voice placeholder")
        make_silent_wave(min(1800, max(300, DURATION // 3)), voice_wav)

    # 5) Prepare rain audio loop
    rain_loop = OUTPUT / "rain_loop.wav"
    try:
        if RAIN_MP3.exists():
            loop_audio(RAIN_MP3, DURATION, rain_loop)
        else:
            logging.warning(f"{RAIN_MP3} not found; creating silent rain")
            make_silent_wave(DURATION, rain_loop)
    except Exception as e:
        logging.warning(f"Looping rain failed: {e}")
        make_silent_wave(DURATION, rain_loop)

    # 6) Mix voice + rain -> final_audio
    final_audio = OUTPUT / "final_audio.wav"
    try:
        mix_voice_and_rain(voice_wav, rain_loop, DURATION, final_audio, voice_db=0.0, rain_db=-12.0)
    except Exception as e:
        logging.warning(f"Mix failed: {e}; using voice only or rain only fallback")
        try:
            if voice_wav.exists():
                sf.write(str(final_audio), sf.read(str(voice_wav))[0], AUDIO_SR)
            else:
                make_silent_wave(DURATION, final_audio)
        except Exception:
            make_silent_wave(DURATION, final_audio)

    # 7) Render final video with rain_bg.mp4 visual and final_audio
    final_video = OUTPUT / "final_video.mp4"
    try:
        render_with_background(RAIN_BG_MP4, final_audio, final_video, DURATION)
    except Exception as e:
        logging.error(f"Render failed: {e}")
        # try a lightweight fallback: static image + audio
        try:
            # if thumbnail exists use it as image background
            if thumb_out.exists():
                img_path = thumb_out
            else:
                # create tiny placeholder
                from PIL import Image, ImageDraw
                ph = OUTPUT / "bg_placeholder.jpg"
                im = Image.new("RGB", (1280, 720), (10, 10, 20))
                d = ImageDraw.Draw(im)
                d.text((50, 300), title[:60], fill=(200, 200, 220))
                im.save(ph)
                img_path = ph
            clip = ImageClip(str(img_path)).set_duration(DURATION)
            audio = AudioFileClip(str(final_audio))
            clip = clip.set_audio(audio)
            clip.write_videofile(str(final_video), fps=1, codec="libx264", audio_codec="aac", threads=2, verbose=False, logger=None)
            logging.info("Fallback static render complete")
        except Exception as e2:
            logging.error(f"Fallback render also failed: {e2}")
            raise

    # Done
    end_time = datetime.datetime.utcnow().isoformat()
    logging.info("=== Pipeline complete ===")
    logging.info(f"Start: {start_time}  End: {end_time}")
    logging.info("Outputs:")
    for p in OUTPUT.glob("*"):
        logging.info(f" - {p.name}")

    # keep files for upload: final_video.mp4, thumbnail.jpg, title.txt, description.txt, tags.txt, story.txt
    return


if __name__ == "__main__":
    main()
