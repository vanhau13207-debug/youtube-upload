#!/usr/bin/env python3
# allinone.py — FULL pipeline (part 1/4)
# - multi-batch Gemini story generation
# - estimate TTS duration, ensure video >= 2h OR = story length if story longer
# - Coqui TTS integration with fallback silent wave
# - helpers for image generation and SEO
# NOTE: this is part 1 of 4 — get part 2 after this

import os
import sys
import json
import time
import base64
import uuid
import logging
import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import numpy as np
import soundfile as sf

from moviepy.editor import VideoFileClip, ImageClip, AudioFileClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont

# Try Coqui TTS
try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except Exception:
    COQUI_AVAILABLE = False

# -------------------------
# PATHS (workspace/assets)
# -------------------------
ROOT = Path.cwd()
WORKSPACE = ROOT / "workspace"
OUTPUT = WORKSPACE / "output"
ASSETS = WORKSPACE / "assets"   # IMPORTANT: assets inside workspace/

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
MIN_DURATION = int(os.getenv("MIN_DURATION", "7200"))  # seconds, default 2 hours
THUMB_FRAME_AT = float(os.getenv("THUMB_FRAME_AT", "10.0"))  # seconds into video to grab frame
AUDIO_SR = int(os.getenv("AUDIO_SR", "22050"))
# TTS words-per-minute assumption for estimation (approx)
TTS_WPM = int(os.getenv("TTS_WPM", "150"))

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
        if "candidates" in js and js["candidates"]:
            cand = js["candidates"][0]
            content = cand.get("content")
            if isinstance(content, list) and content:
                texts = []
                for c in content:
                    if isinstance(c, dict) and "text" in c:
                        texts.append(c["text"])
                if texts:
                    return "\n".join(texts)
        if "candidates" in js and js["candidates"]:
            cand = js["candidates"][0]
            if "content" in cand and isinstance(cand["content"], list) and cand["content"]:
                part = cand["content"][0]
                if isinstance(part, dict) and "text" in part:
                    return part["text"]
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


# ---------- Estimate TTS duration and multi-batch story generation ----------
def estimate_tts_duration_seconds(text: str, wpm: int = TTS_WPM) -> int:
    """Estimate seconds of TTS audio from text using words-per-minute."""
    words = len(text.split())
    minutes = words / float(wpm) if wpm > 0 else words / 150.0
    return int(minutes * 60)


def generate_full_story(seed: str, min_duration_s: int, per_request_words: int = 1400, max_batches: int = 12) -> Tuple[str, int]:
    """
    Generate a full story by calling Gemini multiple times until estimated TTS duration >= min_duration_s.
    Returns (full_story_text, estimated_seconds).
    Note: we DON'T cut story if it exceeds min_duration; we allow story to exceed and video will match it.
    """
    logging.info(f"Generating story (min_duration={min_duration_s}s) for seed: {seed}")
    full = ""
    attempts = 0

    base_prompt = (
        f"Write a long, slow-paced, atmospheric English story for sleep/study. "
        f"Topic: {seed}. Calm, descriptive language. Aim for roughly {per_request_words} words. "
        "Do not include titles or chapter numbers. Provide natural paragraphs suitable for narration."
    )

    while True:
        attempts += 1
        try:
            part = gemini_text(base_prompt)
        except Exception as e:
            logging.warning(f"Gemini fetch failed on attempt {attempts}: {e}")
            part = ""
        if not part.strip():
            # fallback minimal paragraph to avoid infinite loop
            fallback = ("The rain continued its soft, endless pattern, and the narrator guided you through quiet streets and warm rooms.")
            part = fallback

        full += ("\n\n" + part).strip()
        est_secs = estimate_tts_duration_seconds(full)
        logging.info(f"Batch {attempts}: +{len(part.split())} words → estimated total {est_secs}s")

        # If story already exceeds min_duration or we've reached safe batch limit, stop
        if est_secs >= min_duration_s:
            logging.info("Reached minimum target duration based on estimated TTS length.")
            break
        if attempts >= max_batches:
            logging.info("Reached max_batches limit; stopping story generation.")
            break
        # small sleep to avoid hammering API
        time.sleep(0.5)

    return full, est_secs


# ---------- Get actual WAV duration ----------
def get_wav_duration(wav_path: Path) -> float:
    """Return duration (seconds) of wav file or 0.0 if unreadable."""
    try:
        if not wav_path.exists():
            return 0.0
        data, sr = sf.read(str(wav_path), dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1)
        return len(data) / float(sr)
    except Exception:
        return 0.0


# ---------- Coqui TTS generation ----------
def generate_tts_coqui(text: str, out_wav: Path, model: str = COQUI_MODEL):
    if not COQUI_AVAILABLE:
        raise RuntimeError("Coqui TTS not available (install TTS)")
    logging.info(f"Generating TTS with model {model}")
    try:
        tts = TTS(model)
    except Exception as e:
        raise RuntimeError(f"Failed to init TTS model {model}: {e}")
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
        try:
            tts.tts_to_file(text=p, file_path=str(tmp))
        except Exception as e:
            logging.warning(f"TTS chunk failed: {e}")
            raise
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
from PIL import Image

def extract_frame(mp4_path: Path, at_time: float, out_png: Path):
    """Extract frame at <at_time> from mp4_path and save to out_png."""
    if not mp4_path.exists():
        raise RuntimeError(f"Video background {mp4_path} not found")

    clip = VideoFileClip(str(mp4_path))
    t = min(max(0.0, at_time), clip.duration - 0.001)

    frame = clip.get_frame(t)
    img = Image.fromarray(frame)
    img.save(out_png)

    try:
        clip.reader.close()
        clip.audio = None
    except Exception:
        pass

    logging.info(f"Extracted frame at {t}s → {out_png}")



# ---------- loop audio to duration ----------
def loop_audio(src_audio: Path, duration_s: int, out_wav: Path):
    """Loop audio until EXACT duration_s seconds."""
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

    logging.info(f"Looped {src_audio} to {duration_s}s → {out_wav}")


# ---------- mix voice + rain ----------
def mix_voice_and_rain(
    voice_wav: Path, rain_wav: Path, duration_s: int, out_wav: Path,
    voice_db=0.0, rain_db=-12.0
):
    """Mix voice + rain to equal duration."""
    v, sr_v = sf.read(str(voice_wav), dtype='float32')
    r, sr_r = sf.read(str(rain_wav), dtype='float32')

    if v.ndim > 1:
        v = v.mean(axis=1)
    if r.ndim > 1:
        r = r.mean(axis=1)

    sr = sr_r

    # Resample voice if different SR
    if sr_v != sr_r:
        v = np.interp(
            np.linspace(0, len(v), int(len(v) * sr_r / sr_v), endpoint=False),
            np.arange(len(v)),
            v
        )

    need = int(duration_s * sr)

    # Loop rain
    if len(r) < need:
        reps = (need // len(r)) + 1
        r = np.tile(r, reps)[:need]
    else:
        r = r[:need]

    # Expand voice to duration
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
    """Render final MP4 with background video (loop/trim) and full audio."""
    if not video_bg.exists():
        raise RuntimeError(f"Background video missing: {video_bg}")

    bg = VideoFileClip(str(video_bg))

    if bg.duration < duration_s:
        cnt = int(np.ceil(duration_s / bg.duration))
        clips = [bg.copy() for _ in range(cnt)]
        bg_final = concatenate_videoclips(clips).subclip(0, duration_s)
    else:
        bg_final = bg.subclip(0, duration_s)

    audio = AudioFileClip(str(audio_wav))
    bg_final = bg_final.set_audio(audio)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    bg_final.write_videofile(
        str(out_mp4),
        codec="libx264",
        audio_codec="aac",
        fps=24,
        threads=2,
        verbose=False,
        logger=None
    )

    try:
        bg.reader.close()
    except Exception:
        pass

    logging.info(f"Rendered final video → {out_mp4}")


# ---------- JSON parse helper ----------
def try_parse_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        try:
            start = s.index("{")
            end = s.rindex("}") + 1
            return json.loads(s[start:end])
        except Exception:
            return None
# ---------- Thumbnail fallback helper (unicode-safe) ----------
def create_fallback_thumbnail(text: str, out_path: Path, size=(1280, 720)):
    """Create a simple thumbnail image with outlined text (unicode-safe)."""
    try:
        im = Image.new("RGB", size, (20, 20, 30))
        d = ImageDraw.Draw(im)
        # Prefer DejaVuSans which exists on ubuntu runners
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 48)
        except Exception:
            font = ImageFont.load_default()
        x, y = 60, 260
        for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
            d.text((x+dx, y+dy), text[:80], font=font, fill=(0,0,0))
        d.text((x, y), text[:80], font=font, fill=(230,230,230))
        im.save(out_path)
        logging.info(f"Created fallback thumbnail {out_path}")
    except Exception as e:
        logging.error(f"Failed to create fallback thumbnail: {e}")


# ---------- MAIN pipeline (part start) ----------
def main():
    logging.info("=== allinone pipeline start ===")
    start_time = datetime.datetime.utcnow().isoformat()

    seed = os.getenv("VIDEO_SEED", "Cozy rainy focus")
    logging.info(f"Seed/topic: {seed}")

    # -------------------------------
    # 1) Generate story (multi-batch) until estimated >= MIN_DURATION (but do NOT cut if exceeds)
    # -------------------------------
    try:
        story_text, est_secs = generate_full_story(seed, MIN_DURATION)
        logging.info("Story generation complete")
    except Exception as e:
        logging.warning(f"generate_full_story failed: {e}; falling back to short repeated base")
        base = ("The rain whispered against the window as the town settled into a slow, measured breath. "
                "Tonight, the lamps glowed softly and the streets remembered the footsteps of those who passed.")
        story_text = "\n\n".join([base for _ in range(60)])
        est_secs = estimate_tts_duration_seconds(story_text)

    # always persist story text
    (OUTPUT / "story.txt").write_text(story_text, encoding="utf-8")

    # -------------------------------
    # 2) Generate SEO (title/description/tags) via Gemini
    # -------------------------------
    title = ""
    description = ""
    tags: List[str] = []
    try:
        seo_prompt = (
            "Create a YouTube SEO-optimized title (short & compelling), a long description (~300-500 words) with keywords, "
            "and an array of 15-25 concise tags relevant to a chill/study/sleep rain ambience storytelling video. "
            "Return strictly JSON: {\"title\":\"...\",\"description\":\"...\",\"tags\":[\"t1\",\"t2\",...]}.\n\n"
            f"Story excerpt:\n{story_text[:1600]}"
        )
        seo_raw = gemini_text(seo_prompt)
        parsed = try_parse_json(seo_raw)
        if parsed:
            title = parsed.get("title", "") or ""
            description = parsed.get("description", "") or ""
            tags = parsed.get("tags", []) or []
            logging.info("SEO generated by Gemini")
        else:
            logging.warning("Could not parse SEO JSON from Gemini response")
    except Exception as e:
        logging.warning(f"Gemini SEO error: {e}")

    if not title:
        title = f"Relaxing Rain Story — {seed}"
    if not description:
        description = "Relaxing rainy ambience with a soothing narrated story to help you focus, relax, or sleep."
    if not tags:
        tags = ["rain","relax","sleep","study","ambient","story"]

    save_seo_files(title, description, tags)

    # -------------------------------
    # 3) Prepare thumbnail: extract frame + ask Gemini to enhance (fallback to simple image)
    # -------------------------------
    frame_png = OUTPUT / "thumb_frame.png"
    thumb_out = OUTPUT / "thumbnail.jpg"
    try:
        extract_frame(RAIN_BG_MP4, THUMB_FRAME_AT, frame_png)
        # call Gemini image-from-reference using base64
        with open(frame_png, "rb") as f:
            frame_b64 = base64.b64encode(f.read()).decode("utf-8")
        try:
            thumb_bytes = gemini_image_from_reference(frame_b64, prompt_extra="Make it high-CTR, keep composition, emphasize cozy vibe.")
            thumb_out.write_bytes(thumb_bytes)
            logging.info("Thumbnail created by Gemini")
        except Exception as e:
            logging.warning(f"Gemini thumbnail failed: {e}; using extracted frame as thumbnail")
            thumb_out.write_bytes(frame_png.read_bytes())
    except Exception as e:
        logging.warning(f"Frame extract / thumbnail generation failed: {e}")
        create_fallback_thumbnail(title, thumb_out)

    # -------------------------------
    # 4) Generate TTS for entire story (do NOT cut story)
    # -------------------------------
    voice_wav = OUTPUT / "voice.wav"
    try:
        if COQUI_AVAILABLE:
            generate_tts_coqui(story_text, voice_wav, model=COQUI_MODEL)
        else:
            raise RuntimeError("Coqui TTS not available")
    except Exception as e:
        logging.warning(f"TTS generation failed: {e}; creating silent placeholder")
        # create a reasonable-length silent voice equal to estimated seconds (or MIN_DURATION/3 fallback)
        create_secs = int(max(300, min(est_secs or 600, MIN_DURATION)))
        make_silent_wave(create_secs, voice_wav)

    # compute actual voice duration (seconds)
    voice_duration = int(get_wav_duration(voice_wav))
    logging.info(f"Voice duration (actual WAV) = {voice_duration}s (estimated was {est_secs}s)")

    # -------------------------------
    # 5) Determine final duration: video must be long enough to include full voice
    # -------------------------------
    final_duration = max(MIN_DURATION, voice_duration)
    logging.info(f"Final video duration set to {final_duration}s (min {MIN_DURATION}s, voice {voice_duration}s)")

    # persist some info
    (OUTPUT / "meta.json").write_text(json.dumps({
        "seed": seed,
        "title": title,
        "voice_duration": voice_duration,
        "final_duration": final_duration
    }, indent=2), encoding="utf-8")

    # -------------------------------
    # 6) Prepare rain audio loop to final_duration
    # -------------------------------
    rain_loop = OUTPUT / "rain_loop.wav"
    try:
        if RAIN_MP3.exists():
            loop_audio(RAIN_MP3, final_duration, rain_loop)
        else:
            logging.warning(f"{RAIN_MP3} missing; creating silent rain")
            make_silent_wave(final_duration, rain_loop)
    except Exception as e:
        logging.warning(f"Looping rain failed: {e}; creating silent rain")
        make_silent_wave(final_duration, rain_loop)

    # -------------------------------
    # 7) Mix voice + rain -> final_audio (voice not looped; rain looped to cover entire duration)
    # -------------------------------
    final_audio = OUTPUT / "final_audio.wav"
    try:
        mix_voice_and_rain(voice_wav, rain_loop, final_duration, final_audio, voice_db=0.0, rain_db=-12.0)
    except Exception as e:
        logging.warning(f"Mixing failed: {e}; attempting fallback to voice-only or silent")
        try:
            if voice_wav.exists():
                data, sr = sf.read(str(voice_wav), dtype='float32')
                sf.write(str(final_audio), data, sr)
            else:
                make_silent_wave(final_duration, final_audio)
        except Exception:
            make_silent_wave(final_duration, final_audio)

    # After this point we have:
    # - OUTPUT/story.txt
    # - OUTPUT/voice.wav
    # - OUTPUT/rain_loop.wav
    # - OUTPUT/final_audio.wav
    # - OUTPUT/thumbnail.jpg
    # - OUTPUT/title.txt, description.txt, tags.txt
    # -------------------------------
    # 8) Render final video with rain_bg.mp4 visual
    # -------------------------------
    final_video = OUTPUT / "final_video.mp4"
    try:
        render_with_background(RAIN_BG_MP4, final_audio, final_video, final_duration)
    except Exception as e:
        logging.error(f"Full video render failed: {e}")
        # fallback: static thumbnail video
        try:
            fallback_img = thumb_out if thumb_out.exists() else frame_png
            clip = ImageClip(str(fallback_img)).set_duration(final_duration)
            audio = AudioFileClip(str(final_audio))
            clip = clip.set_audio(audio)
            clip.write_videofile(
                str(final_video),
                fps=1,
                codec="libx264",
                audio_codec="aac",
                threads=2,
                verbose=False,
                logger=None
            )
            logging.info("Fallback static video render COMPLETE")
        except Exception as e2:
            logging.error(f"Fallback static render failed: {e2}")
            raise

    # -------------------------------
    # DONE
    # -------------------------------
    end_time = datetime.datetime.utcnow().isoformat()
    logging.info("=== Pipeline complete ===")
    logging.info(f"Start: {start_time}  End: {end_time}")
    logging.info("Outputs:")
    for p in OUTPUT.glob("*"):
        logging.info(f" - {p.name}")

    # Keep these final files for uploader:
    # final_video.mp4
    # thumbnail.jpg
    # title.txt
    # description.txt
    # tags.txt
    # story.txt

    return


if __name__ == "__main__":
    main()





