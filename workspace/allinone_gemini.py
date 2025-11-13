#!/usr/bin/env python3
"""
allinone.py
Full pipeline:
  1) generate or pick a story (from Gemini if configured, else local fallback)
  2) generate TTS audio via Coqui TTS (if installed/configured), else fallback silent voice
  3) mix voice + rain bed to target duration
  4) render MP4 (static image + audio) via moviepy (ffmpeg)
  5) output SEO files (title.txt, description.txt, tags.txt) in workspace/output/
Usage:
  python workspace/allinone.py --seed "cozy rainy night" --duration 7200
Environment:
  - OPENAI_API_KEY or GEMINI_API_KEY (optional) to generate story
  - paths: assets/rain.mp3, assets/bg.jpg (optional)
  - outputs-> workspace/output/
Notes:
  - Designed to run in CI: fast muxing (no heavy video processing).
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
from typing import Optional

# audio libs
try:
    from TTS.api import TTS  # coqui tts
    COQUI_AVAILABLE = True
except Exception:
    COQUI_AVAILABLE = False

import numpy as np
import soundfile as sf

# moviepy for muxing
from moviepy.editor import ImageClip, AudioFileClip, concatenate_audioclips, CompositeAudioClip

# small helper to call external generator if present
import requests

# Setup
ROOT = Path.cwd()
WORKSPACE = ROOT / "workspace"
STORIES = WORKSPACE / "stories"
ASSETS = ROOT / "assets"
OUTPUT = WORKSPACE / "output"
RAIN_PATH = ASSETS / "rain.mp3"
BG_IMAGE = ASSETS / "bg.jpg"

# Ensure dirs
OUTPUT.mkdir(parents=True, exist_ok=True)
STORIES.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def generate_story_via_gemini(seed: Optional[str]) -> str:
    """Try calling Gemini-like API if GEMINI_API_KEY env is present.
    This is a best-effort: if no key or request fails, raise RuntimeError."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not key:
        raise RuntimeError("No GEMINI/OPENAI key found")
    prompt = (seed or "A calm cozy rainy night story to relax and sleep") + "\n\nWrite a long, slow, soothing narrated story suitable for a 2-hour ambient video. Use gentle language."
    # NOTE: this is a placeholder: user should replace with their provider call.
    # Attempt a generic POST to a user-provided endpoint
    endpoint = os.getenv("GENERATOR_ENDPOINT")  # optional custom endpoint
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if endpoint:
        payload = {"prompt": prompt, "max_tokens": 3000}
        r = requests.post(endpoint, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        # Expect text in data['text'] or data['choices'][0]['text']
        text = data.get("text") or (data.get("choices") and data["choices"][0].get("text"))
        if not text:
            raise RuntimeError("Generator returned unexpected payload")
        return text
    # If no endpoint, raise
    raise RuntimeError("No generator endpoint configured")


def pick_local_story(seed: Optional[str]) -> str:
    # If there are story files in STORIES, pick random one, else make a filler
    files = list(STORIES.glob("*.txt"))
    if files:
        chosen = random.choice(files)
        logging.info(f"Using local story file: {chosen}")
        return chosen.read_text(encoding="utf-8")
    # fallback: generate repeated paragraph to reach longer length
    base = (
        "The rain whispered against the window as the town settled into a slow, measured breath. "
        "Tonight, the lamps glowed softly and the streets remembered the footsteps of those who passed. "
        "You breathe in time with the rain and let the world blur at the edges..."
    )
    # repeat to create long text (note: TTS will be looped or slowed)
    return "\n\n".join([base for _ in range(40)])


def save_seo(title: str, description: str, tags: list[str]):
    (OUTPUT / "title.txt").write_text(title, encoding="utf-8")
    (OUTPUT / "description.txt").write_text(description, encoding="utf-8")
    (OUTPUT / "tags.txt").write_text(",".join(tags), encoding="utf-8")
    logging.info("Saved SEO files.")


def tts_coqui_generate(text: str, out_path: Path) -> None:
    """Generate TTS using Coqui TTS API (local). Will attempt streaming model 'tts_models/en/ljspeech/glow-tts' or fallback."""
    if not COQUI_AVAILABLE:
        raise RuntimeError("Coqui TTS not available")
    # choose a model that is likely to work offline; user can change
    model_name = os.getenv("COQUI_MODEL", "tts_models/en/vctk/vits")
    logging.info(f"Using Coqui TTS model: {model_name}")
    tts = TTS(model_name)
    # split text to chunks to avoid memory spikes
    chunks = []
    max_chars = 2500
    parts = []
    cur = ""
    for line in text.splitlines():
        if len(cur) + len(line) + 1 > max_chars:
            parts.append(cur)
            cur = line
        else:
            cur = (cur + "\n" + line).strip()
    if cur:
        parts.append(cur)
    # generate each part and append
    tmp_files = []
    for i, p in enumerate(parts):
        tmp = OUTPUT / f"voice_part_{i}.wav"
        logging.info(f"Generating TTS chunk {i+1}/{len(parts)} ({len(p)} chars)")
        tts.tts_to_file(text=p, file_path=str(tmp))
        tmp_files.append(str(tmp))
    # concatenate into out_path using soundfile
    # read all and concatenate arrays
    data_list = []
    sr = None
    for f in tmp_files:
        d, s = sf.read(f, dtype='float32')
        if sr is None:
            sr = s
        if s != sr:
            raise RuntimeError("Sample rate mismatch in tts parts")
        data_list.append(d)
    if not data_list:
        raise RuntimeError("No TTS data generated")
    full = np.concatenate(data_list, axis=0)
    sf.write(str(out_path), full, sr)
    # cleanup tmp
    for f in tmp_files:
        try:
            os.remove(f)
        except Exception:
            pass
    logging.info(f"TTS written to {out_path}")


def make_silent_voice(seconds: int, out_path: Path, sr=22050):
    """Create silent wave of given seconds as fallback voice"""
    total = int(seconds * sr)
    data = np.zeros((total,), dtype='float32')
    sf.write(str(out_path), data, sr)
    logging.info(f"Created silent voice placeholder {out_path} ({seconds}s)")


def loop_audio_to_duration(src_path: Path, duration_s: int, out_path: Path):
    """Loop src_path (rain or voice) to exactly duration_s and write to out_path."""
    # Use soundfile + numpy to stitch loops (avoid re-encoding)
    data, sr = sf.read(str(src_path), dtype='float32')
    if data.ndim > 1:
        data = data.mean(axis=1)  # to mono
    needed = int(duration_s * sr)
    chunks = []
    idx = 0
    while idx < needed:
        take = min(len(data), needed - idx)
        chunks.append(data[:take])
        idx += take
    full = np.concatenate(chunks, axis=0)
    sf.write(str(out_path), full, sr)
    logging.debug(f"Looped {src_path} to {out_path} ({duration_s}s)")


def mix_voice_and_rain(voice_path: Path, rain_path: Path, duration_s: int, out_path: Path, voice_gain_db: float = 0.0, rain_gain_db: float = -10.0):
    """Mix voice (mono) and rain (mono/stereo) to reach duration and export as out_path wav."""
    # load
    v, sr_v = sf.read(str(voice_path), dtype='float32')
    r, sr_r = sf.read(str(rain_path), dtype='float32')
    if v.ndim > 1:
        v = v.mean(axis=1)
    if r.ndim > 1:
        r = r.mean(axis=1)
    # resample if needed (simple method: expect same sr)
    if sr_v != sr_r:
        logging.warning("Sampling rates differ; resampling voice to rain sample rate")
        # naive resample via numpy (not high quality). Better to require same sr.
        import math
        factor = sr_r / sr_v
        new_len = int(len(v) * factor)
        v = np.interp(np.linspace(0, len(v), new_len, endpoint=False), np.arange(len(v)), v)
        sr = sr_r
    else:
        sr = sr_v
    # loop rain to duration
    needed = int(duration_s * sr)
    if len(r) < needed:
        reps = (needed // len(r)) + 1
        r = np.tile(r, reps)[:needed]
    else:
        r = r[:needed]
    # make voice same length
    if len(v) < needed:
        v = np.concatenate([v, np.zeros(needed - len(v), dtype='float32')])
    else:
        v = v[:needed]
    # apply gains
    def db_to_mul(db): return 10 ** (db / 20.0)
    v = v * db_to_mul(voice_gain_db)
    r = r * db_to_mul(rain_gain_db)
    mixed = v + r
    # avoid clipping
    maxv = np.max(np.abs(mixed)) if mixed.size else 1.0
    if maxv > 1.0:
        mixed = mixed / maxv
    sf.write(str(out_path), mixed, sr)
    logging.info(f"Mixed voice+rain => {out_path} ({duration_s}s)")


def render_video_from_audio(audio_path: Path, out_mp4: Path, duration_s: int, bg_image: Optional[Path] = None):
    """Render MP4 with static image + audio using moviepy"""
    if bg_image and bg_image.exists():
        img = str(bg_image)
    else:
        # create a simple black image if not exists
        tmp_img = OUTPUT / "bg_placeholder.jpg"
        if not tmp_img.exists():
            from PIL import Image, ImageDraw, ImageFont
            im = Image.new("RGB", (1280, 720), (10, 10, 20))
            draw = ImageDraw.Draw(im)
            draw.text((50, 300), "Relaxing Rainy Story", fill=(200, 200, 220))
            im.save(tmp_img)
        img = str(tmp_img)
    # build clip
    clip = ImageClip(img).set_duration(duration_s)
    audio = AudioFileClip(str(audio_path))
    clip = clip.set_audio(audio)
    # write file (fast; codec libx264, audio aac)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(str(out_mp4), fps=1, codec="libx264", audio_codec="aac", threads=2, verbose=False, logger=None)
    logging.info(f"Wrote video {out_mp4}")


def make_title_and_description(seed: Optional[str]) -> tuple[str, str, list]:
    # Simple title/desc generator — user can replace with Gemini-based generator separately
    title = f"Relaxing Rainy Story — {seed or 'Cozy Night'} • Sleep & Focus"
    description = (
        "A long relaxing rainy story with ambient rain sounds to help you sleep, relax, or focus. "
        "Generated automatically. Enjoy the calm atmosphere."
    )
    tags = ["rain", "sleep", "relax", "asmr", "storytelling", "chill", "ambient"]
    return title, description, tags


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=str, default=None)
    p.add_argument("--duration", type=int, default=7200, help="target duration in seconds (default 7200 = 2h)")
    p.add_argument("--no-tts", action="store_true", help="skip TTS generation and use silent placeholder voice")
    p.add_argument("--keep-intermediate", action="store_true", help="do not delete generated intermediates")
    args = p.parse_args()

    seed = args.seed
    duration = args.duration
    start = datetime.datetime.utcnow().isoformat()

    logging.info("Starting allinone pipeline")
    # 1) story generation
    story_text = None
    try:
        # try external generator
        story_text = generate_story_via_gemini(seed)
        logging.info("Generated story via external generator")
    except Exception as ex:
        logging.info("Generator unavailable, using local story/fallback")
        story_text = pick_local_story(seed)

    # Save story text
    story_file = OUTPUT / "story.txt"
    story_file.write_text(story_text, encoding="utf-8")
    logging.info(f"Saved story to {story_file}")

    # 2) prepare SEO
    title, description, tags = make_title_and_description(seed or "Cozy Night")
    save_seo(title, description, tags)

    # 3) TTS generation
    voice_path = OUTPUT / "voice.wav"
    try:
        if args.no_tts:
            raise RuntimeError("no-tts requested")
        if COQUI_AVAILABLE:
            tts_coqui_generate(story_text, voice_path)
        else:
            raise RuntimeError("Coqui not available")
    except Exception as ex:
        logging.warning(f"TTS generation failed ({ex}), generating silent placeholder voice")
        # create silent voice of length 1/3 duration (reasonable speaking density) or shorter
        # We'll create ~duration/3 seconds of voice and let mixing/muxing repeat as needed
        approx_voice_secs = max(300, min(1800, duration // 3))
        make_silent_voice(approx_voice_secs, voice_path)

    # 4) Prepare rain loop (if not present, create a silent rain placeholder)
    rain_loop_path = OUTPUT / "rain_loop.wav"
    if RAIN_PATH.exists():
        try:
            loop_audio_to_duration(RAIN_PATH, duration, rain_loop_path)
        except Exception as ex:
            logging.warning("Failed to loop provided rain file, creating silent rain")
            make_silent_voice(duration, rain_loop_path)
    else:
        logging.warning("No assets/rain.mp3 found, creating silent rain")
        make_silent_voice(duration, rain_loop_path)

    # 5) Mix voice and rain to final audio (voice may be shorter; mix function pads/loops)
    final_audio = OUTPUT / "final_audio.wav"
    mix_voice_and_rain(voice_path, rain_loop_path, duration, final_audio, voice_gain_db=0.0, rain_gain_db=-12.0)

    # 6) Render video (static image + final_audio)
    out_video = OUTPUT / f"final_video_{uuid.uuid4().hex[:8]}.mp4"
    render_video_from_audio(final_audio, out_video, duration, bg_image=BG_IMAGE if BG_IMAGE.exists() else None)

    # write also a fixed path for schedule_upload script to find
    final_link = OUTPUT / "final_video.mp4"
    if final_link.exists():
        final_link.unlink()
    out_video.rename(final_link)
    logging.info(f"Final video ready at: {final_link}")

    # done
    logging.info("Pipeline complete.")
    logging.info(f"Started at {start}; finished at {datetime.datetime.utcnow().isoformat()}")

    # cleanup if requested
    if not args.keep_intermediate:
        # keep final files only
        for f in OUTPUT.glob("voice_part_*.wav"):
            try:
                f.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()
