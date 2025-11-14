#!/usr/bin/env python3
# allinone.py — FULL pipeline (FFmpeg-based render) with short intro (5-7s)
import os
import sys
import json
import time
import base64
import logging
import datetime
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

# Optional Coqui TTS
try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except Exception:
    COQUI_AVAILABLE = False

# Paths
ROOT = Path.cwd()
WORKSPACE = ROOT / "workspace"
OUTPUT = WORKSPACE / "output"
ASSETS = WORKSPACE / "assets"

OUTPUT.mkdir(parents=True, exist_ok=True)
WORKSPACE.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

# Config / env
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.0-flash")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.0-flash")
COQUI_MODEL = os.getenv("COQUI_MODEL", "tts_models/en/vctk/vits")

RAIN_MP3 = Path(os.getenv("RAIN_FILE", ASSETS / "rain.mp3"))
RAIN_BG = Path(os.getenv("RAIN_BG", ASSETS / "rain_bg.mp4"))
MIN_DURATION = int(os.getenv("MIN_DURATION", "7200"))  # seconds (default 2 hours)
THUMB_AT = float(os.getenv("THUMB_FRAME_AT", "10.0"))
AUDIO_SR = int(os.getenv("AUDIO_SR", "22050"))
TTS_WPM = int(os.getenv("TTS_WPM", "150"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ----------------- helpers -----------------
def run_cmd(cmd: List[str], check=True):
    logging.debug("RUN: " + " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and proc.returncode != 0:
        logging.error("Command failed: " + " ".join(cmd))
        logging.error("stdout: " + proc.stdout)
        logging.error("stderr: " + proc.stderr)
        raise RuntimeError(f"Command failed: {cmd}\n{proc.stderr}")
    return proc.stdout, proc.stderr, proc.returncode

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

# ----------------- Gemini text/image -----------------
def gemini_text(prompt: str, max_tokens: int = 2048) -> str:
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in env")
    # Use generateContent endpoint (v1)
    url = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_TEXT_MODEL}:generateContent"
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "maxOutputTokens": max_tokens
    }
    r = requests.post(url, params={"key": GEMINI_KEY}, json=payload, timeout=60)
    r.raise_for_status()
    js = r.json()
    # Best-effort extraction
    try:
        # Newer responses: candidates -> content -> parts
        if "candidates" in js and js["candidates"]:
            cand = js["candidates"][0]
            if "content" in cand:
                content = cand["content"]
                if isinstance(content, dict) and "parts" in content:
                    parts = content["parts"]
                    if isinstance(parts, list) and parts:
                        return "".join(p.get("text","") for p in parts)
        # Fallback older shape
        if "candidates" in js and js["candidates"]:
            c = js["candidates"][0]
            if isinstance(c.get("content"), list) and c["content"]:
                p = c["content"][0]
                if isinstance(p, dict) and "text" in p:
                    return p["text"]
    except Exception:
        pass
    return json.dumps(js)

def gemini_image_from_reference(img_b64: str, prompt_extra: str = "") -> bytes:
    """Call Gemini image endpoint with reference image base64. Return raw image bytes."""
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    # Try v1 generateImage
    url = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_IMAGE_MODEL}:generateImage"
    payload = {
        "prompt": {
            "text": "Create a highly-clickable YouTube thumbnail for a chill/study/sleep rain ambience video. " + prompt_extra
        },
        "image_context": {
            "reference_images": [{"image_bytes": img_b64}]
        },
        "format": "PNG"
    }
    r = requests.post(url, params={"key": GEMINI_KEY}, json=payload, timeout=90)
    r.raise_for_status()
    js = r.json()
    # try multiple possible shapes
    if "images" in js and isinstance(js["images"], list) and js["images"]:
        b64 = js["images"][0].get("imageBytes") or js["images"][0].get("imageBytes")
        if b64:
            return base64.b64decode(b64)
    if "generatedImages" in js and isinstance(js["generatedImages"], list) and js["generatedImages"]:
        data = js["generatedImages"][0].get("data")
        if data:
            return base64.b64decode(data)
    # fallback: stringify
    raise RuntimeError(f"Unexpected Gemini image response: {js}")

# ----------------- SEO files -----------------
def save_seo_files(title: str, description: str, tags: List[str]):
    (OUTPUT / "title.txt").write_text(title or "", encoding="utf-8")
    (OUTPUT / "description.txt").write_text(description or "", encoding="utf-8")
    (OUTPUT / "tags.txt").write_text(",".join(tags or []), encoding="utf-8")

# ----------------- story generation -----------------
def estimate_tts_seconds(text: str, wpm: int = TTS_WPM) -> int:
    words = len(text.split())
    minutes = words / float(wpm) if wpm > 0 else words / 150.0
    return int(minutes * 60)

def generate_full_story(seed: str, min_seconds: int, per_batch_words: int = 1400, max_batches: int = 12) -> Tuple[str, int]:
    logging.info(f"Generating story for seed='{seed}', target {min_seconds}s")
    full = ""
    attempts = 0
    base_prompt = (
        f"Write a long slow atmospheric English story for sleep/study. Topic: {seed}. "
        f"Calm, descriptive language. Aim for roughly {per_batch_words} words per batch. "
        "Do not include titles or chapter numbers; output plain paragraphs suitable for narration."
    )
    while True:
        attempts += 1
        try:
            part = gemini_text(base_prompt)
        except Exception as e:
            logging.warning(f"Gemini text failed attempt {attempts}: {e}")
            part = ""
        if not part.strip():
            part = "The rain whispered softly as the night held the city in its gentle hush."
        full += ("\n\n" + part).strip()
        secs = estimate_tts_seconds(full)
        logging.info(f"Batch {attempts}: estimated {secs}s")
        if secs >= min_seconds:
            break
        if attempts >= max_batches:
            logging.info("Reached max_batches; stopping generation")
            break
        time.sleep(0.5)
    return full.strip(), estimate_tts_seconds(full)

# ----------------- intro (5-7s) -----------------
def generate_intro(title: str, seed: str) -> str:
    prompt = (
        "Write a VERY SHORT spoken intro for a relaxing rain storytelling video. "
        "Length: 1–2 short sentences, total ~5–7 seconds when spoken. "
        "Tone: calm, warm, soft. Do NOT summarize or reveal plot. "
        f"Title: {title}\nTopic: {seed}"
    )
    try:
        out = gemini_text(prompt, max_tokens=128)
        return out.strip().replace("\n", " ")
    except Exception:
        return "Welcome. Tonight's story begins now."

# ----------------- TTS (Coqui) -----------------
def generate_tts_coqui(text: str, out_wav: Path, model: str = COQUI_MODEL):
    if not COQUI_AVAILABLE:
        raise RuntimeError("Coqui TTS not available (install TTS)")
    logging.info(f"Generating TTS with Coqui model {model}")
    tts = TTS(model)
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
        tts.tts_to_file(text=p, file_path=str(tmp), speaker="p225")
        tmp_files.append(tmp)
    # concat
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
    for f in tmp_files:
        try:
            f.unlink()
        except Exception:
            pass
    logging.info(f"TTS saved to {out_wav}")

# ----------------- audio/video helpers using ffmpeg -----------------
def ffmpeg_loop_audio_to_wav(src_mp3: Path, dur: int, out_wav: Path):
    """Create a WAV by looping src_mp3 via ffmpeg to exact dur seconds."""
    if not src_mp3.exists():
        raise RuntimeError(f"Source audio not found: {src_mp3}")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-stream_loop", "-1", "-i", str(src_mp3),
        "-t", str(dur),
        "-ar", str(AUDIO_SR),
        "-ac", "1",
        "-vn",
        str(out_wav)
    ]
    run_cmd(cmd)

def ffmpeg_mix_voice_and_rain(voice_wav: Path, rain_wav: Path, out_wav: Path, voice_gain: float = 1.0, rain_gain: float = 0.25):
    """Mix voice and rain using ffmpeg filter_complex; outputs WAV (float32 pcm)."""
    if not voice_wav.exists():
        raise RuntimeError("voice wav missing")
    if not rain_wav.exists():
        raise RuntimeError("rain wav missing")
    # Use amix with volumes: apply volume filters then amix
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(voice_wav),
        "-i", str(rain_wav),
        "-filter_complex",
        f"[0:a]volume={voice_gain}[v];[1:a]volume={rain_gain}[r];[v][r]amix=inputs=2:duration=longest:dropout_transition=0,volume=1",
        "-ar", str(AUDIO_SR),
        "-ac", "1",
        str(out_wav)
    ]
    run_cmd(cmd)

def ffmpeg_loop_video_and_mux_audio(video_in: Path, audio_in: Path, out_mp4: Path, dur: int, crf: int = 20):
    """Loop video (stream_loop) and mux in provided audio, re-encoding to h264/aac for compatibility."""
    if not video_in.exists():
        raise RuntimeError(f"Background video missing: {video_in}")
    if not audio_in.exists():
        raise RuntimeError(f"Audio missing: {audio_in}")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-stream_loop", "-1", "-i", str(video_in),
        "-i", str(audio_in),
        "-t", str(dur),
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        str(out_mp4)
    ]
    run_cmd(cmd)

def ffmpeg_extract_frame(video_in: Path, at_sec: float, out_png: Path):
    if not video_in.exists():
        raise RuntimeError(f"Video missing: {video_in}")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(at_sec),
        "-i", str(video_in),
        "-frames:v", "1",
        "-q:v", "2",
        str(out_png)
    ]
    run_cmd(cmd)

# ----------------- fallback thumbnail -----------------
def create_fallback_thumbnail(text: str, out: Path, size=(1280, 720)):
    im = Image.new("RGB", size, (20, 20, 30))
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 48)
    except Exception:
        font = ImageFont.load_default()
    x, y = 60, 260
    # outlined text for readability
    for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
        d.text((x+dx, y+dy), text[:80], font=font, fill=(0,0,0))
    d.text((x, y), text[:80], font=font, fill=(230,230,230))
    im.save(out)
    logging.info(f"Created fallback thumbnail {out}")

# ----------------- main pipeline -----------------
def main():
    logging.info("=== allinone FFmpeg pipeline start ===")
    start_time = datetime.datetime.utcnow().isoformat()

    seed = os.getenv("VIDEO_SEED", "Cozy rainy focus")
    logging.info(f"Seed: {seed}")

    # 1) Generate story until estimated >= MIN_DURATION
    try:
        story_text, est_secs = generate_full_story(seed, MIN_DURATION)
        logging.info("Story generated")
    except Exception as e:
        logging.warning(f"Story generation failed: {e}")
        base = ("The rain whispered against the window as the town settled into a slow, measured breath. "
                "Tonight, the lamps glowed softly and the streets remembered the footsteps of those who passed.")
        story_text = "\n\n".join([base for _ in range(60)])
        est_secs = estimate_tts_seconds(story_text)

    (OUTPUT / "story_raw.txt").write_text(story_text, encoding="utf-8")

    # 2) Generate SEO (title/description/tags)
    title = ""
    description = ""
    tags: List[str] = []
    try:
        seo_prompt = (
            "Create a YouTube SEO-optimized title (short & compelling), a long description (~300 words), "
            "and an array of 15-25 concise tags relevant to a chill/study/sleep rain ambience storytelling video. "
            "Return strictly JSON: {\"title\":\"...\",\"description\":\"...\",\"tags\":[...]}.\n\n"
            f"Story excerpt:\n{story_text[:1600]}"
        )
        seo_raw = gemini_text(seo_prompt)
        parsed = try_parse_json(seo_raw)
        if parsed:
            title = parsed.get("title", "") or ""
            description = parsed.get("description", "") or ""
            tags = parsed.get("tags", []) or []
            logging.info("SEO generated")
    except Exception as e:
        logging.warning(f"SEO generation failed: {e}")

    if not title:
        title = f"Relaxing Rain Story — {seed}"
    if not description:
        description = "Relaxing rainy ambience with a soothing narrated story to help you focus, relax, or sleep."
    if not tags:
        tags = ["rain", "relax", "sleep", "study", "ambient", "story"]

    save_seo_files(title, description, tags)

    # 3) Create 5-7s intro
    intro_text = generate_intro(title, seed)
    logging.info(f"Intro: {intro_text}")

    # 4) Combine intro + story
    full_text = intro_text + "\n\n" + story_text
    (OUTPUT / "story_full.txt").write_text(full_text, encoding="utf-8")

    # 5) TTS -> voice.wav (Coqui) or fallback silent
    voice_wav = OUTPUT / "voice.wav"
    try:
        if COQUI_AVAILABLE:
            generate_tts_coqui(full_text, voice_wav, model=COQUI_MODEL)
        else:
            raise RuntimeError("Coqui not available")
    except Exception as e:
        logging.warning(f"TTS failed: {e}; creating silent voice placeholder (30s)")
        # create reasonable silent placeholder (30s) to avoid subsequent failures
        dur = min(60, max(10, int(est_secs)))  # make some silent if needed
        make_silent_wave(dur, voice_wav)

    # 6) compute durations
    try:
        voice_dur = int(wav_len(voice_wav))
    except Exception:
        voice_dur = 0
    final_duration = max(MIN_DURATION, voice_dur)
    logging.info(f"Voice duration: {voice_dur}s, final video duration: {final_duration}s")

    # 7) Prepare rain loop WAV via ffmpeg
    rain_loop_wav = OUTPUT / "rain_loop.wav"
    try:
        if RAIN_MP3.exists():
            ffmpeg_loop_audio_to_wav(RAIN_MP3, final_duration, rain_loop_wav)
        else:
            logging.warning("rain mp3 missing, creating silent rain")
            silent(final_duration, rain_loop_wav)
    except Exception as e:
        logging.warning(f"ffmpeg loop audio failed: {e}; falling back to numpy loop")
        # fallback numpy loop
        try:
            data, sr = sf.read(str(RAIN_MP3), dtype='float32')
            if data.ndim > 1:
                data = data.mean(axis=1)
            need = int(final_duration * sr)
            rep = (need // len(data)) + 1
            full = np.tile(data, rep)[:need]
            sf.write(str(rain_loop_wav), full, sr)
        except Exception:
            silent(final_duration, rain_loop_wav)

    # 8) Mix voice + rain -> final_audio.wav (using ffmpeg)
    final_audio = OUTPUT / "final_audio.wav"
    try:
        ffmpeg_mix_voice_and_rain(voice_wav, rain_loop_wav, final_audio, voice_gain=1.0, rain_gain=0.25)
    except Exception as e:
        logging.warning(f"ffmpeg mix failed: {e}; fallback to numpy mix")
        # fallback naive numpy mix
        try:
            v, sv = sf.read(str(voice_wav), dtype='float32')
            r, sr = sf.read(str(rain_loop_wav), dtype='float32')
            if v.ndim > 1: v = v.mean(axis=1)
            if r.ndim > 1: r = r.mean(axis=1)
            if sv != sr:
                v = np.interp(np.linspace(0, len(v), int(len(v) * sr / sv), endpoint=False), np.arange(len(v)), v)
            need = int(final_duration * sr)
            if len(r) < need:
                r = np.tile(r, (need // len(r)) + 1)[:need]
            else:
                r = r[:need]
            if len(v) < need:
                v = np.concatenate([v, np.zeros(need - len(v))])
            else:
                v = v[:need]
            mix = v + r * 0.25
            m = np.max(np.abs(mix)) if mix.size else 1.0
            if m > 1.0:
                mix = mix / m
            sf.write(str(final_audio), mix, sr)
        except Exception:
            silent(final_duration, final_audio)

    # 9) Thumbnail: extract frame via ffmpeg and ask Gemini to enhance
    frame_png = OUTPUT / "thumb_frame.png"
    thumb_out = OUTPUT / "thumbnail.jpg"
    try:
        ffmpeg_extract_frame(RAIN_BG, THUMB_AT, frame_png)
        with open(frame_png, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        try:
            thumb_bytes = gemini_image_from_reference(b64, prompt_extra="Make it high-CTR and cozy; keep composition.")
            thumb_out.write_bytes(thumb_bytes)
            logging.info("Thumbnail created by Gemini")
        except Exception as e:
            logging.warning(f"Gemini thumbnail failed: {e}; using extracted frame")
            thumb_out.write_bytes(frame_png.read_bytes())
    except Exception as e:
        logging.warning(f"Frame extraction / thumbnail generation failed: {e}; creating fallback thumbnail")
        create_fallback_thumbnail(title, thumb_out)

    # 10) Render final video via FFmpeg: loop video + mux final audio
    final_video = OUTPUT / "final_video.mp4"
    try:
        ffmpeg_loop_video_and_mux_audio(RAIN_BG, final_audio, final_video, final_duration)
        logging.info(f"Final video rendered: {final_video}")
    except Exception as e:
        logging.error(f"FFmpeg render failed: {e}")
        # fallback: create static image video via ffmpeg
        try:
            # create a jpg fallback if necessary
            if not thumb_out.exists():
                create_fallback_thumbnail(title, thumb_out)
            # use ffmpeg to make static video from image + audio
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-loop", "1", "-i", str(thumb_out),
                "-i", str(final_audio),
                "-t", str(final_duration),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                str(final_video)
            ]
            run_cmd(cmd)
            logging.info("Fallback static video created")
        except Exception as e2:
            logging.error(f"Fallback static render also failed: {e2}")
            raise

    # 11) Done — list outputs
    end_time = datetime.datetime.utcnow().isoformat()
    logging.info("=== Pipeline complete ===")
    logging.info(f"Start: {start_time}  End: {end_time}")
    outputs = {
        "video": str(final_video),
        "thumbnail": str(thumb_out),
        "title": (OUTPUT / "title.txt").read_text(encoding="utf-8") if (OUTPUT / "title.txt").exists() else title,
        "description": (OUTPUT / "description.txt").read_text(encoding="utf-8") if (OUTPUT / "description.txt").exists() else description,
        "tags": (OUTPUT / "tags.txt").read_text(encoding="utf-8") if (OUTPUT / "tags.txt").exists() else ",".join(tags)
    }
    logging.info(json.dumps(outputs, indent=2))

if __name__ == "__main__":
    main()

