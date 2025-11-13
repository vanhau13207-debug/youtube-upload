#!/usr/bin/env python3
# allinone.py — Gemini Flash full pipeline (video background from rain_bg.mp4, thumbnail from frame, rain.mp3 BGM)
import os, sys, json, base64, uuid, logging, datetime
from pathlib import Path
import requests
import numpy as np
import soundfile as sf
from moviepy.editor import VideoFileClip, ImageClip, AudioFileClip, concatenate_audioclips

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
ASSETS.mkdir(parents=True, exist_ok=True)

# Config
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.0-flash")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.0-flash")
COQUI_MODEL = os.getenv("COQUI_MODEL", "tts_models/en/vctk/vits")
RAIN_MP3 = Path(os.getenv("RAIN_FILE", str(ASSETS / "rain.mp3")))
RAIN_BG_MP4 = Path(os.getenv("RAIN_BG", str(ASSETS / "rain_bg.mp4")))
DURATION = int(os.getenv("TARGET_DURATION", "7200"))  # seconds (2h)
THUMB_FRAME_AT = float(os.getenv("THUMB_FRAME_AT", "10.0"))  # seconds into video to grab frame
AUDIO_SR = int(os.getenv("AUDIO_SR", "22050"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------- Gemini text (generate story / seo) ----------
def gemini_text(prompt: str, model=GEMINI_TEXT_MODEL, key=GEMINI_KEY, max_output_chars=30000) -> str:
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateText"
    payload = {"prompt": {"text": prompt}, "maxOutputTokens": 2048}
    r = requests.post(url, params={"key": key}, json=payload, timeout=60)
    r.raise_for_status()
    js = r.json()
    # try to read multiple possible shapes
    try:
        return js["candidates"][0]["content"][0]["text"]
    except:
        # fallback to stringified
        return json.dumps(js)

# ---------- Gemini image (generate thumbnail from frame) ----------
def gemini_image_from_base64_input(img_b64: str, prompt_extra: str = "", model=GEMINI_IMAGE_MODEL, key=GEMINI_KEY) -> bytes:
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateImage"
    # we pass prompt describing we want an SEO thumbnail based on provided image
    payload = {
        "prompt": {
            "text": f"Create a highly-clickable YouTube thumbnail for a chill/study/sleep rain ambience video. "
                    f"Use the provided reference image and enhance color, contrast, composition, and add subtle cinematic vibes. {prompt_extra}"
        },
        "image_context": {"reference_images": [{"image_bytes": img_b64}]},
        "format": "PNG"
    }
    r = requests.post(url, params={"key": key}, json=payload, timeout=60)
    r.raise_for_status()
    js = r.json()
    # Expect base64 in images[0].imageBytes
    b64 = js["images"][0]["imageBytes"]
    return base64.b64decode(b64)

# ---------- UTIL: save seo files ----------
def save_seo(title, desc, tags):
    (OUTPUT / "title.txt").write_text(title, encoding="utf-8")
    (OUTPUT / "description.txt").write_text(desc, encoding="utf-8")
    (OUTPUT / "tags.txt").write_text(",".join(tags), encoding="utf-8")

# ---------- TTS via Coqui ----------
def generate_tts_coqui(text: str, out_wav: Path):
    if not COQUI_AVAILABLE:
        raise RuntimeError("Coqui TTS not installed")
    tts = TTS(COQUI_MODEL)
    # chunk into safe sizes
    MAX = 2500
    parts = []
    cur = ""
    for line in text.splitlines():
        if len(cur) + len(line) + 1 > MAX:
            parts.append(cur)
            cur = line
        else:
            cur = (cur + "\n" + line).strip()
    if cur:
        parts.append(cur)
    tmp_list = []
    for i, p in enumerate(parts):
        tmp = OUTPUT / f"voice_part_{i}.wav"
        tts.tts_to_file(text=p, file_path=str(tmp))
        tmp_list.append(tmp)
    # concat
    arrays = []
    sr = None
    for t in tmp_list:
        a, s = sf.read(str(t), dtype='float32')
        if sr is None: sr = s
        arrays.append(a)
    if not arrays:
        raise RuntimeError("No TTS produced")
    full = np.concatenate(arrays, axis=0)
    sf.write(str(out_wav), full, sr)
    # cleanup parts
    for t in tmp_list:
        try: t.unlink()
        except: pass

# ---------- FALLBACK silent wave ----------
def make_silent(seconds: int, out_wav: Path, sr=AUDIO_SR):
    data = np.zeros(int(seconds * sr), dtype='float32')
    sf.write(str(out_wav), data, sr)

# ---------- extract frame from mp4 ----------
def extract_frame_as_png(mp4_path: Path, at_time: float, out_png: Path):
    clip = VideoFileClip(str(mp4_path))
    frame = clip.get_frame(at_time)
    from PIL import Image
    im = Image.fromarray(frame)
    im.save(out_png)
    clip.reader.close()
    clip.audio = None

# ---------- loop rain mp3 to duration ----------
def loop_audio_to_duration(src_mp3: Path, duration_s: int, out_wav: Path):
    data, sr = sf.read(str(src_mp3), dtype='float32')
    if data.ndim > 1: data = data.mean(axis=1)
    need = int(duration_s * sr)
    reps = (need // len(data)) + 1
    full = np.tile(data, reps)[:need]
    sf.write(str(out_wav), full, sr)

# ---------- mix voice + rain ----------
def mix_voice_and_rain(voice_wav: Path, rain_wav: Path, duration_s: int, out_wav: Path, voice_db=0.0, rain_db=-12.0):
    v, sr1 = sf.read(str(voice_wav), dtype='float32')
    r, sr2 = sf.read(str(rain_wav), dtype='float32')
    if v.ndim > 1: v = v.mean(axis=1)
    if r.ndim > 1: r = r.mean(axis=1)
    sr = sr1
    if sr1 != sr2:
        # naive resample v -> sr2
        v = np.interp(np.linspace(0, len(v), int(len(v)*sr2/sr1), endpoint=False), np.arange(len(v)), v)
        sr = sr2
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
    def dbmul(db): return 10 ** (db/20.0)
    v = v * dbmul(voice_db)
    r = r * dbmul(rain_db)
    mix = v + r
    mx = np.max(np.abs(mix)) if mix.size else 1.0
    if mx > 1.0: mix = mix / mx
    sf.write(str(out_wav), mix, sr)

# ---------- render final video: use rain_bg.mp4 as visual, replace audio ----------
def render_final_video_with_background(video_bg: Path, final_audio: Path, out_mp4: Path):
    bg = VideoFileClip(str(video_bg))
    # set audio
    audio = AudioFileClip(str(final_audio))
    # if bg shorter than target, loop bg
    if bg.duration < DURATION:
        # create looped clip
        clips = []
        times = int(np.ceil(DURATION / bg.duration))
        for _ in range(times):
            clips.append(bg.copy())
        from moviepy.editor import concatenate_videoclips
        bg_final = concatenate_videoclips(clips).subclip(0, DURATION)
    else:
        bg_final = bg.subclip(0, DURATION)
    bg_final = bg_final.set_audio(audio)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    bg_final.write_videofile(str(out_mp4), codec="libx264", audio_codec="aac", fps=24, threads=2, verbose=False, logger=None)
    # cleanup
    try:
        bg.reader.close()
    except: pass

# ---------- main ----------
def main():
    logging.info("Start allinone pipeline")
    # 1) generate story & SEO via Gemini
    seed = os.getenv("VIDEO_SEED", "Cozy rainy focus")
    prompt_story = f"Write a long, slow, soothing english story in a chill tone suitable for listening while studying, focusing or sleeping. Topic: {seed}. Aim for content that reads for ~40-60 minutes; produce clean paragraphs."
    story = gemini_text(prompt_story)
    (OUTPUT / "story.txt").write_text(story, encoding="utf-8")

    # SEO
    prompt_seo = f"Create a YouTube SEO-optimized title, a long description (~400 words) and 20 concise tags for this video. Return JSON: {{\"title\":\"\",\"description\":\"\",\"tags\":[...]}}. Input story excerpt:\n\n{story[:1500]}"
    seo_json = {}
    try:
        seo_text = gemini_text(prompt_seo)
        seo_json = json.loads(seo_text)
    except Exception:
        # fallback
        seo_json = {"title": f"Relaxing Rainy Story — {seed}", "description": "Relaxing rainy ambience with story", "tags": ["rain","relax","focus","sleep"]}
    title = seo_json.get("title", f"Relaxing Rainy Story — {seed}")
    description = seo_json.get("description", "")
    tags = seo_json.get("tags", ["rain","relax","focus","sleep"])
    save_seo(title, description, tags)

    # 2) extract frame from rain_bg.mp4 → send to Gemini image generator (thumbnail)
    if not RAIN_BG_MP4.exists():
        raise RuntimeError(f"Missing video background: {RAIN_BG_MP4}")
    frame_png = OUTPUT / "thumb_frame.png"
    extract_frame_as_png(RAIN_BG_MP4, THUMB_FRAME_AT, frame_png)
    # encode frame to base64
    with open(frame_png, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    try:
        thumb_bytes = gemini_image_from_base64_input(b64, prompt_extra="Make it a YouTube thumbnail, high contrast, cinematic, keep original composition.")
        (OUTPUT / "thumbnail.jpg").write_bytes(thumb_bytes)
    except Exception as e:
        logging.warning("Gemini image generation failed: " + str(e))
        # fallback: use frame as thumbnail
        (OUTPUT / "thumbnail.jpg").write_bytes(frame_png.read_bytes())

    # 3) TTS
    voice_wav = OUTPUT / "voice.wav"
    try:
        generate_tts_coqui(story, voice_wav)
    except Exception as e:
        logging.warning("Coqui TTS failed: " + str(e))
        make_silent(min(1800, max(300, DURATION // 3)), voice_wav)

    # 4) loop rain.mp3 to duration
    rain_wav = OUTPUT / "rain_loop.wav"
    if RAIN_MP3.exists():
        loop_audio_to_duration(RAIN_MP3, DURATION, rain_wav)
    else:
        make_silent(DURATION, rain_wav)

    # 5) mix voice + rain
    final_audio = OUTPUT / "final_audio.wav"
    mix_voice_and_rain(voice_wav, rain_wav, DURATION, final_audio, voice_db=0.0, rain_db=-12.0)

    # 6) render final video using rain_bg.mp4 as visual
    final_video = OUTPUT / "final_video.mp4"
    render_final_video_with_background(RAIN_BG_MP4, final_audio, final_video)

    logging.info("Pipeline finished. Outputs in workspace/output/")
    logging.info("Files: " + ", ".join([str(p.name) for p in OUTPUT.glob("*")]))

if __name__ == "__main__":
    main()
