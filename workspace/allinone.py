# workspace/allinone.py
"""
All-in-one tool:
1) generate story text using GPT-4o
2) create SEO metadata (title, description, tags) in English using GPT-4o
3) synthesize voice via Coqui TTS
4) mix ambient rain track
5) render video (moviepy)
6) create thumbnail (calls workspace/make_thumbnail.py)
7) optionally upload to YouTube if YT_UPLOAD=true and secrets provided
"""
import os
import json
import time
import uuid
import subprocess
from pathlib import Path
from datetime import datetime
import openai
from TTS.api import TTS
from pydub import AudioSegment
from moviepy.editor import (
    ImageClip,
    AudioFileClip,
    concatenate_audioclips,
    CompositeVideoClip,
)
from make_thumbnail import create_thumbnail
from PIL import Image
import requests

# --- Configuration / env ---
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    print("Warning: OPENAI_API_KEY not set. GPT features will fail.")
openai.api_key = OPENAI_KEY

# Coqui TTS model name (english natural)
COQUI_MODEL = os.environ.get("COQUI_MODEL", "tts_models/en/vctk/vits")

# Path structure
ROOT = Path.cwd()
WORKSPACE = ROOT / "workspace"
OUTPUT_DIR = WORKSPACE / "output"
LOGS_DIR = WORKSPACE / "logs"
ASSETS_DIR = WORKSPACE / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# rain audio sample path (you should include a small rain loop in workspace/assets/rain_loop.wav)
RAIN_PATH = os.environ.get("RAIN_PATH", str(ASSETS_DIR / "rain_loop.wav"))

# YouTube upload control
YT_UPLOAD = os.environ.get("YT_UPLOAD", "false").lower() in ("1", "true", "yes")
YT_CLIENT_ID = os.environ.get("YT_CLIENT_ID")
YT_CLIENT_SECRET = os.environ.get("YT_CLIENT_SECRET")
YT_REFRESH_TOKEN = os.environ.get("YT_REFRESH_TOKEN")
YT_PRIVACY = os.environ.get("YT_PRIVACY", "public")

# Video params
VIDEO_RES = (1280, 720)  # 720p
FPS = 24

# duration target in seconds (e.g., 3600s = 1 hour). For faster tests, use lower.
TARGET_DURATION = int(os.environ.get("TARGET_DURATION", 600))  # default 10 minutes for CI; set 3600 locally if you want 1 hour

# voice style
VOICE_STYLE = os.environ.get("VOICE_STYLE", "male_warm")  # for prompting GPT or selecting model

# Helper logging
def log(msg):
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{ts}] {msg}")

# --- GPT helper: generate story and SEO ---
def gpt_generate_story_and_metadata(prompt_seed: str = None, language="English"):
    """
    Use GPT-4o to generate:
    - story text (target duration estimate)
    - SEO: title, short description (first 2 lines), long description, tags (10)
    We return dict with story_text and metadata
    """
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    system = {
        "role": "system",
        "content": (
            "You are a professional storyteller and YouTube SEO copywriter. "
            "Produce a calm, chill bedtime story suitable for a relaxing 'rain ambience' video. "
            "Output JSON only with fields: story, estimated_minutes, title, short_description, long_description, tags[]."
        ),
    }
    user_template = (
        "Create a relaxing English short story suitable for audio-only listening while rain plays in the background.\n"
        "Target total audio duration: approximately {minutes} minutes.\n"
        "Keep language simple, soothing, present tense. No explicit gore or graphic content.\n"
        "Also create a SEO-optimized English title (<=65 chars), a short 1-2 sentence description, "
        "a longer description (3-5 paragraphs) including keywords 'rain', 'relax', 'sleep', 'calm', and 10 short tags.\n"
        "Return JSON only.\n\nSeed: {seed}\n"
    )
    target_min = max(5, TARGET_DURATION // 60)
    if not prompt_seed:
        prompt_seed = "A small lakeside town where the main character remembers childhood summer rains."
    user = {"role": "user", "content": user_template.format(minutes=target_min, seed=prompt_seed)}
    # call chat completion
    log("Calling GPT-4o to generate story and metadata...")
    resp = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[system, user],
        temperature=0.8,
        max_tokens=1400,
    )
    text = resp["choices"][0]["message"]["content"].strip()
    # Try to parse JSON from response; GPT instructed to produce JSON only
    try:
        data = json.loads(text)
    except Exception:
        # fallback: attempt to extract JSON block
        import re
        m = re.search(r"(\{[\s\S]*\})", text)
        if m:
            data = json.loads(m.group(1))
        else:
            raise RuntimeError("GPT response could not be parsed as JSON:\n" + text)
    # Basic validation
    required = ["story", "title", "short_description", "long_description", "tags"]
    for k in required:
        if k not in data:
            raise RuntimeError(f"GPT output missing required key: {k}. Full:\n{data}")
    return data

# --- TTS: Coqui ---
def synthesize_tts_coqui(text: str, out_wav: str, model_name=COQUI_MODEL):
    """
    Use TTS library to synthesize text to wav file.
    """
    log("Loading TTS model: " + model_name)
    tts = TTS(model_name)  # uses local model cache; ensure model is available/installed
    log("Synthesizing TTS to " + out_wav)
    # TTS.tts_to_file(text=..., file_path=...)
    tts.tts_to_file(text=text, file_path=out_wav)
    return out_wav

# --- Audio mixing (rain + speech) ---
def mix_rain_and_speech(speech_wav: str, rain_wav: str, out_path: str, target_duration=TARGET_DURATION):
    """
    Create final mixed audio with speech centered and rain looped to target_duration.
    """
    log("Loading speech audio and rain track.")
    speech = AudioSegment.from_file(speech_wav)
    # ensure speech is mono/16kHz? pydub handles formats but ensure consistent.
    # Load or create rain loop
    if not os.path.exists(rain_wav):
        log("No rain track found; creating silence fill.")
        rain = AudioSegment.silent(duration=target_duration*1000)
    else:
        rain = AudioSegment.from_file(rain_wav)
    # Loop rain until target duration
    rain_loop = AudioSegment.empty()
    while len(rain_loop) < target_duration * 1000:
        rain_loop += rain
    rain_loop = rain_loop[:target_duration * 1000]
    # Mix: lower rain volume so speech dominates
    rain_loop = rain_loop - 12  # reduce by 12 dB
    # Place speech at start
    final = rain_loop.overlay(speech, position=0)
    final.export(out_path, format="wav")
    log(f"Exported mixed audio to {out_path}")
    return out_path

# --- Video rendering (image + audio) ---
def render_video_from_audio(audio_path: str, image_path: str, out_video: str, duration=TARGET_DURATION, fps=FPS, resolution=VIDEO_RES):
    """
    Render a simple video: static image stretched to duration + audio.
    For long videos, better to loop small subtle motion (we implement gentle zoom).
    """
    log("Rendering video...")
    # Use ImageClip with duration and slight zoom via lambda (moviepy)
    clip = ImageClip(image_path).set_duration(duration)
    # apply small zoom effect
    def zoom(t):
        # zoom from 1.0 to 1.03 over duration
        return 1 + 0.03 * (t / max(1, duration))
    clip = clip.resize(lambda t: (resolution[0] * zoom(t) / clip.w, resolution[1] * zoom(t) / clip.h))
    # ensure final size
    clip = clip.set_fps(fps).resize(newsize=resolution)
    audio = AudioFileClip(audio_path)
    clip = clip.set_audio(audio).set_duration(audio.duration)
    clip.write_videofile(out_video, codec="libx264", audio_codec="aac", fps=fps, threads=2, preset="medium", verbose=False, logger=None)
    log(f"Saved video to {out_video}")
    return out_video

# --- YouTube upload (optional) ---
def upload_to_youtube(video_file, title, description, tags, thumbnail_path, privacy="public"):
    """
    Upload video via YouTube Data API v3 (requires OAuth2 refresh token).
    Expects env vars: YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN
    """
    if not (YT_CLIENT_ID and YT_CLIENT_SECRET and YT_REFRESH_TOKEN):
        raise RuntimeError("YouTube credentials not provided in env (YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN).")
    log("Preparing YouTube upload...")
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    creds = Credentials(
        token=None,
        refresh_token=YT_REFRESH_TOKEN,
        client_id=YT_CLIENT_ID,
        client_secret=YT_CLIENT_SECRET,
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube.force-ssl"]
    )
    # refresh to get access token
    creds.refresh(requests.Request())
    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags[:50],
            "categoryId": "22"  # People & Blogs / maybe change
        },
        "status": {
            "privacyStatus": privacy
        },
    }
    media = MediaFileUpload(video_file, chunksize=-1, resumable=True, mimetype="video/*")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    log("Uploading video (this may take time)...")
    while response is None:
        status, response = request.next_chunk()
        if status:
            log(f"Upload progress: {int(status.progress() * 100)}%")
    video_id = response.get("id")
    log(f"Video uploaded with id: {video_id}")
    # set thumbnail
    if thumbnail_path and os.path.exists(thumbnail_path):
        t_req = youtube.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(thumbnail_path))
        t_resp = t_req.execute()
        log("Thumbnail set.")
    return video_id

# --- Main orchestration ---
def run_pipeline(seed_prompt=None, voice_model=COQUI_MODEL):
    run_id = uuid.uuid4().hex[:8]
    log(f"Run id: {run_id}")
    # 1. Generate story + SEO via GPT
    meta = gpt_generate_story_and_metadata(prompt_seed=seed_prompt)
    story_text = meta["story"]
    title = meta["title"][:65]
    short_desc = meta["short_description"]
    long_desc = meta["long_description"]
    tags = meta.get("tags", [])[:50]
    # Save metadata
    meta_path = OUTPUT_DIR / f"meta_{run_id}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log("Saved metadata to " + str(meta_path))

    # 2. Synthesize speech via Coqui
    speech_wav = OUTPUT_DIR / f"speech_{run_id}.wav"
    synthesize_tts_coqui(story_text, str(speech_wav), model_name=voice_model)

    # 3. Mix rain + speech
    mixed_wav = OUTPUT_DIR / f"final_audio_{run_id}.wav"
    mix_rain_and_speech(str(speech_wav), RAIN_PATH, str(mixed_wav), target_duration=TARGET_DURATION)

    # 4. Render video - we need a base image: either a generated one, or fallback
    # Option: user can place base image at workspace/assets/background.jpg
    bg = ASSETS_DIR / "background.jpg"
    if not bg.exists():
        # create a simple gradient/warm background as fallback
        log("No background.jpg found in assets. Creating simple fallback image.")
        im = Image.new("RGB", VIDEO_RES, color=(25, 20, 30))
        im.save(str(bg))
    # Compose final image (we can also overlay subtle noise)
    out_video = OUTPUT_DIR / f"video_{run_id}.mp4"
    render_video_from_audio(str(mixed_wav), str(bg), str(out_video), duration=TARGET_DURATION)

    # 5. Thumbnail: generate from frame -> style -> save
    thumb_path = OUTPUT_DIR / f"thumbnail_{run_id}.jpg"
    tpath = create_thumbnail(str(out_video), title, thumb_path)

    # 6. Save final metadata + log
    result = {
        "run_id": run_id,
        "title": title,
        "short_description": short_desc,
        "long_description": long_desc,
        "tags": tags,
        "video": str(out_video),
        "thumbnail": str(tpath),
        "audio": str(mixed_wav),
    }
    result_path = OUTPUT_DIR / f"result_{run_id}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log("Pipeline finished. Outputs saved to " + str(OUTPUT_DIR))

    # 7. Optionally upload to YouTube
    if YT_UPLOAD:
        try:
            vid = upload_to_youtube(str(out_video), title, long_desc, tags, str(tpath), privacy=YT_PRIVACY)
            log("Uploaded video id: " + vid)
        except Exception as e:
            log("YouTube upload failed: " + str(e))

    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, default=None, help="Seed prompt for story generation")
    parser.add_argument("--duration", type=int, default=None, help="target duration seconds")
    parser.add_argument("--voice", type=str, default=None, help="Coqui model name to use")
    args = parser.parse_args()
    if args.duration:
        TARGET_DURATION = args.duration
    if args.voice:
        COQUI_MODEL = args.voice
    run_pipeline(seed_prompt=args.seed, voice_model=COQUI_MODEL)
