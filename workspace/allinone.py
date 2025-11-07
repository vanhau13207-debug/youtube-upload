#!/usr/bin/env python3
# coding: utf-8
"""
Full-auto chill story tool:
- Generate story with GPT (or Gemini fallback)
- Generate SEO (title/description/tags + thumbnail prompt)
- TTS (gTTS by default; optional Coqui if USE_COQUI=1)
- Mix rain ambience, normalize
- Render video (static/AI background + subtle Ken Burns)
- Upload to YouTube (OAuth refresh_token)
"""

import os, json, time, random, re, tempfile, subprocess, pathlib
from datetime import datetime, timezone
from typing import List, Tuple
import requests
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import AudioFileClip, ImageClip, CompositeAudioClip, afx

# ====== CONFIG ======
OUT_DIR = pathlib.Path("output")
ASSETS_DIR = pathlib.Path("workspace/assets")
RAIN_FILE = ASSETS_DIR / "rain.wav"       # đặt 1 file mưa 5–10 phút (loop)
BG_IMAGE = ASSETS_DIR / "bg.jpg"          # ảnh nền fallback nếu không dùng AI thumbnail
FONT_PATH = None                          # để None thì PIL sẽ dùng default

# video length: theo audio
VIDEOS_PER_RUN = int(os.getenv("VIDEOS_PER_RUN", "1"))
LANG = os.getenv("LANG", "vi")            # ngôn ngữ TTS (gTTS)
VOICE = os.getenv("COQUI_VOICE", "en_vctk")  # voice cho Coqui (nếu dùng)

USE_COQUI = os.getenv("USE_COQUI", "0") == "1"

# ====== HELPERS ======
def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ASSETS_DIR).mkdir(parents=True, exist_ok=True)

def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

def run(cmd: List[str], check=True):
    print(">>", " ".join(cmd))
    return subprocess.run(cmd, check=check)

def save_text(path: pathlib.Path, content: str):
    path.write_text(content, encoding="utf-8")

def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")

# ====== 1) STORY GENERATION ======
def generate_story_with_openai(prompt_seed: str) -> str:
    api = os.getenv("OPENAI_API_KEY")
    if not api:
        raise RuntimeError("OPENAI_API_KEY not set")
    # lightweight prompt
    system = "You are a writer who crafts calm, cozy, rain-night bedtime stories in Vietnamese. Keep it PG, soothing, 1800-2300 words."
    user = f"Viết một truyện chill đêm mưa giọng kể gần gũi, nhiều âm thanh gợi tả (nhưng không rườm rà), chủ đề: {prompt_seed}. Chia đoạn ngắn dễ nghe."
    import openai
    openai.api_key = api
    # Use Responses API for latest SDKs if available; fallback to ChatCompletion
    try:
        resp = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.7,
        )
        return resp.choices[0].message["content"].strip()
    except Exception as e:
        # fallback: Responses API style (if using new SDK)
        from openai import OpenAI
        client = OpenAI(api_key=api)
        r = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.7,
        )
        return r.choices[0].message.content.strip()

def generate_story_with_gemini(prompt_seed: str) -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("No Gemini key")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={key}"
    payload = {"contents":[{"parts":[{"text": f"Viết truyện chill đêm mưa 1800-2300 từ, giọng kể nhẹ nhàng. Chủ đề: {prompt_seed}"}]}]}
    r = requests.post(url, json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()

def generate_story(seed_topics: List[str]) -> str:
    seed = random.choice(seed_topics)
    try:
        return generate_story_with_openai(seed)
    except Exception as e:
        print("OpenAI story failed:", e)
        try:
            return generate_story_with_gemini(seed)
        except Exception as e2:
            print("Gemini fallback failed:", e2)
            # last-resort: simple template
            return ("Đêm mưa rơi tí tách... " * 400)[:4000]

# ====== 2) SEO (title/desc/tags + thumbnail prompt) ======
def generate_seo(text: str):
    api = os.getenv("OPENAI_API_KEY")
    if not api:
        # simple fallback
        title = "Truyện Chill Đêm Mưa | Ngủ ngon an yên"
        desc = "Một câu chuyện nhẹ nhàng đưa bạn vào giấc ngủ giữa tiếng mưa rơi. Chúc bạn ngủ thật ngon!"
        tags = ["chill", "ngủ ngon", "tiếng mưa", "bedtime story", "relax"]
        thumb_prompt = "cozy room at night, gentle rain on window, warm lamp, cinematic"
        return title, desc, tags, thumb_prompt

    import openai
    openai.api_key = api
    prompt = (
        "Từ truyện sau, tạo:\n"
        "1) Một tiêu đề YouTube ngắn <= 72 ký tự, có keyword 'đêm mưa', 'chill', tự nhiên, không spam.\n"
        "2) Mô tả 2–3 đoạn (khoảng 250–400 chữ), kêu gọi subscribe nhẹ, có hashtag hợp lý.\n"
        "3) Danh sách 10–15 tag dạng CSV.\n"
        "4) Một prompt ngắn để vẽ thumbnail (tiếng Anh), phong cách ảnh thật 4K, cinematic rain-night.\n\n"
        f"Nội dung:\n{text[:4000]}"
    )
    try:
        resp = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role":"user","content":prompt}],
            temperature=0.6
        )
        raw = resp.choices[0].message["content"]
    except Exception:
        from openai import OpenAI
        client = OpenAI(api_key=api)
        r = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role":"user","content":prompt}],
            temperature=0.6
        )
        raw = r.choices[0].message.content

    # very small parser
    title = re.search(r"(?i)tiêu đề[:\-]\s*(.*)", raw)
    title = (title.group(1) if title else raw.splitlines()[0]).strip()[:72]
    tags_match = re.search(r"(?i)tag[s]?:\s*(.*)", raw)
    tags = [t.strip() for t in (tags_match.group(1) if tags_match else "chill,ngủ ngon,tiếng mưa").split(",") if t.strip()]
    thumb_prompt = re.search(r"(?i)thumbnail.*?:\s*(.*)", raw)
    thumb_prompt = (thumb_prompt.group(1) if thumb_prompt else "cozy room at night, rain on window, warm lamp, cinematic, 4k").strip()

    # description: take remainder
    desc = raw
    return title, desc, tags, thumb_prompt

# ====== 3) TTS ======
def tts_gtts(text: str, out_path: pathlib.Path, lang="vi"):
    from gtts import gTTS
    gTTS(text=text, lang=lang).save(str(out_path))

def tts_coqui(text: str, out_path: pathlib.Path, voice: str):
    # Coqui TTS CLI if available
    # model default tts_models/en/vctk/vits or vi model if preinstalled
    cmd = [
        "python", "-m", "TTS.bin.synthesize",
        "--text", text,
        "--model_name", os.getenv("COQUI_MODEL", "tts_models/en/vctk/vits"),
        "--out_path", str(out_path)
    ]
    run(cmd)

def make_tts(text: str, out_wav: pathlib.Path):
    if USE_COQUI:
        tts_coqui(text, out_wav, VOICE)
    else:
        # gTTS outputs mp3; convert to wav with ffmpeg
        tmp_mp3 = out_wav.with_suffix(".mp3")
        tts_gtts(text, tmp_mp3, LANG)
        run(["ffmpeg","-y","-i",str(tmp_mp3),"-ar","44100","-ac","2",str(out_wav)])
        tmp_mp3.unlink(missing_ok=True)

# ====== 4) Mix rain, normalize ======
def loop_rain_to_length(target_len_s: float, out_wav: pathlib.Path):
    if RAIN_FILE.exists():
        # ffmpeg loop
        run([
            "ffmpeg","-y","-stream_loop","-1","-t", str(target_len_s),
            "-i", str(RAIN_FILE),
            "-filter:a","volume=0.25",
            str(out_wav)
        ])
    else:
        # pink noise fallback (quiet)
        import numpy as np, soundfile as sf
        sr = 44100
        n = int(target_len_s*sr)
        noise = np.random.normal(0, 0.02, n).astype("float32")
        sf.write(out_wav, noise, sr)

def mix_voice_and_rain(voice_wav: pathlib.Path, rain_wav: pathlib.Path, out_wav: pathlib.Path):
    # duck the rain a bit when voice present
    run([
        "ffmpeg","-y",
        "-i", str(voice_wav),
        "-i", str(rain_wav),
        "-filter_complex",
        "[1:a]volume=0.25[a1];[0:a]dynaudnorm=f=75:g=15[mv];[mv][a1]amix=inputs=2:weight=1 1:duration=first,volume=1.0[out]",
        "-map","[out]","-ar","44100","-ac","2", str(out_wav)
    ])

# ====== 5) Thumbnail / Background ======
def generate_thumb_via_openai(prompt: str, out_path: pathlib.Path) -> bool:
    api = os.getenv("OPENAI_API_KEY")
    if not api:
        return False
    try:
        # Images API
        from openai import OpenAI
        client = OpenAI(api_key=api)
        r = client.images.generate(
            model=os.getenv("OPENAI_IMAGE_MODEL","gpt-image-1"),
            prompt=prompt,
            size="1024x1024",
            n=1
        )
        import base64
        b64 = r.data[0].b64_json
        img_bytes = base64.b64decode(b64)
        out_path.write_bytes(img_bytes)
        return True
    except Exception as e:
        print("OpenAI image failed:", e)
        return False

def make_fallback_thumb(text_title: str, out_path: pathlib.Path):
    img = Image.new("RGB", (1280, 720), (20, 24, 28))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 72) if FONT_PATH and pathlib.Path(FONT_PATH).exists() else ImageFont.load_default()
    # wrap title
    words = text_title.split()
    lines, line = [], ""
    for w in words:
        if len(line + " " + w) < 26:
            line = (line + " " + w).strip()
        else:
            lines.append(line); line = w
    if line: lines.append(line)
    y = 200
    for ln in lines[:4]:
        draw.text((80,y), ln, fill=(235, 235, 235), font=font)
        y += 90
    img.save(out_path)

# ====== 6) Render video ======
def render_video(bg_image: pathlib.Path, audio_wav: pathlib.Path, out_mp4: pathlib.Path):
    audio = AudioFileClip(str(audio_wav))
    img = ImageClip(str(bg_image)).set_duration(audio.duration)
    # subtle ken burns: zoom 1.0 -> 1.05
    w, h = img.size
    def zoom(get_frame, t):
        f = get_frame(t)
        z = 1.0 + 0.05*(t/audio.duration)
        return ImageClip(f).resize(z).get_frame(0)
    # moviepy's resize per-frame is heavy; simpler: static image
    video = img.set_audio(audio)
    # write
    video = video.fx(afx.audio_normalize)
    video.write_videofile(str(out_mp4), fps=30, codec="libx264", audio_codec="aac", threads=2, preset="veryfast", verbose=False, logger=None)
    audio.close(); video.close()

# ====== 7) Upload YouTube ======
def youtube_upload(path_mp4: pathlib.Path, title: str, description: str, tags: List[str]):
    # Using refresh_token flow
    client_id = os.getenv("YT_CLIENT_ID")
    client_secret = os.getenv("YT_CLIENT_SECRET")
    refresh_token = os.getenv("YT_REFRESH_TOKEN")
    if not (client_id and client_secret and refresh_token):
        print("⚠️ Missing YouTube secrets; skip upload. (YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN)")
        return False

    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    creds = Credentials(
        None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=["https://www.googleapis.com/auth/youtube.upload"]
    )
    service = build("youtube","v3", credentials=creds)
    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": tags[:15],
            "categoryId": "22"  # People & Blogs
        },
        "status": {
            "privacyStatus": os.getenv("YT_PRIVACY","public"),  # public|private|unlisted
            "selfDeclaredMadeForKids": False
        }
    }
    media = MediaFileUpload(str(path_mp4), chunksize=-1, resumable=True, mimetype="video/*")
    req = service.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        status, resp = req.next_chunk()
    print("✅ Uploaded:", resp.get("id"))
    return True

# ====== MAIN ======
def main():
    ensure_dirs()
    # Topics to vary stories
    topics = [
        "ký ức tuổi thơ ở làng quê", "tiệm sách nhỏ trong đêm", "hành trình tìm lại chính mình",
        "chuyến xe đêm về miền biển", "căn phòng trọ và cây đèn vàng", "mùi cà phê nóng và tiếng mưa",
    ]

    for i in range(VIDEOS_PER_RUN):
        uid = ts()
        work = OUT_DIR / f"job_{uid}"
        work.mkdir(parents=True, exist_ok=True)

        print("===> Generating story...")
        story = generate_story(topics)
        save_text(work / "story.txt", story)

        print("===> Generating SEO & thumbnail prompt...")
        title, description, tags, thumb_prompt = generate_seo(story)
        save_text(work / "title.txt", title)
        save_text(work / "description.txt", description)
        save_text(work / "tags.txt", ",".join(tags))
        save_text(work / "thumb_prompt.txt", thumb_prompt)

        print("===> TTS...")
        voice_wav = work / "voice.wav"
        make_tts(story, voice_wav)

        print("===> Make rain & mix...")
        # determine length
        import wave
        with wave.open(str(voice_wav), "rb") as w:
            frames = w.getnframes(); rate = w.getframerate()
            dur = frames / float(rate)
        rain_wav = work / "rain.wav"
        loop_rain_to_length(dur, rain_wav)
        mixed_wav = work / "audio_mix.wav"
        mix_voice_and_rain(voice_wav, rain_wav, mixed_wav)

        print("===> Thumbnail / Background...")
        thumb = work / "thumb.jpg"
        ok = generate_thumb_via_openai(thumb_prompt, thumb)
        if not ok:
            # fallback: use bg + big title
            make_fallback_thumb(title, thumb)

        print("===> Render video...")
        out_mp4 = work / f"{uid}.mp4"
        bg_img = thumb if thumb.exists() else (BG_IMAGE if BG_IMAGE.exists() else thumb)
        render_video(bg_img, mixed_wav, out_mp4)

        print("===> Upload YouTube...")
        uploaded = youtube_upload(out_mp4, title, description, tags)
        if uploaded:
            print("✅ DONE:", out_mp4)
        else:
            print("⚠️ Skipped upload; file saved at:", out_mp4)

if __name__ == "__main__":
    main()
