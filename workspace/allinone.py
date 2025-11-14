#!/usr/bin/env python3
import os, json, time, base64, logging, datetime
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import numpy as np
import soundfile as sf

from moviepy.editor import VideoFileClip, ImageClip, AudioFileClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont

# Coqui
try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except:
    COQUI_AVAILABLE = False

# DIR
ROOT = Path.cwd()
WORKSPACE = ROOT / "workspace"
OUTPUT = WORKSPACE / "output"
ASSETS = WORKSPACE / "assets"

OUTPUT.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

# ENV
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_TEXT_MODEL = "gemini-2.0-flash"
GEMINI_IMAGE_MODEL = "gemini-2.0-flash"
COQUI_MODEL = os.getenv("COQUI_MODEL", "tts_models/en/vctk/vits")

RAIN_MP3 = Path(os.getenv("RAIN_FILE", ASSETS/"rain.mp3"))
RAIN_BG = Path(os.getenv("RAIN_BG", ASSETS/"rain_bg.mp4"))
MIN_DURATION = int(os.getenv("MIN_DURATION","7200"))
THUMB_AT = float(os.getenv("THUMB_FRAME_AT","10"))
AUDIO_SR = 22050
TTS_WPM = 150

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# GEMINI TEXT
def gemini_text(prompt: str):
    if not GEMINI_KEY:
        raise RuntimeError("Missing Gemini API key")

    url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"

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

# GEMINI IMAGE
def gemini_image(ref_b64: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_IMAGE_MODEL}:generateImage"
    payload={
        "prompt":{"text":"Create a high CTR YouTube thumbnail for chill rain ambience study/sleep."},
        "image_context":{"reference_images":[{"image_bytes":ref_b64}]},
        "format":"PNG"
    }
    r = requests.post(url, params={"key":GEMINI_KEY}, json=payload, timeout=90)
    r.raise_for_status()
    js = r.json()
    return base64.b64decode(js["images"][0]["imageBytes"])

# SEO FILES
def save_seo(title, desc, tags):
    (OUTPUT/"title.txt").write_text(title, encoding="utf-8")
    (OUTPUT/"description.txt").write_text(desc, encoding="utf-8")
    (OUTPUT/"tags.txt").write_text(",".join(tags), encoding="utf-8")

# ESTIMATE
def estimate_secs(t:str):
    return int((len(t.split())/TTS_WPM)*60)

# STORY MULTI-BATCH
def generate_story(seed:str, min_s:int):
    full=""
    tries=0
    base=(
        f"Write a long slow atmospheric English story for sleep/study.\n"
        f"Topic: {seed}. Calm, descriptive, 1400+ words.\n"
        "No titles. Only story paragraphs."
    )
    while True:
        tries+=1
        part=gemini_text(base)
        if not part.strip():
            part="The rain whispered softly as the quiet room embraced the night."
        full += "\n\n"+part
        secs=estimate_secs(full)
        logging.info(f"Batch {tries} -> est {secs}s")
        if secs>=min_s: break
        if tries>=12: break
        time.sleep(0.5)
    return full.strip()

# WAV duration
def wav_len(w):
    try:
        d,sr=sf.read(w,dtype='float32')
        if d.ndim>1: d=d.mean(axis=1)
        return len(d)/sr
    except:
        return 0

# TTS
def tts(text:str, out:Path):
    if not COQUI_AVAILABLE:
        raise RuntimeError("Coqui missing")
    tts=TTS(COQUI_MODEL)
    maxc=2500
    parts=[]; buf=""
    for p in text.split("\n"):
        if len(buf)+len(p)>maxc:
            parts.append(buf); buf=p
        else:
            buf=(buf+"\n"+p).strip()
    if buf: parts.append(buf)
    segs=[]
    for i,p in enumerate(parts):
        tmp=out.parent/f"tts_{i}.wav"
        tts.tts_to_file(text=p,file_path=str(tmp))
        segs.append(tmp)
    arr=[]; sr=None
    for s in segs:
        a,r=sf.read(s,dtype="float32")
        sr=r if sr is None else sr
        arr.append(a)
    full=np.concatenate(arr)
    sf.write(out, full, sr)
    for s in segs: s.unlink(missing_ok=True)

# SILENT
def silent(sec,out):
    d=np.zeros(int(sec*AUDIO_SR),dtype='float32')
    sf.write(out,d,AUDIO_SR)

# FRAME
def extract_frame(mp4:Path, at:float, out:Path):
    clip=VideoFileClip(str(mp4))
    t=min(max(0,at),clip.duration-0.001)
    frame=clip.get_frame(t)
    Image.fromarray(frame).save(out)
    clip.reader.close()
    logging.info(f"Frame {t}s -> {out}")

# LOOP AUDIO
def loop_audio(mp3:Path, dur:int, out:Path):
    d,sr=sf.read(mp3,dtype='float32')
    if d.ndim>1: d=d.mean(axis=1)
    need=int(dur*sr)
    rep=(need//len(d))+1
    full=np.tile(d,rep)[:need]
    sf.write(out,full,sr)

# MIX
def mix(vw:Path,rw:Path,dur:int,out:Path):
    v,sv=sf.read(vw,dtype='float32')
    r,sr=sf.read(rw,dtype='float32')
    if v.ndim>1:v=v.mean(axis=1)
    if r.ndim>1:r=r.mean(axis=1)
    if sv!=sr:
        v=np.interp(
            np.linspace(0,len(v),int(len(v)*sr/sv),endpoint=False),
            np.arange(len(v)),v
        )
    need=int(dur*sr)
    if len(r)<need: r=np.tile(r,(need//len(r))+1)[:need]
    else: r=r[:need]
    if len(v)<need: v=np.concatenate([v,np.zeros(need-len(v))])
    else: v=v[:need]

    v=v*1
    r=r*(10**(-12/20))
    mix=v+r
    m=np.max(np.abs(mix)); 
    if m>1: mix/=m
    sf.write(out,mix,sr)

# RENDER
def render(bg:Path,audio:Path,out:Path,dur:int):
    clip=VideoFileClip(str(bg))
    if clip.duration<dur:
        rep=int(np.ceil(dur/clip.duration))
        final=concatenate_videoclips([clip.copy() for _ in range(rep)]).subclip(0,dur)
    else:
        final=clip.subclip(0,dur)
    final=final.set_audio(AudioFileClip(str(audio)))
    final.write_videofile(str(out),fps=24,codec="libx264",audio_codec="aac",threads=2,verbose=False,logger=None)
    clip.reader.close()

# FALLBACK THUMB
def fallback_thumb(text,out):
    img=Image.new("RGB",(1280,720),(20,20,30))
    d=ImageDraw.Draw(img)
    try: font=ImageFont.truetype("DejaVuSans.ttf",48)
    except: font=ImageFont.load_default()
    d.text((60,260),text[:80],font=font,fill=(230,230,230))
    img.save(out)

# MAIN
def main():
    seed=os.getenv("VIDEO_SEED","Cozy rainy focus")
    story=generate_story(seed,MIN_DURATION)
    (OUTPUT/"story.txt").write_text(story,encoding="utf-8")

    # SEO
    seo_raw=gemini_text(
        "Make JSON title, description 400 words, tags array for chill rain study story.\n\n"+story[:1500]
    )
    try:
        seo=json.loads(seo_raw)
        title=seo.get("title","Rain Story")
        desc=seo.get("description","Rain ambience.")
        tags=seo.get("tags",["rain","relax"])
    except:
        title=f"Rain Story — {seed}"
        desc="Rain ambience with storytelling."
        tags=["rain","story","sleep"]
    save_seo(title,desc,tags)

    # Thumbnail
    frame=OUTPUT/"frame.png"
    thumb=OUTPUT/"thumbnail.jpg"
    try:
        extract_frame(RAIN_BG,THUMB_AT,frame)
        b64=base64.b64encode(frame.read_bytes()).decode()
        try:
            thumb.write_bytes(gemini_image(b64))
        except:
            thumb.write_bytes(frame.read_bytes())
    except:
        fallback_thumb(title,thumb)

    # TTS
    voice=OUTPUT/"voice.wav"
    try:
        tts(story,voice)
    except:
        silent(600,voice)

    vdur=int(wav_len(voice))
    final_dur=max(MIN_DURATION,vdur)

    rain_loop=OUTPUT/"rain_loop.wav"
    try:
        loop_audio(RAIN_MP3,final_dur,rain_loop)
    except:
        silent(final_dur,rain_loop)

    final_a=OUTPUT/"final_audio.wav"
    try:
        mix(voice,rain_loop,final_dur,final_a)
    except:
        silent(final_dur,final_a)

    final_v=OUTPUT/"final_video.mp4"
    try:
        render(RAIN_BG,final_a,final_v,final_dur)
    except:
        img=thumb if thumb.exists() else frame
        clip=ImageClip(str(img)).set_duration(final_dur)
        clip=clip.set_audio(AudioFileClip(str(final_a)))
        clip.write_videofile(str(final_v),fps=1,codec="libx264",audio_codec="aac",threads=2,verbose=False,logger=None)

    logging.info("DONE.")

if __name__=="__main__":
    main()

