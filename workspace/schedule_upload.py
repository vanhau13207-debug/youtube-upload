#!/usr/bin/env python3
import os, datetime
import googleapiclient.discovery, googleapiclient.http
from google.oauth2.credentials import Credentials

OUTPUT_DIR = "workspace/output"
VIDEO_PATH = os.path.join(OUTPUT_DIR, "final_video.mp4")
THUMB_PATH = os.path.join(OUTPUT_DIR, "thumbnail.jpg")

TITLE = "Relaxing Rainy Story 🌧️ 2H Chill Sleep Ambience"
DESCRIPTION = """A calm storytelling video with gentle rain sounds — perfect for sleep, focus, or relaxation."""
TAGS = ["chill", "rain sounds", "storytelling", "relax", "sleep", "ASMR", "cozy", "ambient", "peaceful", "night"]

def next_vn_upload_time():
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
    slots = [8, 16, 23]
    for h in slots:
        if now.hour < h:
            next_time = now.replace(hour=h, minute=0, second=0)
            break
    else:
        next_time = (now + datetime.timedelta(days=1)).replace(hour=8, minute=0, second=0)
    return (next_time - datetime.timedelta(hours=7)).isoformat() + "Z"

def get_youtube_client():
    creds = Credentials(
        None,
        refresh_token=os.getenv("YT_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("YT_CLIENT_ID"),
        client_secret=os.getenv("YT_CLIENT_SECRET"),
    )
    return googleapiclient.discovery.build("youtube", "v3", credentials=creds)

def schedule_upload():
    youtube = get_youtube_client()
    publish_time = next_vn_upload_time()
    print(f"📅 Scheduling upload for {publish_time} (UTC)")
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ No video found at {VIDEO_PATH}")
        return
    req = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": TITLE,
                "description": DESCRIPTION,
                "tags": TAGS,
                "categoryId": "22",
            },
            "status": {
                "privacyStatus": "private",
                "publishAt": publish_time,
                "selfDeclaredMadeForKids": False,
            },
        },
        media_body=googleapiclient.http.MediaFileUpload(VIDEO_PATH, chunksize=-1, resumable=True),
    )
    res = req.execute()
    vid = res["id"]
    if os.path.exists(THUMB_PATH):
        youtube.thumbnails().set(
            videoId=vid,
            media_body=googleapiclient.http.MediaFileUpload(THUMB_PATH)
        ).execute()
    print(f"✅ Scheduled video {vid} for {publish_time}")

if __name__ == "__main__":
    schedule_upload()
