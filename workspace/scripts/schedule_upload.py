#!/usr/bin/env python3
import os
import datetime
import json
import googleapiclient.discovery
import google_auth_oauthlib.flow
import google.auth.transport.requests
from google.oauth2.credentials import Credentials

# === CONFIG ===
OUTPUT_DIR = "workspace/output"
VIDEO_PATH = os.path.join(OUTPUT_DIR, "final_video.mp4")
THUMB_PATH = os.path.join(OUTPUT_DIR, "thumbnail.jpg")

TITLE = "Rainy Chill Night 🌧️ Relaxing Story & Ambience"
DESCRIPTION = """
A calm and cozy 2-hour storytelling video with soft rain sounds and peaceful vibes.
Perfect for sleep, study, or relaxation 🌙☕
"""
TAGS = ["chill", "rain sounds", "sleep", "study", "storytelling", "relax", "rainy night"]

# === Get next upload time ===
def next_vn_upload_time():
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
    slots = [8, 16, 23]
    for h in slots:
        if now.hour < h:
            next_time = now.replace(hour=h, minute=0, second=0, microsecond=0)
            break
    else:
        next_time = (now + datetime.timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
    return next_time - datetime.timedelta(hours=7)  # convert back to UTC

# === Auth ===
def get_youtube_client():
    creds_data = {
        "installed": {
            "client_id": os.getenv("YT_CLIENT_ID"),
            "client_secret": os.getenv("YT_CLIENT_SECRET"),
            "redirect_uris": ["http://localhost"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token"
        }
    }

    creds = Credentials(
        None,
        refresh_token=os.getenv("YT_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("YT_CLIENT_ID"),
        client_secret=os.getenv("YT_CLIENT_SECRET"),
    )
    return googleapiclient.discovery.build("youtube", "v3", credentials=creds)

# === Upload & Schedule ===
def schedule_upload():
    youtube = get_youtube_client()
    publish_time_utc = next_vn_upload_time().isoformat() + "Z"

    print(f"📅 Scheduling upload for {publish_time_utc} (UTC)")

    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": TITLE,
                "description": DESCRIPTION,
                "tags": TAGS,
                "categoryId": "22"
            },
            "status": {
                "privacyStatus": "private",
                "publishAt": publish_time_utc,
                "selfDeclaredMadeForKids": False
            }
        },
        media_body=googleapiclient.http.MediaFileUpload(VIDEO_PATH, chunksize=-1, resumable=True)
    )

    response = request.execute()
    print("✅ Video uploaded, setting thumbnail...")

    youtube.thumbnails().set(
        videoId=response["id"],
        media_body=THUMB_PATH
    ).execute()

    print(f"🎉 Scheduled video {response['id']} for {publish_time_utc}")

if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print("❌ No video found at", VIDEO_PATH)
    else:
        schedule_upload()
