#!/usr/bin/env python3
import os
import datetime
import googleapiclient.discovery
import googleapiclient.http
from google.oauth2.credentials import Credentials

# === CONFIG ===
OUTPUT_DIR = "workspace/output"
VIDEO_PATH = os.path.join(OUTPUT_DIR, "final_video.mp4")
THUMB_PATH = os.path.join(OUTPUT_DIR, "thumbnail.jpg")

TITLE = "Relaxing Rainy Night 🌧️ 2-Hour Chill Story"
DESCRIPTION = """A calm 2-hour storytelling video with gentle rain sounds, perfect for sleep, study, and relaxation.
Sit back, close your eyes, and enjoy the sound of rain with peaceful narration.
"""
TAGS = ["chill", "rain sounds", "storytelling", "relax", "sleep", "rainy night", "ASMR", "cozy", "peaceful", "study"]

# === DETERMINE NEXT UPLOAD TIME ===
def next_vn_upload_time():
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
    slots = [8, 16, 23]  # Giờ Việt Nam
    for h in slots:
        if now.hour < h:
            next_time = now.replace(hour=h, minute=0, second=0, microsecond=0)
            break
    else:
        next_time = (now + datetime.timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
    utc_time = next_time - datetime.timedelta(hours=7)
    return utc_time

# === AUTH ===
def get_youtube_client():
    creds = Credentials(
        None,
        refresh_token=os.getenv("YT_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("YT_CLIENT_ID"),
        client_secret=os.getenv("YT_CLIENT_SECRET"),
    )
    return googleapiclient.discovery.build("youtube", "v3", credentials=creds)

# === UPLOAD AND SCHEDULE ===
def schedule_upload():
    youtube = get_youtube_client()
    publish_time_utc = next_vn_upload_time().isoformat() + "Z"

    print(f"📅 Scheduling upload for: {publish_time_utc} (UTC)")

    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video file not found at: {VIDEO_PATH}")
        return

    request = youtube.videos().insert(
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
                "publishAt": publish_time_utc,
                "selfDeclaredMadeForKids": False,
            },
        },
        media_body=googleapiclient.http.MediaFileUpload(VIDEO_PATH, chunksize=-1, resumable=True),
    )

    response = request.execute()
    video_id = response["id"]
    print(f"✅ Video uploaded (ID: {video_id}), setting thumbnail...")

    if os.path.exists(THUMB_PATH):
        youtube.thumbnails().set(
            videoId=video_id,
            media_body=googleapiclient.http.MediaFileUpload(THUMB_PATH)
        ).execute()
        print("🖼️ Thumbnail uploaded successfully.")

    print(f"🎉 Scheduled video {video_id} for {publish_time_utc} (UTC)")

if __name__ == "__main__":
    schedule_upload()
