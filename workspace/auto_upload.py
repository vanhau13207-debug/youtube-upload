#!/usr/bin/env python3
# auto_upload.py — upload final_video.mp4 + thumbnail + SEO to YouTube using refresh token
import os, json, logging, datetime
from pathlib import Path
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
OUTPUT = Path("workspace/output")
VIDEO = OUTPUT / "final_video.mp4"
THUMB = OUTPUT / "thumbnail.jpg"

# Secrets expected in env
CLIENT_ID = os.getenv("YT_CLIENT_ID")
CLIENT_SECRET = os.getenv("YT_CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("YT_REFRESH_TOKEN")
CHANNEL_ID = os.getenv("YT_CHANNEL_ID", "")

def get_youtube_service():
    if not (CLIENT_ID and CLIENT_SECRET and REFRESH_TOKEN):
        raise RuntimeError("Missing YouTube OAuth secrets")
    creds = Credentials(
        token=None,
        refresh_token=REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"]
    )
    request = Request()
    creds.refresh(request)
    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
    return youtube

def load_seo():
    title = (OUTPUT / "title.txt").read_text(encoding="utf-8") if (OUTPUT / "title.txt").exists() else "Relaxing Rainy Story"
    description = (OUTPUT / "description.txt").read_text(encoding="utf-8") if (OUTPUT / "description.txt").exists() else ""
    tags_raw = (OUTPUT / "tags.txt").read_text(encoding="utf-8") if (OUTPUT / "tags.txt").exists() else ""
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    return title, description, tags

def upload_video():
    if not VIDEO.exists():
        raise RuntimeError("No video to upload at " + str(VIDEO))
    youtube = get_youtube_service()
    title, description, tags = load_seo()
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "22"
        },
        "status": {
            "privacyStatus": "private"
        }
    }
    media = MediaFileUpload(str(VIDEO), chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    status = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            logging.info(f"Upload progress: {int(status.progress() * 100)}%")
    video_id = response.get("id")
    logging.info(f"Video uploaded with id {video_id}")
    # set thumbnail
    if THUMB.exists():
        thumb_media = MediaFileUpload(str(THUMB))
        youtube.thumbnails().set(videoId=video_id, media_body=thumb_media).execute()
        logging.info("Thumbnail set")
    # publish time: set to publish immediately or keep private. Here we keep private (could schedule)
    logging.info("Upload complete")
    return video_id

if __name__ == "__main__":
    upload_video()
