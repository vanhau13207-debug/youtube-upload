import os
from datetime import datetime, timedelta, time
import json
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

UPLOAD_TIMES_VN = [
    time(8, 0),   # 08:00 VN
    time(16, 0),  # 16:00 VN
    time(23, 0),  # 23:00 VN
]

def next_upload_time_vn():
    now = datetime.utcnow() + timedelta(hours=7)
    today = now.date()

    for t in UPLOAD_TIMES_VN:
        sched = datetime.combine(today, t)
        if sched > now:
            return sched - timedelta(hours=7)

    # tomorrow 08:00 VN
    tomorrow = today + timedelta(days=1)
    sched = datetime.combine(tomorrow, UPLOAD_TIMES_VN[0])
    return sched - timedelta(hours=7)

def load_seo():
    with open("workspace/output/seo.txt") as f:
        txt = f.read()

    title = txt.split("TITLE:")[1].split("DESCRIPTION:")[0].strip()
    description = txt.split("DESCRIPTION:")[1].split("TAGS:")[0].strip()
    tags = txt.split("TAGS:")[1].strip().split(",")

    return title, description, tags

def upload_video():
    title, description, tags = load_seo()

    video_path = "workspace/output/final_video.mp4"
    thumb_path = "workspace/output/thumbnail.jpg"

    sched_utc = next_upload_time_vn()
    sched_iso = sched_utc.replace(microsecond=0).isoformat() + "Z"

    print("📅 Scheduling upload at:", sched_iso)

    creds = Credentials.from_authorized_user_info(
        {
            "client_id": os.getenv("YT_CLIENT_ID"),
            "client_secret": os.getenv("YT_CLIENT_SECRET"),
            "refresh_token": os.getenv("YT_REFRESH_TOKEN"),
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    )

    youtube = build("youtube", "v3", credentials=creds)

    request_body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags
        },
        "status": {
            "privacyStatus": "private",
            "publishAt": sched_iso,
            "selfDeclaredMadeForKids": False
        }
    }

    upload = youtube.videos().insert(
        part="snippet,status",
        body=request_body,
        media_body=video_path
    ).execute()

    # Thumbnail
    youtube.thumbnails().set(
        videoId=upload["id"],
        media_body=thumb_path
    ).execute()

    print("✅ Scheduled video:", upload["id"])

if __name__ == "__main__":
    upload_video()
