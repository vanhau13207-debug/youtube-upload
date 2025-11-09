#!/usr/bin/env python3
import os, datetime, json
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

VIDEO_PATH = "workspace/output/final_video.mp4"
THUMB_PATH = "workspace/output/thumbnail.jpg"
META_PATH = "workspace/output/meta.json"

CLIENT_ID = os.getenv("YT_CLIENT_ID")
CLIENT_SECRET = os.getenv("YT_CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("YT_REFRESH_TOKEN")

if not all([CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN]):
    raise Exception("Missing YouTube credentials in repo secrets!")

# build credentials
creds_info = {
    "token": "",
    "refresh_token": REFRESH_TOKEN,
    "token_uri": "https://oauth2.googleapis.com/token",
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
}
creds = Credentials.from_authorized_user_info(creds_info)

youtube = build("youtube", "v3", credentials=creds)

# compute publish time (next VN target slot logic)
now_vn = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
# slots (VN): 8,16,23 -> pick next
slots = [8,16,23]
for h in slots:
    candidate = now_vn.replace(hour=h, minute=0, second=0, microsecond=0)
    if now_vn.hour < h or (now_vn.hour == h and now_vn.minute < 1):
        publish_local = candidate
        break
else:
    publish_local = (now_vn + datetime.timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)

publish_utc = (publish_local - datetime.timedelta(hours=7)).isoformat() + "Z"
print("📅 Scheduling upload for", publish_utc)

if not os.path.exists(VIDEO_PATH):
    raise Exception("Video not found at " + VIDEO_PATH)

meta = {"title":"Auto Chill","description":"Auto video","tags": ["chill","rain"]}
if os.path.exists(META_PATH):
    try:
        meta = json.load(open(META_PATH,"r",encoding="utf-8"))
    except:
        pass

body = {
    "snippet": {
        "title": meta.get("title"),
        "description": meta.get("description"),
        "tags": meta.get("tags"),
        "categoryId": "22"
    },
    "status": {
        "privacyStatus":"private",
        "publishAt": publish_utc,
        "selfDeclaredMadeForKids": False
    }
}

media = VIDEO_PATH
request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
resp = request.execute()
vid = resp.get("id")
print("✅ Scheduled video", vid, "for", publish_utc)
# set thumbnail if exists
if os.path.exists(THUMB_PATH):
    try:
        youtube.thumbnails().set(videoId=vid, media_body=THUMB_PATH).execute()
        print("✅ Thumbnail set.")
    except Exception as e:
        print("⚠️ Thumbnail set failed:", e)
