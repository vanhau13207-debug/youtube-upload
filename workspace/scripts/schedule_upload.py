import os, datetime
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

def get_next_publish_time_vn():
    """Trả về giờ đăng kế tiếp: 8h, 16h, 23h VN"""
    now_utc = datetime.datetime.utcnow()
    vn_time = now_utc + datetime.timedelta(hours=7)
    today = vn_time.date()
    schedule_hours = [8, 16, 23]
    for h in schedule_hours:
        publish_vn = datetime.datetime(today.year, today.month, today.day, h)
        if vn_time < publish_vn:
            return publish_vn - datetime.timedelta(hours=7)
    next_day = today + datetime.timedelta(days=1)
    return datetime.datetime(next_day.year, next_day.month, next_day.day, 8) - datetime.timedelta(hours=7)

def youtube_upload_with_schedule():
    creds = Credentials.from_authorized_user_info({
        "client_id": os.getenv("YT_CLIENT_ID"),
        "client_secret": os.getenv("YT_CLIENT_SECRET"),
        "refresh_token": os.getenv("YT_REFRESH_TOKEN")
    })

    youtube = build("youtube", "v3", credentials=creds)
    video_path = "workspace/output/final_video.mp4"
    if not os.path.exists(video_path):
        print("❌ No video found to upload.")
        return

    title = "Relaxing Rainy Story 🌧️ | Chill Ambience for Sleep or Focus"
    description = "A calm, cinematic rainy story to relax, sleep, or focus. 🌙"
    tags = ["rain", "chill", "sleep", "relax", "asmr", "ambience", "storytelling"]

    publish_time = get_next_publish_time_vn()
    publish_iso = publish_time.isoformat("T") + "Z"
    vn_time_str = (publish_time + datetime.timedelta(hours=7)).strftime("%H:%M %d-%m-%Y")
    print(f"🕒 Scheduling upload for {vn_time_str} (VN time)")

    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": "24"
            },
            "status": {
                "privacyStatus": "private",
                "publishAt": publish_iso,
                "selfDeclaredMadeForKids": False
            }
        },
        media_body=video_path
    )
    response = request.execute()
    print("✅ Scheduled video uploaded:", response.get("id"))

if __name__ == "__main__":
    youtube_upload_with_schedule()
