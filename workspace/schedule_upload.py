import os
import datetime
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

def schedule_upload():
    creds = Credentials.from_authorized_user_info({
        "client_id": os.getenv("YT_CLIENT_ID"),
        "client_secret": os.getenv("YT_CLIENT_SECRET"),
        "refresh_token": os.getenv("YT_REFRESH_TOKEN")
    })

    youtube = build("youtube", "v3", credentials=creds)

    title = open("workspace/output/title.txt").read().strip()
    description = open("workspace/output/description.txt").read().strip()
    tags = open("workspace/output/tags.txt").read().strip().split(",")

    video_path = "workspace/output/final_video.mp4"
    thumb_path = "workspace/output/thumbnail.jpg"

    publish_time = (datetime.datetime.utcnow() + datetime.timedelta(hours=2)).isoformat() + "Z"

    upload = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": "22"
            },
            "status": {
                "privacyStatus": "private",
                "publishAt": publish_time
            }
        },
        media_body=video_path
    ).execute()

    youtube.thumbnails().set(
        videoId=upload["id"],
        media_body=thumb_path
    ).execute()

if __name__ == "__main__":
    schedule_upload()
