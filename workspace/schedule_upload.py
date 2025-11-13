import os
import datetime
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

def schedule_upload():
    client_id = os.getenv("YT_CLIENT_ID")
    client_secret = os.getenv("YT_CLIENT_SECRET")
    refresh_token = os.getenv("YT_REFRESH_TOKEN")

    creds = Credentials.from_authorized_user_info({
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token
    })

    youtube = build("youtube", "v3", credentials=creds)

    video_path = "workspace/output/final_video.mp4"
    thumb_path = "workspace/output/thumbnail.jpg"

    title_path = "workspace/output/title.txt"
    desc_path = "workspace/output/description.txt"
    tags_path = "workspace/output/tags.txt"

    title = open(title_path).read().strip()
    description = open(desc_path).read().strip()
    tags = open(tags_path).read().strip().split(",")

    # publish sau 2 giờ
    publish_time = (datetime.datetime.utcnow() + datetime.timedelta(hours=2)).isoformat("T") + "Z"

    request = youtube.videos().insert(
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
    )

    response = request.execute()

    youtube.thumbnails().set(
        videoId=response["id"],
        media_body=thumb_path
    ).execute()

    print("🎉 Scheduled:", response["id"], publish_time)


if __name__ == "__main__":
    schedule_upload()
