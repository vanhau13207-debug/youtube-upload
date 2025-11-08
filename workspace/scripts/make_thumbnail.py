# workspace/make_thumbnail.py
import os
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip
import requests
import io
import textwrap

def extract_frame(video_path, t=2.0):
    """Extract frame at t seconds (default 2s) and return a PIL.Image."""
    clip = VideoFileClip(video_path)
    # if duration shorter, pick mid
    dur = clip.duration or 0
    sec = min(t, max(0.5, dur * 0.05))
    frame = clip.get_frame(sec)
    clip.reader.close()
    clip.audio = None
    img = Image.fromarray(frame)
    return img

def local_style_thumbnail(img: Image.Image, title_text: str, out_path: str, width=1280, height=720):
    """Apply local styling: crop/resize, mild blur, warm overlay, add small text."""
    # Resize & crop center to 16:9
    img = img.convert("RGB")
    img_ratio = img.width / img.height
    target_ratio = width / height
    if img_ratio > target_ratio:
        # crop width
        new_w = int(img.height * target_ratio)
        left = (img.width - new_w) // 2
        img = img.crop((left, 0, left + new_w, img.height))
    else:
        # crop height
        new_h = int(img.width / target_ratio)
        top = (img.height - new_h) // 2
        img = img.crop((0, top, img.width, top + new_h))
    img = img.resize((width, height), Image.LANCZOS)

    # mild gaussian blur background then blend with original (soften)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=6))
    base = Image.blend(img, blurred, alpha=0.25)

    # warm overlay
    overlay = Image.new("RGB", base.size, (20, 10, 30))
    mask = Image.new("L", base.size, 120)  # transparency
    base = Image.composite(base, overlay, mask)

    draw = ImageDraw.Draw(base)
    # add subtle vignette
    vignette = Image.new("L", base.size, 0)
    vdraw = ImageDraw.Draw(vignette)
    for i in range(0, 300, 10):
        vdraw.ellipse(
            (-i, -i, base.size[0] + i, base.size[1] + i),
            fill=int(120 - i * 0.12)
        )
    base.putalpha(vignette)

    # Title text: small, 2 lines max, bottom-left with padding
    font_path = os.environ.get("THUMB_FONT_PATH")  # optional custom font path
    try:
        font = ImageFont.truetype(font_path, 36) if font_path else ImageFont.truetype("arial.ttf", 36)
    except Exception:
        font = ImageFont.load_default()

    # prepare caption
    title_lines = textwrap.wrap(title_text, width=30)[:3]
    text = "\n".join(title_lines)
    padding = 28
    # draw rounded rectangle background
    text_w, text_h = draw.multiline_textsize(text, font=font, spacing=6)
    box_w = text_w + padding * 2
    box_h = text_h + padding
    box_x = 40
    box_y = height - box_h - 40

    # semi-transparent box
    box = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 140))
    base_rgba = base.convert("RGBA")
    base_rgba.paste(box, (box_x, box_y), box)

    # draw text
    draw = ImageDraw.Draw(base_rgba)
    draw.multiline_text((box_x+padding, box_y+padding//2), text, font=font, fill=(255,255,255,220), spacing=6)

    # Save
    base_rgb = base_rgba.convert("RGB")
    base_rgb.save(out_path, quality=92)
    return out_path

def call_thumbnail_ai(image: Image.Image, prompt: str, api_url: str, api_key: str=None):
    """
    Optional: send an image + prompt to a third-party image-to-image API and return PIL.Image.
    The function assumes the API accepts multipart/form-data with 'image' and 'prompt'.
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    files = {"image": ("frame.png", buf, "image/png")}
    data = {"prompt": prompt, "width": 1280, "height": 720}
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(api_url, files=files, data=data, headers=headers, timeout=120)
    resp.raise_for_status()
    out = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return out

def create_thumbnail(video_path, title_text, out_path="thumbnail.jpg"):
    print(f"[thumb] extracting frame from {video_path}")
    frame = extract_frame(video_path, t=2.0)
    # If THUMB_AI_API is set, call it
    api_url = os.environ.get("THUMB_AI_API")
    api_key = os.environ.get("THUMB_AI_KEY")
    prompt = os.environ.get("THUMB_PROMPT") or f"Enhance this cozy rainy scene, cinematic, warm tones, high detail, photorealistic, no people, negative space for small text."
    if api_url:
        try:
            print("[thumb] calling external AI thumbnail API...")
            ai_img = call_thumbnail_ai(frame, prompt, api_url, api_key)
            print("[thumb] AI returned image, post-processing locally...")
            ai_img.save(out_path, quality=92)
            # add title text locally to keep consistent
            local_style_thumbnail(ai_img, title_text, out_path)
            return out_path
        except Exception as e:
            print("[thumb] AI thumbnail failed:", e)
            # fallback to local
    print("[thumb] generating local styled thumbnail")
    return local_style_thumbnail(frame, title_text, out_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python make_thumbnail.py <video_path> <title_text> [out_path]")
        sys.exit(1)
    video_path = sys.argv[1]
    title_text = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) >= 4 else "thumbnail.jpg"
    p = create_thumbnail(video_path, title_text, out_path)
    print("Saved thumbnail:", p)
