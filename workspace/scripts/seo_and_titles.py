#!/usr/bin/env python3
# Generate simple SEO title/description/tags without paid APIs
# Usage: python3 scripts/seo_and_titles.py input_text.txt output_meta.json [duration_hours=2]

import sys, json, re, os

def sanitize(s): 
    return re.sub(r"\s+", " ", s).strip()

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/seo_and_titles.py input_text.txt output_meta.json [duration_hours=2]")
        sys.exit(1)

    in_txt = sys.argv[1]
    out_json = sys.argv[2]
    hours = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    with open(in_txt, "r", encoding="utf-8") as f:
        txt = f.read()

    # Very simple heuristics for theme
    first_line = sanitize(txt.splitlines()[0] if txt else "Chill Story")
    theme = first_line[:80] if first_line else "Midnight Rain"

    title = f"Midnight Rain – {hours}H Chill Bedtime Story 🌙 Calm Voice & Rain Ambience"
    description = (
        f"Relax with this {hours}-hour chill bedtime story narrated with a calm voice over gentle rain.\\n"
        "Perfect for sleep, studying, deep focus, and late-night relaxation.\\n\\n"
        "Chapters:\\n"
        "00:00 Intro\\n"
        "...\\n\\n"
        "#ChillStory #RainAmbience #SleepStory #ASMR #DeepFocus #BedtimeStory"
    )
    tags = [
        "chill story", "bedtime story", "sleep story", "rain ambience",
        "calm voice", "relaxing sounds", "asmr", "deep sleep", "focus music",
        "background rain", "story to sleep"
    ]

    payload = {
        "title": title,
        "description": description,
        "tags": tags,
        "theme": theme
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote SEO meta -> {os.path.abspath(out_json)}")

if __name__ == "__main__":
    main()