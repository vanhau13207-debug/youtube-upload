#!/usr/bin/env python3
# Create a thumbnail using free Pollinations image API (no key needed)
# Usage: python3 scripts/make_thumbnail.py "prompt text" output/thumbnail.jpg

import sys, os, urllib.parse, requests

def main():
    if len(sys.argv) < 3:
        print('Usage: python3 scripts/make_thumbnail.py "prompt text" output/thumbnail.jpg')
        sys.exit(1)

    prompt = sys.argv[1]
    out = sys.argv[2]
    url = "https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(out, "wb") as f:
        f.write(r.content)
    print(f"[OK] Thumbnail saved -> {os.path.abspath(out)}")

if __name__ == "__main__":
    main()