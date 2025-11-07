#!/usr/bin/env python3
# Minimal story fetcher: if a local .txt exists use it, else create a placeholder.
# In production you should use n8n HTTP Request node to fetch from Reddit JSON.
# Usage: python3 scripts/fetch_story_reddit.py output/story.txt

import sys, os, random, glob

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/fetch_story_reddit.py output/story.txt")
        sys.exit(1)

    out = sys.argv[1]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # Prefer random local story if available
    local = glob.glob("stories/*.txt")
    if local:
        import random
        src = random.choice(local)
        with open(src, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = (
            "A gentle night rain tapped against the window while the city breathed in slow, sleepy rhythms.\\n"
            "This is a placeholder story. Replace me with real stories fetched by n8n from Reddit r/shortstories.\\n"
        )

    with open(out, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[OK] Saved story -> {os.path.abspath(out)}")

if __name__ == "__main__":
    main()