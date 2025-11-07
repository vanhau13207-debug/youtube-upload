#!/usr/bin/env bash
# Mix voice with rain + optional music, render 2h video
# Usage: scripts/mix_and_render.sh <voice.wav> <bg.jpg> <rain.mp3> [music.mp3] [duration_seconds=7200] [output_mp4=output/final.mp4]

set -euo pipefail

VOICE="${1:-output/voice.wav}"
BG="${2:-assets/bg/bg1.jpg}"
RAIN="${3:-assets/rain.mp3}"
MUSIC="${4:-}"
DUR="${5:-7200}"
OUT="${6:-output/final.mp4}"

mkdir -p "$(dirname "$OUT")"

# Build inputs for ffmpeg filter
if [[ -n "$MUSIC" ]]; then
  # voice + rain + music
  ffmpeg -y -stream_loop -1 -i "$RAIN" -i "$VOICE" -stream_loop -1 -i "$MUSIC" \
    -filter_complex "[0:a]volume=0.25[a0];[1:a]volume=1.0[a1];[2:a]volume=0.15[a2];[a1][a0]amix=inputs=2:duration=longest:normalize=0[a12];[a12][a2]amix=inputs=2:duration=longest:normalize=0[aout]" \
    -map "[aout]" -t "$DUR" -ar 44100 -ac 2 output/mix.wav
else
  # voice + rain
  ffmpeg -y -stream_loop -1 -i "$RAIN" -i "$VOICE" \
    -filter_complex "[0:a]volume=0.25[a0];[1:a]volume=1.0[a1];[a1][a0]amix=inputs=2:duration=longest:normalize=0[aout]" \
    -map "[aout]" -t "$DUR" -ar 44100 -ac 2 output/mix.wav
fi

# Render video (CPU fast, near-lossless for still image)
ffmpeg -y -loop 1 -i "$BG" -i output/mix.wav \
  -c:v libx264 -preset veryfast -crf 23 -t "$DUR" -pix_fmt yuv420p -r 30 \
  -c:a aac -b:a 96k "$OUT"

echo "[OK] Rendered video -> $OUT"