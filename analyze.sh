#!/usr/bin/env bash
set -euo pipefail

API_URL="${WP_API_URL:-https://patrickfengsr--wp-film-coach-fastapi-app.modal.run}"

if [ $# -lt 1 ]; then
  echo "Usage: ./analyze.sh <youtube-url> [label]"
  exit 1
fi

URL="$1"
LABEL="${2:-}"
TMPFILE="$(mktemp -d)/game.mp4"

echo "⬇  Downloading video..."
yt-dlp -f "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720]/best" \
  --merge-output-format mp4 -o "$TMPFILE" "$URL"

echo "⬆  Uploading to Modal ($(du -h "$TMPFILE" | cut -f1))..."
RESPONSE=$(curl -s -X POST "$API_URL/process-upload" \
  -F "file=@$TMPFILE" \
  ${LABEL:+-F "label=$LABEL"})

JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

rm -f "$TMPFILE"

echo "✓  Job started: $JOB_ID"
echo "   Status: $API_URL/status/$JOB_ID"
