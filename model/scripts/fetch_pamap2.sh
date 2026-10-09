#!/usr/bin/env bash
# Download and extract PAMAP2 into data/raw/pamap2 (git-ignored). ~656 MB zip, ~1.6 GB extracted.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/data/raw/pamap2"
URL="https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"

mkdir -p "$DEST"
if [ -d "$DEST/PAMAP2_Dataset/Protocol" ]; then
  echo "PAMAP2 already present at $DEST"; exit 0
fi
curl -fL -o "$DEST/../pamap2.zip" "$URL"
unzip -oq "$DEST/../pamap2.zip" -d "$DEST"
unzip -oq "$DEST/PAMAP2_Dataset.zip" -d "$DEST"
rm "$DEST/PAMAP2_Dataset.zip"
echo "Extracted to $DEST/PAMAP2_Dataset"
