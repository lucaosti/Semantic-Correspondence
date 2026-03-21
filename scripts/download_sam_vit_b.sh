#!/usr/bin/env bash
# Download official Segment Anything ViT-B weights (~357 MiB) into ``checkpoints/``.
# Same URL as https://github.com/facebookresearch/segment-anything#model-checkpoints
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/checkpoints/sam_vit_b_01ec64.pth"
URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

mkdir -p "${ROOT}/checkpoints"
if [[ -f "${OUT}" ]]; then
  echo "Already present: ${OUT}"
  exit 0
fi
echo "Downloading ${URL} -> ${OUT}"
curl -L --fail -o "${OUT}" "${URL}"
echo "Done."
