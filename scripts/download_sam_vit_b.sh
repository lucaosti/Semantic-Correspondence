#!/usr/bin/env bash
# Download official Segment Anything ViT-B weights (~357 MiB) into ``checkpoints/``.
# Same URL as https://github.com/facebookresearch/segment-anything#model-checkpoints
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/checkpoints/sam_vit_b_01ec64.pth"
URLS=(
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
  "https://huggingface.co/scenario-labs/sam_vit/resolve/1c5c33ad1abe579854e3e2a6228568026aea6758/sam_vit_b_01ec64.pth?download=true"
  "https://huggingface.co/fofr/comfyui/resolve/76574336fbb61a96825dbe0b41bda2f2ec214084/sams/sam_vit_b_01ec64.pth?download=true"
)

mkdir -p "${ROOT}/checkpoints"
if [[ -f "${OUT}" ]]; then
  echo "Already present: ${OUT}"
  exit 0
fi

tmp="${OUT}.tmp"
trap 'rm -f "${tmp}"' EXIT

for url in "${URLS[@]}"; do
  echo "Trying ${url}"
  if curl -L --fail -o "${tmp}" "${url}"; then
    mv "${tmp}" "${OUT}"
    echo "Done: ${OUT}"
    exit 0
  fi
done

echo "ERROR: failed to download SAM ViT-B from all configured URLs." >&2
exit 1
