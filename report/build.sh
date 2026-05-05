#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEX_DIR="$SCRIPT_DIR/tex"
PDF_OUT="$SCRIPT_DIR/main.pdf"
PDFLATEX="${PDFLATEX:-/Library/TeX/texbin/pdflatex}"
BIBTEX="${BIBTEX:-/Library/TeX/texbin/bibtex}"

if ! command -v "$PDFLATEX" &>/dev/null; then
    echo "pdflatex not found. Install BasicTeX: brew install --cask basictex" >&2
    exit 1
fi

cd "$TEX_DIR"

echo "==> Pass 1/3: pdflatex"
"$PDFLATEX" -interaction=nonstopmode -halt-on-error main.tex

echo "==> bibtex"
"$BIBTEX" main

echo "==> Pass 2/3: pdflatex"
"$PDFLATEX" -interaction=nonstopmode main.tex

echo "==> Pass 3/3: pdflatex"
"$PDFLATEX" -interaction=nonstopmode main.tex

cp main.pdf "$PDF_OUT"
echo "==> Done: $PDF_OUT"
