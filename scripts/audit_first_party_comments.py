#!/usr/bin/env python3
"""Report non-ASCII characters in first-party Python (comments should stay ASCII)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_FIRST_PARTY_DIRS = ("data", "training", "evaluation", "models/common", "scripts", "utils", "tests")
_SKIP_PARTS = frozenset({"SPair-71k", "__pycache__"})


def main() -> int:
    bad: list[str] = []
    for dirname in _FIRST_PARTY_DIRS:
        root = _REPO / dirname
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if _SKIP_PARTS.intersection(path.parts):
                continue
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), 1):
                if any(ord(ch) > 127 for ch in line):
                    rel = path.relative_to(_REPO)
                    bad.append(f"{rel}:{lineno}: {line.rstrip()}")

    if bad:
        print("Non-ASCII in first-party Python (use ASCII in comments/docstrings):\n")
        print("\n".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
