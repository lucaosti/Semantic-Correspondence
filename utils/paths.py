"""
Filesystem paths relative to the repository root (useful for configs and datasets).
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """
    Return the repository root directory.

    This assumes the file lives at ``<repo>/utils/paths.py``. For one-off scripts placed
    elsewhere, prefer passing paths explicitly.
    """
    return Path(__file__).resolve().parents[1]


def repo_root_env() -> Path:
    """
    Return ``REPO_ROOT`` from the environment if set, otherwise :func:`repo_root`.
    """
    env = os.environ.get("REPO_ROOT", "").strip()
    return Path(env).expanduser().resolve() if env else repo_root()
