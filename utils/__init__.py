"""
Project-wide helpers (paths, logging, checkpoints).

Keep imports lightweight; heavy dependencies belong next to the features that need them.
"""

from utils.paths import repo_root, repo_root_env

__all__ = ["repo_root", "repo_root_env"]
