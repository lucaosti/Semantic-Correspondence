"""
Filesystem paths for SPair-71k (download location is **not** tracked in git).

Use environment variables so Colab/Drive or local disks work without code edits.
"""

from __future__ import annotations

import os
from typing import Optional

from data.dataset import default_spair_root


def resolve_spair_root(
    explicit: Optional[str] = None,
    *,
    dataset_parent: Optional[str] = None,
) -> str:
    """
    Resolve the ``SPair-71k`` root directory.

    Resolution order:

    1. ``explicit`` if provided.
    2. Environment variable ``SPAIR_ROOT`` (path to the ``SPair-71k`` folder itself).
    3. Environment variable ``DATASET_ROOT`` (parent directory that **contains**
       ``SPair-71k``, same semantics as :func:`data.dataset.default_spair_root`).
    4. ``dataset_parent`` if provided (parent of ``SPair-71k``).
    5. ``<repo>/data/SPair-71k`` as a conventional local layout (may not exist).

    Parameters
    ----------
    explicit:
        Direct path to ``SPair-71k``.
    dataset_parent:
        Parent folder containing ``SPair-71k`` (lowest priority among env/explicit).

    Returns
    -------
    str
        Absolute path to ``SPair-71k``.
    """
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))

    env_spair = os.environ.get("SPAIR_ROOT", "").strip()
    if env_spair:
        return os.path.abspath(os.path.expanduser(env_spair))

    env_parent = os.environ.get("DATASET_ROOT", "").strip()
    if env_parent:
        return default_spair_root(env_parent)

    if dataset_parent:
        return default_spair_root(dataset_parent)

    # Fallback: conventional folder inside the repo's `data/` directory (gitignored).
    from utils.paths import repo_root

    return default_spair_root(repo_root() / "data")
