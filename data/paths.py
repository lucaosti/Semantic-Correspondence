"""
Filesystem paths for evaluation datasets (download locations are **not** tracked in git).

Use environment variables so Colab/Drive or local disks work without code edits.

Supported env vars
------------------
``SPAIR_ROOT``       — path to the ``SPair-71k`` folder itself.
``PF_WILLOW_ROOT``   — path to the ``PF-Willow`` folder itself.
``PF_PASCAL_ROOT``   — path to the ``PF-Pascal`` folder itself.
``DATASET_ROOT``     — parent directory that contains all dataset sub-folders
                       (used as fallback for every resolver).
"""

from __future__ import annotations

import os
from typing import Optional

from data.dataset import default_spair_root


def _resolve_dataset_dir(
    subfolder: str,
    explicit: Optional[str],
    env_key: str,
    *,
    dataset_parent: Optional[str],
) -> str:
    """Generic resolver shared by all dataset path helpers.

    Resolution order:
    1. ``explicit`` if provided.
    2. Environment variable ``env_key``.
    3. Environment variable ``DATASET_ROOT/<subfolder>``.
    4. ``dataset_parent/<subfolder>`` if provided.
    5. ``<repo>/data/<subfolder>`` (conventional local layout; may not exist).
    """
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))

    env_val = os.environ.get(env_key, "").strip()
    if env_val:
        return os.path.abspath(os.path.expanduser(env_val))

    env_parent = os.environ.get("DATASET_ROOT", "").strip()
    if env_parent:
        return os.path.join(os.path.abspath(os.path.expanduser(env_parent)), subfolder)

    if dataset_parent:
        return os.path.join(os.path.abspath(os.path.expanduser(str(dataset_parent))), subfolder)

    from utils.paths import repo_root

    return os.path.join(str(repo_root()), "data", subfolder)


def resolve_spair_root(
    explicit: Optional[str] = None,
    *,
    dataset_parent: Optional[str] = None,
) -> str:
    """Resolve the ``SPair-71k`` root directory.

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


def resolve_pf_willow_root(
    explicit: Optional[str] = None,
    *,
    dataset_parent: Optional[str] = None,
) -> str:
    """Resolve the ``PF-Willow`` root directory.

    Resolution order:

    1. ``explicit`` if provided.
    2. Environment variable ``PF_WILLOW_ROOT``.
    3. Environment variable ``DATASET_ROOT/PF-Willow``.
    4. ``dataset_parent/PF-Willow`` if provided.
    5. ``<repo>/data/PF-Willow`` (may not exist — run the download script first).

    Parameters
    ----------
    explicit:
        Direct path to the ``PF-Willow`` folder.
    dataset_parent:
        Parent folder containing ``PF-Willow`` (lowest priority).

    Returns
    -------
    str
        Absolute path to ``PF-Willow``.
    """
    return _resolve_dataset_dir(
        "PF-Willow",
        explicit,
        "PF_WILLOW_ROOT",
        dataset_parent=dataset_parent,
    )


def resolve_pf_pascal_root(
    explicit: Optional[str] = None,
    *,
    dataset_parent: Optional[str] = None,
) -> str:
    """Resolve the ``PF-Pascal`` root directory.

    Resolution order:

    1. ``explicit`` if provided.
    2. Environment variable ``PF_PASCAL_ROOT``.
    3. Environment variable ``DATASET_ROOT/PF-Pascal``.
    4. ``dataset_parent/PF-Pascal`` if provided.
    5. ``<repo>/data/PF-Pascal`` (may not exist — run the download script first).

    Parameters
    ----------
    explicit:
        Direct path to the ``PF-Pascal`` folder.
    dataset_parent:
        Parent folder containing ``PF-Pascal`` (lowest priority).

    Returns
    -------
    str
        Absolute path to ``PF-Pascal``.
    """
    return _resolve_dataset_dir(
        "PF-Pascal",
        explicit,
        "PF_PASCAL_ROOT",
        dataset_parent=dataset_parent,
    )
