"""
Backward-compatible ``setup.py`` entry point for editable installs on older ``pip`` versions.

Prefer configuring metadata in ``pyproject.toml``; this file only wires setuptools discovery.
"""

from __future__ import annotations

from setuptools import find_packages, setup

setup(
    name="semantic-correspondence",
    version="0.1.0",
    packages=find_packages(exclude=("tests", "scripts")),
    python_requires=">=3.9",
)
