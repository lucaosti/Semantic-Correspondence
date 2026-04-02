# Literature and model references

This repository implements semantic correspondence on **SPair-71k** with dense feature matching and optional fine-tuning / LoRA on late ViT blocks. For **upstream model references**, see module docstrings and loaders:

- [`models/dinov2/hub_loader.py`](../models/dinov2/hub_loader.py), [`models/dinov2/backbone.py`](../models/dinov2/backbone.py) — DINOv2
- [`models/dinov3/hub_loader.py`](../models/dinov3/hub_loader.py), [`models/dinov3/backbone.py`](../models/dinov3/backbone.py) — DINOv3
- [`models/sam/`](../models/sam/) — SAM (Meta architecture; weights downloaded separately)

**Benchmark:** SPair-71k is the standard split for category-level semantic correspondence; pair lists and annotations follow the usual community layout (see [`data/dataset.py`](../data/dataset.py)).

**Project rules** (splits, prefer official weights; Hugging Face mirrors allowed only as a fallback with SHA256 when available; WSA only at inference) are summarized in [`info.md`](info.md).

**State of the art:** [`state-of-art.md`](state-of-art.md) provides a comprehensive analysis of the design rationale, covering backbone selection, loss formulation (Gaussian CE), WSA, LoRA, photometric augmentation strategy, and PCK evaluation methodology.
