# Implementation Plan

The goal of this project is to establish pixel-accurate semantic correspondences between two
images using features extracted from pretrained visual encoders.
Given
1. a source image with annotated keypoints, and
2. a target image: the task is to predict,
for each source keypoint, the corresponding location in the target image.

The project follows four stages:
1. Training-free baseline: use frozen features to perform semantic correspondence.
2. Light fine-tuning: adapt the last layers of the backbone to improve performance.
3. Better prediction rule: replace argmax with window soft-argmax.
4. Extension: LoRA-based efficient fine-tuning of last-block MLP layers.

---

## Training-free baseline:
To evaluate how well different models encode correspondence, we use SPair-71k, a standard benchmark for semantic correspondence. Each image pair in this dataset comes with annotated keypoints that represent the same semantic part (e.g., the tip of a dog’s ear, the wheel of a car) across different object instances or viewpoints.

For each pair of source and target images:
1. We extract dense features from a pretrained backbone.
2. For every source keypoint, we compute its cosine similarity with all patch features in the target image, producing a similarity map.
3. The location with the highest similarity is selected as the predicted match.

### Backbone Candidates
We will compare different pretrained encoders to understand how well they support correspondence:
1. DINOv2
2. DINOv3
3. Segment Anything (SAM)

### Evaluation Protocol
We follow the standard protocol from DIFT, using PCK@T (Percentage of Correct Keypoints) as the main metric. PCK measures the percentage of keypoints predicted within a certain normalized distance from the ground truth. We use multiple thresholds (e.g., 0.05, 0.1, 0.2) to analyze performance at different precision levels.

Results will be reported:
- Per keypoint
- Per image

This analysis will show how each backbone behaves across categories and difficulty levels

---

## Light Fine-tuning of the Last Layers
In the second stage, we keep the same pipeline but unfreeze the last layers of the backbone and fine-tune them using keypoint supervision from SPair-71k.
By testing different numbers of finetuned layers, we can observe how performance evolves as the model is given more flexibility to adapt to the task. This highlights how a small amount
of fine-tuning can significantly boost correspondence quality.

---

## Prediction
In the baselines above, the final correspondence is obtained using argmax on the similarity map. However, this has clear limitations:
1. it only predicts discrete pixel locations and
2. it is sensitive to local noise and can miss subtle details.

As proposed by Zhang et al. [3], we replace this with window soft-argmax:
1. Find the peak location with argmax.
2. Apply soft-argmax only within a small fixed window around the peak.
This allows sub-pixel refinement and makes the prediction more robust to noisy similarity
maps. In this step, you will evaluate how this change affects PCK across different thresholds.

---

## Mandatory Extension
After completing the main steps of the project, students are encouraged to explore and experiment. Below are some example directions you can take, but you are free to choose others: 
- Try different backbones beyond DINOv2 and SAM, such as Stable Diffusion features [1, 2]. 
- Test on new datasets, e.g., PF-Pascal, PF-Willow, or AP-10K, to see how well your method generalizes across domains. 
- Add Adapter [9], Adapformer [10] or LoRA [11] to explore efficient fine-tuning strategies. 
- Test on Geometric tasks, e.g. point tracking (DAVIS)

---

## Bibliography and references

[1] Tang et al., NeurIPS 2023 — Emergent Correspondence from Image Diffusion\
[2] Zhang et al., NeurIPS 2023 — A Tale of Two Features: Stable Diffusion Complements DINO for Zero-Shot Semantic Correspondence\
[3] Zhang et al., CVPR 2024 — Telling Left from Right – Identifying Geometry-Aware Semantic Correspondence\
[4] Min et al., ICCV 2019 — SPair-71k: A Large-scale Benchmark for Semantic Correspondence\
[5] Caron et al., ICCV 2021 — Emerging Properties in Self-Supervised Vision Transformers\
[6] Oquab et al., CVPR 2023 — DINOv2: Learning Robust Visual Features without Supervision\
[7] Simeoni et al., 2025 — DINOv3\
[8] Kirillov et al., ICCV 2023 — Segment Anything (SAM)

---

[9] Houlsby et al., ICML 2019 — Parameter-Efficient Transfer Learning for NLP\
[10] Chen et al., CVPR 2022 — AdaptFormer: Adapting Vision Transformers for Scalable Visual
Recognition\
[11] Hu et al., ICLR 2022 — LoRA: Low-Rank Adaptation of Large Language Models
