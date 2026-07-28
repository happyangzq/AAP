<div align="center">

# Asymmetric Anchoring Paradigm

### Opening the MLLM Black Box for Forgery Detection

[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-6A5ACD.svg)](#citation)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)](#installation)
[![PyTorch 1.13](https://img.shields.io/badge/PyTorch-1.13-EE4C2C.svg?logo=pytorch&logoColor=white)](#installation)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Official implementation of **Asymmetric Anchoring: Opening the MLLM Black Box
for Forgery Detection**.

</div>

## Overview

AAP exposes the forensic signals hidden inside a multimodal large language
model by treating its frozen visual encoder as a **truth anchor**.

- **Anchor real images.** The Anchoring Representation Loss (ARL) minimizes
  patch-wise cosine distance between intermediate MLLM features and
  penultimate visual-encoder features.
- **Preserve forgery discrepancies.** Tampered images are excluded from ARL,
  allowing their representation errors to remain visible.
- **Localize manipulations.** Patch-wise errors form a spatial discrepancy map,
  which is projected into the dense prompt space of the segmentation model.

The optimization objective is

$$
\mathcal{L} = \mathcal{L}_{\mathrm{cls}} + \mathcal{L}_{\mathrm{seg}} + 3\mathcal{L}_{\mathrm{ARL}}.
$$

This repository intentionally contains only the core training and evaluation
path. Datasets, checkpoints, experiment backups, logs, plots, and internal
infrastructure are not included.

## Project Structure

```text
.
├── aap/
│   ├── anchoring.py          # ARL and spatial discrepancy computation
│   ├── modeling_aap.py       # AAP model and loss integration
│   ├── data/                 # Dataset and class-balanced batch sampler
│   ├── llava/                # Minimal multimodal backbone components
│   └── segment_anything/     # Minimal segmentation components
├── scripts/
│   └── train.sh              # Reference training command
├── tests/
│   └── test_anchoring.py     # Tests for asymmetric anchoring
├── train.py                  # Training and evaluation entry point
└── requirements.txt
```

## Installation

The reference environment uses Python 3.10 and CUDA 11.7.

```bash
git clone https://github.com/happyangzq/AAP.git
cd AAP

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the SAM ViT-H checkpoint from the
[official checkpoint page](https://github.com/facebookresearch/segment-anything#model-checkpoints).
The base multimodal model and CLIP checkpoint are resolved through Hugging Face
on first use.

## Data Preparation

Organize each split as follows:

```text
dataset/
├── train/
│   ├── real/
│   ├── full_synthetic/
│   ├── tampered/
│   └── masks/
└── test/
    ├── real/
    ├── full_synthetic/
    ├── tampered/
    └── masks/
```

Images may use JPEG or PNG format. A tampered image named `example.png` must
have a binary mask named `example_mask.png`.

## Training

The reference configuration uses a global batch size of 16, 20 epochs, LoRA
alpha 16, LoRA dropout 0.05, and a fixed learning rate of `1e-4`.

```bash
DATASET_DIR=/path/to/dataset \
SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth \
bash scripts/train.sh
```

Equivalent direct invocation:

```bash
deepspeed train.py \
  --dataset-dir /path/to/dataset \
  --sam-checkpoint /path/to/sam_vit_h_4b8939.pth \
  --precision bf16 \
  --epochs 20 \
  --batch-size 8 \
  --anchor-loss-weight 3.0 \
  --target-layer 16
```

Adjust the per-device batch size and gradient accumulation according to the
available hardware while preserving the desired global batch size.

## Evaluation

```bash
deepspeed train.py \
  --eval-only \
  --resume runs/aap \
  --dataset-dir /path/to/dataset \
  --sam-checkpoint /path/to/sam_vit_h_4b8939.pth \
  --val-splits test
```

The evaluator reports three-class accuracy, macro F1, per-class F1,
localization IoU, and the confusion matrix.

## Tests

```bash
python -m pytest tests
```

## Acknowledgements

We sincerely thank the authors of **SIDA: Social Media Image Deepfake
Detection, Localization and Explanation with Large Multimodal Model** for
their inspiring work. If this repository is useful to your research, please
also consider citing:

```bibtex
@inproceedings{DBLP:conf/cvpr/HuangHLH00W0C25,
  author    = {Zhenglin Huang and Jinwei Hu and Xiangtai Li and Yiwei He and
               Xingyu Zhao and Bei Peng and Baoyuan Wu and Xiaowei Huang and
               Guangliang Cheng},
  title     = {{SIDA:} Social Media Image Deepfake Detection, Localization and
               Explanation with Large Multimodal Model},
  booktitle = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition
               (CVPR)},
  year      = {2025}
}
```

## Citation

If you find AAP useful, please cite:

```bibtex
@inproceedings{yang2026asymmetric,
  title     = {Asymmetric Anchoring: Opening the MLLM Black Box for Forgery
               Detection},
  author    = {Yang, Zhiqiang and Tao, Renshuai and Zhang, Chunjie and
               Liu, Zhaoxiang and Zheng, Xiaolong and Zhao, Yao},
  booktitle = {European Conference on Computer Vision},
  year      = {2026}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE). Third-party
attributions are listed in [NOTICE](NOTICE).
