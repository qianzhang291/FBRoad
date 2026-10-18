# FBRoad

**Fusion and Boundary Refinement for Road Extraction**

PyTorch implementation of **FBRoad**, a two-stage framework for extracting road regions from high-resolution aerial imagery and aligned vehicle GPS trajectory heatmaps.

The method combines:
1. **Cross-Modal Region Network**: A dual-stream encoder–decoder built on CMMPNet with a Cross-Modal Feature Fusion Module (FFM) at the bottleneck, a gated four-direction strip-convolution decoder branch, late-stage pixel-contrastive and Lovász-hinge objectives, and Exponential Moving Average (EMA) model selection.
2. **Boundary & Direction Refiner**: An independent HRNet2x-W20 SegFix boundary and 8-direction offset corrector operating at native $1024 \times 1024$ resolution.

---

## Repository Structure

```text
src/fbroad/       Core models, losses, EMA, dataset loaders, and training/inference routines
scripts/              Command-line entry points for initialization, training, inference, and evaluation
configs/              Dataset split manifest (split.json) and stage hyperparameter configurations
licenses/             Upstream license texts for adapted third-party components
tests/                Unit tests for model architectures, losses, and evaluation metrics
```

---

## Installation

Requires Python $\ge 3.10$ and PyTorch $\ge 2.7$ with matching `torchvision`.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

---

## Dataset Preparation

Download the preprocessed **BJRoad** dataset from the [CMMPNet repository](https://github.com/liulingbo918/CMMPNet) and organize the directory as follows:

```text
/path/to/BJRoad/
├── train_val/
│   ├── image/        # <ID>_sat.png (1024×1024 aerial imagery)
│   ├── gps/          # <ID>_gps.jpg (1024×1024 rendered GPS heatmap)
│   └── mask/         # <ID>_mask.png (1024×1024 binary road mask)
└── test/
    ├── image/
    ├── gps/
    └── mask/
```

Fixed training (1,701 patches from 243 mother scenes), validation (35 scenes), and test (70 scenes) identifiers are defined in [`configs/split.json`](configs/split.json).

---

## Training

### 1. Prepare Common Initialization & Train Region Network

Create a seeded ResNet-34 ImageNet initialization and train the region extraction network (30 epochs, batch size 8, Adam optimizer with $10^{-4} \to 10^{-5}$ learning rate schedule):

```bash
python scripts/prepare_initial.py \
  --imagenet \
  --seed 23456 \
  --output outputs/common23456.pt

python scripts/train_region.py \
  --data-root /path/to/BJRoad \
  --arm ema \
  --seed 23456 \
  --initial outputs/common23456.pt \
  --output outputs/ema23456
```

Supported `--arm` variants for ablation studies: `baseline`, `fusion`, `direction`, `contrast`, `lovasz`, and `ema`.

### 2. Train Boundary & Direction Corrector

Obtain the HRNet2x-W20 ImageNet pretrained backbone (`hr_rnet_bt_w20_imagenet_pretrained.pth`) from the [SegFix Model Zoo](https://github.com/openseg-group/openseg.pytorch/blob/aefc75517b09068d7131a69420bc5f66cb41f0ee/MODEL_ZOO.md), generate validation reference masks, and train the boundary refiner (60 epochs, batch size 4, $512 \times 512$ crops, polynomial SGD schedule):

```bash
python scripts/train_region.py \
  --data-root /path/to/BJRoad \
  --arm lovasz \
  --seed 23456 \
  --initial outputs/common23456.pt \
  --output outputs/lovasz23456

python scripts/predict_split.py \
  --data-root /path/to/BJRoad \
  --part validation \
  --arm lovasz \
  --region-checkpoint outputs/lovasz23456/best.pt \
  --device cuda \
  --output outputs/lovasz23456/validation_masks

python scripts/train_boundary.py \
  --data-root /path/to/BJRoad \
  --seed 23456 \
  --imagenet /path/to/hr_rnet_bt_w20_imagenet_pretrained.pth \
  --validation-reference outputs/lovasz23456/validation_masks \
  --target-root outputs/boundary23456/targets \
  --output outputs/boundary23456/run
```

---

## Inference and Evaluation

### Single-Scene Inference

Run road extraction on an aligned aerial image and GPS heatmap pair (omit `--boundary-checkpoint` to run the region network only):

```bash
python scripts/predict.py \
  --image scene_sat.png \
  --gps scene_gps.jpg \
  --region-checkpoint outputs/ema23456/best.pt \
  --boundary-checkpoint outputs/boundary23456/run/best.pt \
  --device cuda \
  --output outputs/scene_pred.png
```

### Split Evaluation

Generate predictions on the test split and compute global foreground IoU, precision, recall, F1, and per-scene hard clDice:

```bash
python scripts/predict_split.py \
  --data-root /path/to/BJRoad \
  --part test \
  --region-checkpoint outputs/ema23456/best.pt \
  --boundary-checkpoint outputs/boundary23456/run/best.pt \
  --device cuda \
  --output outputs/complete23456/test_masks

python scripts/evaluate.py \
  --data-root /path/to/BJRoad \
  --part test \
  --predictions outputs/complete23456/test_masks \
  --output outputs/complete23456/metrics.json
```

---

## Acknowledgements and License

This project builds upon [CMMPNet](https://github.com/liulingbo918/CMMPNet), [CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation), [Sun et al.](https://github.com/suniique/Leveraging-Crowdsourced-GPS-Data-for-Road-Extraction-from-Aerial-Imagery), [SegFix](https://github.com/openseg-group/openseg.pytorch), [timm](https://github.com/huggingface/pytorch-image-models), [DICN](https://github.com/BubbleSai/DICN), and [Lovász-Softmax](https://github.com/bermanmaxim/LovaszSoftmax).

Released under the [MIT License](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [`licenses/`](licenses/) for upstream component attributions.
