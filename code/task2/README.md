# Task 2 — Reproduce BTCV Segmentation Results

Runs sliding-window inference with the pretrained **Swin UNETR** checkpoints (Base / Small / Tiny)
on the BTCV validation split and saves the predicted segmentation masks, printing per-case and
overall mean Dice.

Source: adapted from the official
[Project-MONAI/research-contributions SwinUNETR/BTCV](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV)
`test.py`.

## Files

| File | Purpose |
|------|---------|
| `test.py` | Entry point: builds the `SwinUNETR` model, loads a checkpoint, runs sliding-window inference over the validation set, computes per-organ/mean Dice, and writes predicted `.nii.gz` masks. |
| `utils/data_utils.py` | MONAI `LoadImaged`/`Spacingd`/`ScaleIntensityRanged`/... transform pipelines, dataset construction from `dataset_0.json`, and a `DistributedSampler`-like `Sampler`. |
| `utils/utils.py` | `dice()` metric, `resample_3d()` (nearest-neighbor resampling of predictions back to original label spacing), `AverageMeter`, and distributed-training gather helpers (unused at inference time). |
| `outputs/base/`, `outputs/small/`, `outputs/tiny/` | Predicted masks (`imgXXXX.nii.gz`) for each of the 6 validation cases (`img0035`–`img0040`), one subfolder per model configuration. |

## Prerequisites

- Dependencies installed (`pip install -r requirements.txt` from repo root).
- BTCV data placed under `data/` at the repo root (see [data/README.md](../../data/README.md)).
- Pretrained checkpoints placed under `weights/` at the repo root (see [weights/README.md](../../weights/README.md)).

## Usage

Run from inside `code/task2/` (relative paths like `./outputs/` and `--data_dir`/`--pretrained_dir`
are resolved from the current directory):

```bash
cd code/task2
```

**Swin UNETR / Base** (`feature_size=48`):
```bash
python test.py \
  --data_dir ../../data \
  --json_list dataset_0.json \
  --pretrained_dir ../../weights \
  --pretrained_model_name swinunetr_base.pt \
  --feature_size 48 \
  --infer_overlap 0.7 \
  --exp_name base
```

**Swin UNETR / Small** (`feature_size=24`):
```bash
python test.py \
  --data_dir ../../data \
  --json_list dataset_0.json \
  --pretrained_dir ../../weights \
  --pretrained_model_name swinunetr_small.pt \
  --feature_size 24 \
  --infer_overlap 0.7 \
  --exp_name small
```

**Swin UNETR / Tiny** (`feature_size=12`):
```bash
python test.py \
  --data_dir ../../data \
  --json_list dataset_0.json \
  --pretrained_dir ../../weights \
  --pretrained_model_name swinunetr_tiny.pt \
  --feature_size 12 \
  --infer_overlap 0.7 \
  --exp_name tiny
```

(`--data_dir`/`--pretrained_dir` also accept absolute paths if you'd rather not `cd` into
`code/task2/` first.)

Each run:
- Loads the 6 validation cases listed in `data/dataset_0.json` (`img0035`–`img0040`).
- Performs sliding-window inference (`roi=(96, 96, 96)`, Gaussian blending). **`--infer_overlap 0.7`
  was used to produce the committed `outputs/`** (the script default is `0.5` — pass `0.7`
  explicitly to reproduce the numbers below; this matches the paper's higher-overlap setting).
- Prints the mean Dice over the 13 foreground organs for each case and the overall mean Dice.
- Saves predicted masks to `./outputs/<exp_name>/imgXXXX.nii.gz`.

Key CLI options (see `test.py` for the full list): `--infer_overlap`, `--roi_x/y/z`,
`--a_min/a_max/b_min/b_max` (CT intensity windowing), `--space_x/y/z` (resampling spacing),
`--use_checkpoint` (gradient checkpointing, mainly relevant for training).

## Results (overlap = 0.7)

Mean Dice over the 13 foreground organs, per validation case (`dice()` from `utils/utils.py`,
as printed by `test.py`):

| Case | Base (f=48) | Small (f=24) | Tiny (f=12) |
|------|:---:|:---:|:---:|
| img0035 | 0.7750 | 0.7411 | 0.6423 |
| img0036 | 0.8378 | 0.8113 | 0.6498 |
| img0037 | 0.8224 | 0.7920 | 0.6788 |
| img0038 | 0.7969 | 0.7349 | 0.6121 |
| img0039 | 0.8437 | 0.8313 | 0.7461 |
| img0040 | 0.8257 | 0.8266 | 0.6681 |
| **Overall mean** | **0.8169** | **0.7895** | **0.6662** |

As expected, Dice ranking follows model capacity: Base > Small > Tiny. These are **mean Dice
averaged across all 13 organs per case** (not organ-wise) — per-organ Dice and Hausdorff distance
are computed in Task 3, which reuses the `outputs/` masks below instead of re-running inference.

## Output

`outputs/<base|small|tiny>/imgXXXX.nii.gz` — one uint8-labeled NIfTI volume per validation case,
in the original label's affine/orientation, with values `0` (background) through `13` (one per
BTCV organ, see `data/dataset_0.json` for the label map). These are small (~0.2–0.4 MB each) and
are committed to the repo as evidence of the reproduced run; they are also reused directly by
Task 3 (quantitative metrics) and Task 4 (qualitative visualization) instead of re-running
inference.