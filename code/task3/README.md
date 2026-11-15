# Task 3 — Quantitative Evaluation (25 pts)

This section evaluates the saved BTCV segmentation predictions from Task 2 (Base, Small, Tiny) against the six held-out reference labels (`img0035`–`img0040`) using Dice, maximum Hausdorff distance (HD), and 95th-percentile HD (HD95).

## Run

Place the original six BTCV labels under `data/labelsTr/` and use the saved predictions under `code/task2/outputs/<model>/`. From the repository root:

```bash
python code/task3/evaluate.py
```

## Generated outputs

- `outputs/per_case_per_organ.csv`: Dice, HD, HD95, mask sizes and status for every model, case and organ (234 rows).
- `outputs/per_case_summary.csv`: 18 model–case means.
- `outputs/per_organ_summary.csv`: 39 model–organ means.
- `outputs/per_model_summary.csv`: 3 overall model means.
- `outputs/suggested_examples.json`: model–case pairs with highest and lowest mean Dice.
- `outputs/figures/organ_wise_dice.png`: comparison figure, if plotting script is run.

## Metric definitions and caveats

- Dice is computed per organ; when the reference mask is empty, Dice is recorded as `0` to match Task 2's convention.
- Symmetric maximum HD is the larger of the two directed maxima over 3D surface voxels.
- HD95 is the larger of the two directed 95th-percentiles over surface voxels (not a pooled-surface percentile). Both HD and HD95 use physical voxel spacing from the NIfTI affine.
- HD/HD95 are **undefined (NaN)** when either mask is empty. Undefined pairs are counted and excluded from the distance means; Dice includes all 13 organs.
- Models are compared on the **six-case validation split** identified in `data/dataset_0.json`, not on the original unlabeled BTCV test images.
- Task 4 can use the example IDs. Task 5 can use the summary tables. The bonus can reuse `dice_score` and `hausdorff_mm`, but needs its **own TotalSegmentator data loading and label mapping**; the current evaluation entry point is BTCV-specific.