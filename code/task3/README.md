# Task 3 — Quantitative Evaluation (25 pts)

## Goal

For each of the three evaluated model configurations (Base / Small / Tiny), using the predicted
masks from [code/task2/outputs/](../task2/outputs) against the ground-truth labels in
[data/labelsTr/](../../data/labelsTr), compute:

- **Dice score** — overall and per-organ (13 BTCV organs, see label map in `data/dataset_0.json`).
- **Hausdorff distance** — overall and per-organ.
- **Organ-wise performance** table for each model.

## Suggested structure

```text
code/task3/
├── README.md
├── evaluate.py        # computes Dice + Hausdorff per case/organ for each model
└── outputs/           # per-model CSV/JSON metric tables, used by the report
```