# Task 4 — Qualitative Evaluation (10 pts)

## Goal

Quantitative metrics (Task 3) don't fully describe segmentation quality — visually inspect
representative predictions from [code/task2/outputs/](../task2/outputs) against the CT images
([data/imagesTr/](../../data/imagesTr)) and ground-truth labels
([data/labelsTr/](../../data/labelsTr)).

## Suggested structure

```text
code/task4/
├── README.md
├── visualize.py       # generates slice overlays for selected cases
└── outputs/           # saved figures (PNG), used directly in the report
```