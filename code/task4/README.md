# Task 4 — Qualitative Evaluation (10 pts)

## Goal

Quantitative metrics (Task 3) don't fully describe segmentation quality — visually inspect
representative predictions from [code/task2/outputs/](../task2/outputs) against the CT images
([data/imagesTr/](../../data/imagesTr)) and ground-truth labels
([data/labelsTr/](../../data/labelsTr)).

## Run

From the repository root (needs `data/imagesTr`, `data/labelsTr`, `code/task2/outputs` and
`pip install nibabel matplotlib`):

```bash
python code/task4/visualize.py                                  # all six validation cases
python code/task4/visualize.py --cases img0038 --slices 68 78 85  # pick the axial slices yourself
python code/task4/visualize.py --cases img0038 --zoom sto lad     # organ close-ups
```

Figures are written to `code/task4/outputs/`:

- `compare_<case>.png`: CT, ground truth and Base / Small / Tiny on the same axial slices.
- `errors_<case>.png`: missed / extra / wrong-organ voxels on the slices with the most errors.
- `projection_<case>.png`: front-view projection of the whole label volume; shows stray
  false-positive blobs far from the organs.
- `zoom_<case>_<organ>.png`: one organ, prediction filled with the ground-truth outline in white.

## Structure

```text
code/task4/
├── README.md
├── visualize.py       # generates slice overlays for selected cases
└── outputs/           # saved figures (PNG), used directly in the report
```