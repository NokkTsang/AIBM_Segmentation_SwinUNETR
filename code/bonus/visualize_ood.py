"""Bonus task: the Task 4 figures, drawn for the TotalSegmentator cases.

Reuses code/task4/visualize.py unchanged and only points it at the bonus data.
Run from the repository root after prepare_totalseg.py, inference and evaluate_ood.py:

    python code/bonus/visualize_ood.py --cases s0441 s0543
    python code/bonus/visualize_ood.py --cases s0441 --zoom aorta eso

Figures are written to code/bonus/outputs/figures/.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "task4"))
import visualize as viz  # noqa: E402

DATA = REPO / "data" / "totalseg"
viz.IMAGES = DATA / "images"
viz.LABELS = DATA / "labels"
viz.PREDS = REPO / "code" / "bonus" / "outputs"
viz.METRICS = REPO / "code" / "bonus" / "outputs" / "ood_per_case_per_organ.csv"
viz.OUT = REPO / "code" / "bonus" / "outputs" / "figures"
viz.CASES = [
    Path(c["image"]).name.replace(".nii.gz", "")
    for c in json.loads((DATA / "dataset_ood.json").read_text(encoding="utf-8"))["validation"]
]


def load_case(case):
    ct, spacing = viz.load_las(viz.IMAGES / f"{case}.nii.gz")
    gt, _ = viz.load_las(viz.LABELS / f"{case}.nii.gz", is_label=True)
    preds = {}
    for model in viz.MODELS:
        path = viz.PREDS / model / f"{case}.nii.gz"
        if path.exists():
            preds[model], _ = viz.load_las(path, is_label=True)
    if not preds:
        raise FileNotFoundError(f"no predictions for {case} under {viz.PREDS}")
    return ct, gt, preds, spacing


viz.load_case = load_case

if __name__ == "__main__":
    viz.main()
