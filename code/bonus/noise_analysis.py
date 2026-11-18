"""Bonus task: image noise and contrast per scan, and how they relate to Dice.

For every TotalSegmentator case (and the six BTCV validation scans as reference) this measures
  liver_noise_hu   half the 16th-84th percentile range of the CT values inside the liver
                   mask after eroding it by 4 voxels (a robust standard deviation)
  liver_hu         median CT value in the liver mask
  aorta_hu         median CT value in the aorta mask
and reports the Spearman correlation of each model's mean Dice with noise and liver intensity.

Run from the repository root after evaluate_ood.py:
    python code/bonus/noise_analysis.py
Writes code/bonus/outputs/ood_noise_per_case.csv.
"""

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "code" / "bonus" / "outputs"
MODELS = ("base", "small", "tiny")
LIVER, AORTA = 6, 8


def measure(image_path, label_path):
    hu = np.asanyarray(nib.load(str(image_path)).dataobj).astype(np.float32)
    label = np.asanyarray(nib.load(str(label_path)).dataobj)
    core = hu[ndi.binary_erosion(label == LIVER, iterations=4)]
    noise = float(np.percentile(core, 84) - np.percentile(core, 16)) / 2
    return noise, float(np.median(hu[label == LIVER])), float(np.median(hu[label == AORTA]))


def main():
    with (OUT / "ood_per_case_summary.csv").open(encoding="utf-8") as f:
        dice = {(r["model"], r["case"]): float(r["mean_dice"]) for r in csv.DictReader(f)}
    dataset = json.loads((REPO / "data" / "totalseg" / "dataset_ood.json").read_text(encoding="utf-8"))

    rows = []
    for case in dataset["validation"]:
        cid = Path(case["image"]).name.replace(".nii.gz", "")
        noise, liver, aorta = measure(REPO / "data" / "totalseg" / case["image"], REPO / "data" / "totalseg" / case["label"])
        rows.append(dict(dataset="totalsegmentator", case=cid, liver_noise_hu=noise, liver_hu=liver, aorta_hu=aorta,
                         **{f"dice_{m}": dice[(m, cid)] for m in MODELS}))
    rows.sort(key=lambda r: r["dice_base"])

    btcv = json.loads((REPO / "data" / "dataset_0.json").read_text(encoding="utf-8"))
    for case in btcv["validation"]:
        cid = Path(case["image"]).name.replace(".nii.gz", "")
        noise, liver, aorta = measure(REPO / "data" / case["image"], REPO / "data" / case["label"])
        rows.append(dict(dataset="btcv", case=cid, liver_noise_hu=noise, liver_hu=liver, aorta_hu=aorta,
                         **{f"dice_{m}": "" for m in MODELS}))

    with (OUT / "ood_noise_per_case.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for r in rows:
        d = " ".join(f"{r[f'dice_{m}']:.3f}" if r[f"dice_{m}"] != "" else "  -  " for m in MODELS)
        print(f"{r['dataset']:16s} {r['case']} noise {r['liver_noise_hu']:6.1f} HU  liver {r['liver_hu']:4.0f}  "
              f"aorta {r['aorta_hu']:4.0f}  Dice base/small/tiny {d}")
    ts = [r for r in rows if r["dataset"] == "totalsegmentator"]
    print("\nSpearman correlation of mean Dice over the TotalSegmentator cases:")
    for m in MODELS:
        d = [r[f"dice_{m}"] for r in ts]
        rn, pn = stats.spearmanr(d, [r["liver_noise_hu"] for r in ts])
        rl, pl = stats.spearmanr(d, [r["liver_hu"] for r in ts])
        print(f"  {m:5s} with liver noise {rn:+.2f} (p={pn:.3f})   with liver intensity {rl:+.2f} (p={pl:.3f})")


if __name__ == "__main__":
    main()
