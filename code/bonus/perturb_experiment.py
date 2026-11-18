"""Bonus task: does image noise cause the Base model's failures on TotalSegmentator?

Creates perturbed copies of a few prepared cases:
  <id>_smooth  Gaussian smoothing (sigma = 1 voxel = 1.5 mm) of scans on which Base failed
  <id>_noise   added Gaussian noise (sd = 50 HU) on scans on which Base worked
and writes data/totalseg/dataset_ood_perturb.json. Labels are copied unchanged.

    python code/bonus/perturb_experiment.py prepare
    # then run code/task2/test.py with --json_list dataset_ood_perturb.json
    # and --exp_name perturb_base / perturb_small (see the steps document)
    python code/bonus/perturb_experiment.py report
"""

import csv
import json
import re
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "totalseg"
OUT = REPO / "code" / "bonus" / "outputs"
SMOOTH = ["s0470", "s0441", "s1322"]   # Base failed, Small did not
NOISE = ["s0235", "s0236"]             # Base worked
SIGMA_VOXELS, NOISE_SD_HU, SEED = 1.0, 50.0, 0


def liver_noise(hu, label):
    values = hu[ndi.binary_erosion(label == 6, iterations=4)]
    return float(np.percentile(values, 84) - np.percentile(values, 16)) / 2


def prepare():
    rng = np.random.default_rng(SEED)
    cases = []
    for case, kind in [(c, "smooth") for c in SMOOTH] + [(c, "noise") for c in NOISE]:
        img = nib.load(str(DATA / "images" / f"{case}.nii.gz"))
        hu = np.asanyarray(img.dataobj).astype(np.float32)
        label = np.asanyarray(nib.load(str(DATA / "labels" / f"{case}.nii.gz")).dataobj)
        if kind == "smooth":
            new = ndi.gaussian_filter(hu, SIGMA_VOXELS)
        else:
            new = hu + rng.normal(0.0, NOISE_SD_HU, hu.shape).astype(np.float32)
        name = f"{case}_{kind}"
        nib.save(nib.Nifti1Image(np.round(new).astype(np.int16), img.affine), str(DATA / "images" / f"{name}.nii.gz"))
        shutil.copyfile(DATA / "labels" / f"{case}.nii.gz", DATA / "labels" / f"{name}.nii.gz")
        cases.append({"image": f"images/{name}.nii.gz", "label": f"labels/{name}.nii.gz"})
        print(f"{name}: liver noise {liver_noise(hu, label):.1f} -> {liver_noise(new, label):.1f} HU")
    dataset = json.loads((DATA / "dataset_ood.json").read_text(encoding="utf-8"))
    dataset["validation"] = cases
    (DATA / "dataset_ood_perturb.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")


def report():
    with (OUT / "ood_per_case_summary.csv").open(encoding="utf-8") as f:
        original = {(r["model"], r["case"]): float(r["mean_dice"]) for r in csv.DictReader(f)}
    rows = []
    for model in ("base", "small"):
        log = (OUT / f"inference_perturb_{model}.txt").read_text(encoding="utf-8", errors="ignore")
        for name, value in re.findall(r"case (\w+)\.nii\.gz\s+Mean Organ Dice: ([\d.eE+-]+)", log):
            case, kind = name.split("_")
            rows.append(dict(model=model, case=case, perturbation=kind,
                             dice_original=original[(model, case)], dice_perturbed=float(value)))
    with (OUT / "ood_perturbation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['model']:5s} {r['case']} {r['perturbation']:6s} {r['dice_original']:.3f} -> {r['dice_perturbed']:.3f}")


if __name__ == "__main__":
    {"prepare": prepare, "report": report}[sys.argv[1]]()
