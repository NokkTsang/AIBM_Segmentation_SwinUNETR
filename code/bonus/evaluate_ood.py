"""Bonus task: evaluate the BTCV-trained Swin UNETR models on TotalSegmentator.

Uses the same Dice and Hausdorff implementation as Task 3 (imported from
code/task3/evaluate.py) and compares the result with the BTCV numbers in
code/task3/outputs/per_organ_summary.csv.

Two Dice values are reported per organ:
  dice      the whole scan, exactly as in Task 3;
  dice_abd  only the slices between the lowest and highest slice of the solid upper
            abdominal organs (liver, spleen, kidneys, stomach, gallbladder, pancreas,
            adrenals). TotalSegmentator labels aorta, esophagus and IVC through the
            chest and pelvis, BTCV does not, so this removes that labelling difference.

Run from the repository root after inference:
    python code/bonus/evaluate_ood.py
"""

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "task3"))
from evaluate import dice_score, hausdorff_mm, mean_finite, verify_alignment, write_csv  # noqa: E402

MODELS = ("base", "small", "tiny")  # default; override with --models
ABDOMEN_ANCHORS = (1, 2, 3, 4, 6, 7, 11, 12, 13)


def evaluate_case(job):
    import nibabel as nib

    model, case_id, pred_path, gt_path, labels = job
    pred_img, gt_img = nib.load(pred_path), nib.load(gt_path)
    verify_alignment(pred_img, gt_img, f"{model}/{case_id}")
    spacing = tuple(float(v) for v in nib.affines.voxel_sizes(gt_img.affine)[:3])
    pred = np.asanyarray(pred_img.dataobj)
    gt = np.asanyarray(gt_img.dataobj)

    z_axis = int(np.argmax(np.abs(gt_img.affine[2, :3])))
    other_axes = tuple(a for a in range(3) if a != z_axis)
    anchor_slices = np.flatnonzero(np.any(np.isin(gt, ABDOMEN_ANCHORS), axis=other_axes))
    window = [slice(None)] * 3
    window[z_axis] = slice(int(anchor_slices[0]), int(anchor_slices[-1]) + 1)
    window = tuple(window)

    rows = []
    for label_id, organ in labels.items():
        p, g = pred == label_id, gt == label_id
        hd, hd95 = hausdorff_mm(p, g, spacing)
        rows.append(dict(
            model=model, case=case_id, organ_id=label_id, organ=organ,
            dice=dice_score(p, g), dice_abd=dice_score(p[window], g[window]),
            hausdorff_mm=hd, hd95_mm=hd95,
            pred_voxels=int(p.sum()), gt_voxels=int(g.sum()),
            gt_voxels_abd=int(g[window].sum()),
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate Swin UNETR predictions on TotalSegmentator")
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "totalseg")
    parser.add_argument("--pred-dir", type=Path, default=REPO / "code" / "bonus" / "outputs")
    parser.add_argument("--output-dir", type=Path, default=REPO / "code" / "bonus" / "outputs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    args = parser.parse_args()
    models = tuple(args.models)

    dataset = json.loads((args.data_dir / "dataset_ood.json").read_text(encoding="utf-8"))
    labels = {int(k): v for k, v in dataset["labels"].items() if int(k) != 0}
    with (args.data_dir / "selection.csv").open(encoding="utf-8") as f:
        meta = {r["image_id"]: r for r in csv.DictReader(f)}

    jobs = []
    for model in models:
        for case in dataset["validation"]:
            name = Path(case["image"]).name
            pred_path = args.pred_dir / model / name
            if not pred_path.is_file():
                raise FileNotFoundError(f"Missing prediction: {pred_path}")
            jobs.append((model, name.replace(".nii.gz", ""), str(pred_path),
                         str(args.data_dir / case["label"]), labels))

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, case_rows in enumerate(pool.map(evaluate_case, jobs), 1):
            rows.extend(case_rows)
            r = case_rows[0]
            print(f"[{i}/{len(jobs)}] {r['model']} {r['case']}: "
                  f"mean Dice {np.mean([x['dice'] for x in case_rows]):.3f}", flush=True)

    for r in rows:
        study = meta[r["case"]]["study_type"]
        r["study_type"] = study
        r["institute"] = meta[r["case"]]["institute"]
        r["fov"] = "with_thorax" if any(k in study for k in ("thorax", "neck")) else "abdomen_only"

    out = args.output_dir
    write_csv(out / "ood_per_case_per_organ.csv", rows,
              ["model", "case", "study_type", "institute", "fov", "organ_id", "organ", "dice", "dice_abd",
               "hausdorff_mm", "hd95_mm", "pred_voxels", "gt_voxels", "gt_voxels_abd"])

    # per case
    per_case = []
    for model in models:
        for case in sorted({r["case"] for r in rows}):
            cr = [r for r in rows if r["model"] == model and r["case"] == case]
            per_case.append(dict(model=model, case=case, study_type=cr[0]["study_type"],
                                 institute=cr[0]["institute"], fov=cr[0]["fov"],
                                 mean_dice=float(np.mean([r["dice"] for r in cr])),
                                 mean_dice_abd=float(np.mean([r["dice_abd"] for r in cr])),
                                 mean_hausdorff_mm=mean_finite([r["hausdorff_mm"] for r in cr]),
                                 mean_hd95_mm=mean_finite([r["hd95_mm"] for r in cr]),
                                 undefined_hd_organs=sum(not np.isfinite(r["hd95_mm"]) for r in cr)))
    write_csv(out / "ood_per_case_summary.csv", per_case, list(per_case[0].keys()))

    # per organ, next to the BTCV numbers from Task 3
    with (REPO / "code" / "task3" / "outputs" / "per_organ_summary.csv").open(encoding="utf-8") as f:
        btcv = {(r["model"], int(r["organ_id"])): r for r in csv.DictReader(f)}
    per_organ = []
    for model in models:
        for label_id, organ in labels.items():
            orows = [r for r in rows if r["model"] == model and r["organ_id"] == label_id]
            b = btcv[(model, label_id)]
            d, da = float(np.mean([r["dice"] for r in orows])), float(np.mean([r["dice_abd"] for r in orows]))
            per_organ.append(dict(
                model=model, organ_id=label_id, organ=organ,
                btcv_dice=float(b["mean_dice"]), ood_dice=d, ood_dice_abd=da,
                dice_change=d - float(b["mean_dice"]), dice_change_abd=da - float(b["mean_dice"]),
                ood_dice_std=float(np.std([r["dice"] for r in orows])),
                btcv_hd95_mm=float(b["mean_hd95_mm"]), ood_hd95_mm=mean_finite([r["hd95_mm"] for r in orows]),
                btcv_hausdorff_mm=float(b["mean_hausdorff_mm"]),
                ood_hausdorff_mm=mean_finite([r["hausdorff_mm"] for r in orows]),
                ood_missing_predictions=sum(r["pred_voxels"] == 0 for r in orows), n_cases=len(orows)))
    write_csv(out / "ood_vs_btcv_per_organ.csv", per_organ, list(per_organ[0].keys()))

    # per model, overall and by field of view
    with (REPO / "code" / "task3" / "outputs" / "per_model_summary.csv").open(encoding="utf-8") as f:
        btcv_model = {r["model"]: r for r in csv.DictReader(f)}
    per_model = []
    for model in models:
        for group in ("all", "abdomen_only", "with_thorax"):
            cr = [r for r in per_case if r["model"] == model and group in ("all", r["fov"])]
            rr = [r for r in rows if r["model"] == model and group in ("all", r["fov"])]
            per_model.append(dict(
                model=model, group=group, n_cases=len(cr),
                btcv_dice=float(btcv_model[model]["mean_dice"]),
                ood_dice=float(np.mean([r["mean_dice"] for r in cr])),
                ood_dice_abd=float(np.mean([r["mean_dice_abd"] for r in cr])),
                btcv_hd95_mm=float(btcv_model[model]["mean_hd95_mm"]),
                ood_hd95_mm=mean_finite([r["hd95_mm"] for r in rr]),
                ood_hausdorff_mm=mean_finite([r["hausdorff_mm"] for r in rr])))
    write_csv(out / "ood_per_model_summary.csv", per_model, list(per_model[0].keys()))

    print("\nmodel  group         n  BTCV Dice  OOD Dice  OOD Dice (abdominal slices)  OOD HD95 mm")
    for r in per_model:
        print(f"{r['model']:6s} {r['group']:13s} {r['n_cases']:2d}  {r['btcv_dice']:.3f}      {r['ood_dice']:.3f}     "
              f"{r['ood_dice_abd']:.3f}                        {r['ood_hd95_mm']:.1f}")
    print(f"\nSaved CSVs to {out}")


if __name__ == "__main__":
    main()
