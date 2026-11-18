"""Bonus task: turn TotalSegmentator subjects into BTCV-style cases.

For every selected subject this script
  1. merges the 13 binary masks that have a BTCV counterpart into one label volume
     that uses the BTCV label numbers,
  2. reorients CT and label to LAS, the orientation of the BTCV scans that
     code/task2/test.py feeds to the network without reorienting,
  3. writes data/totalseg/images/<id>.nii.gz and data/totalseg/labels/<id>.nii.gz,
and finally writes data/totalseg/dataset_ood.json (same format as data/dataset_0.json)
plus data/totalseg/selection.csv describing the chosen subjects.

Run from the repository root:
    python code/bonus/prepare_totalseg.py --ts-dir <path to Totalsegmentator_dataset>
"""

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

# BTCV label id -> TotalSegmentator mask file (without .nii.gz)
BTCV_TO_TOTALSEG = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "inferior_vena_cava",
    10: "portal_vein_and_splenic_vein",
    11: "pancreas",
    12: "adrenal_gland_right",
    13: "adrenal_gland_left",
}
TARGET_AXCODES = ("L", "A", "S")


def find_subject_root(ts_dir: Path) -> Path:
    """The zip unpacks to a nested folder; accept either level."""
    for candidate in (ts_dir, ts_dir / ts_dir.name):
        if (candidate / "meta.csv").is_file():
            return candidate
    raise FileNotFoundError(f"meta.csv not found under {ts_dir}")


def to_las(img: nib.Nifti1Image) -> nib.Nifti1Image:
    transform = ornt_transform(io_orientation(img.affine), axcodes2ornt(TARGET_AXCODES))
    return img.as_reoriented(transform)


def merge_labels(subject_dir: Path):
    """Return (label image, voxel count per BTCV id). Later ids never overwrite earlier ones."""
    merged, reference = None, None
    counts = {}
    for label_id, name in BTCV_TO_TOTALSEG.items():
        mask_img = nib.load(str(subject_dir / "segmentations" / f"{name}.nii.gz"))
        mask = np.asanyarray(mask_img.dataobj) > 0
        if merged is None:
            merged = np.zeros(mask.shape, dtype=np.uint8)
            reference = mask_img
        mask &= merged == 0
        merged[mask] = label_id
        counts[label_id] = int(mask.sum())
    return nib.Nifti1Image(merged, reference.affine), counts


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Prepare TotalSegmentator cases for BTCV-style inference")
    parser.add_argument("--ts-dir", type=Path, required=True, help="Totalsegmentator_dataset folder")
    parser.add_argument("--out-dir", type=Path, default=repo / "data" / "totalseg")
    parser.add_argument("--split", default="test", help="TotalSegmentator split to draw from (train/val/test/all)")
    parser.add_argument("--study-contains", default="abdomen", help="keep study types containing this text")
    parser.add_argument("--max-cases", type=int, default=0, help="stop after this many accepted cases (0 = no limit)")
    parser.add_argument("--min-organs", type=int, default=13, help="minimum number of non-empty BTCV organs")
    args = parser.parse_args()

    root = find_subject_root(args.ts_dir)
    with (root / "meta.csv").open(encoding="utf-8-sig") as f:
        meta = list(csv.DictReader(f, delimiter=";"))
    candidates = [
        r for r in meta
        if args.study_contains in r["study_type"] and args.split in ("all", r["split"])
    ]
    print(f"{len(candidates)} candidate subjects (split={args.split}, study type contains '{args.study_contains}')")

    (args.out_dir / "images").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "labels").mkdir(parents=True, exist_ok=True)
    labels = json.loads((repo / "data" / "dataset_0.json").read_text(encoding="utf-8"))["labels"]

    selection, validation = [], []
    for row in candidates:
        sid = row["image_id"]
        subject_dir = root / sid
        label_img, counts = merge_labels(subject_dir)
        present = sum(c > 0 for c in counts.values())
        accepted = present >= args.min_organs
        ct_img = nib.load(str(subject_dir / "ct.nii.gz"))
        record = dict(
            image_id=sid, accepted=int(accepted), organs_present=present,
            study_type=row["study_type"].strip(), institute=row["institute"], age=row["age"],
            gender=row["gender"], split=row["split"],
            shape="x".join(str(s) for s in ct_img.shape),
            spacing_mm="x".join(f"{float(z):.2f}" for z in ct_img.header.get_zooms()[:3]),
            **{f"vox_{labels[str(k)]}": v for k, v in counts.items()},
        )
        selection.append(record)
        print(f"  {sid}: {present}/13 organs, {row['study_type'].strip()}, {'kept' if accepted else 'skipped'}")
        if not accepted:
            continue

        if not np.allclose(ct_img.affine, label_img.affine, atol=1e-3) or ct_img.shape != label_img.shape:
            raise ValueError(f"{sid}: CT and masks are not on the same grid")
        ct_las, label_las = to_las(ct_img), to_las(label_img)
        ct_out = nib.Nifti1Image(np.asanyarray(ct_las.dataobj).astype(np.int16), ct_las.affine)
        lab_out = nib.Nifti1Image(np.asanyarray(label_las.dataobj).astype(np.uint8), label_las.affine)
        nib.save(ct_out, str(args.out_dir / "images" / f"{sid}.nii.gz"))
        nib.save(lab_out, str(args.out_dir / "labels" / f"{sid}.nii.gz"))
        validation.append({"image": f"images/{sid}.nii.gz", "label": f"labels/{sid}.nii.gz"})
        if args.max_cases and len(validation) >= args.max_cases:
            break

    dataset = {
        "name": "totalsegmentator_ood",
        "description": "TotalSegmentator subjects with labels mapped to the 13 BTCV organs",
        "labels": labels,
        "label_source": {str(k): v for k, v in BTCV_TO_TOTALSEG.items()},
        "orientation": "".join(TARGET_AXCODES),
        "validation": validation,
    }
    (args.out_dir / "dataset_ood.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    with (args.out_dir / "selection.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(selection[0].keys()))
        writer.writeheader()
        writer.writerows(selection)
    print(f"\nKept {len(validation)} of {len(selection)} inspected subjects -> {args.out_dir}")


if __name__ == "__main__":
    main()
