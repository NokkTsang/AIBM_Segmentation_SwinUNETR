"""Task 4: qualitative evaluation of the Swin UNETR BTCV predictions.

Run from the repository root (needs data/imagesTr, data/labelsTr and code/task2/outputs):

    python code/task4/visualize.py                       # all six validation cases
    python code/task4/visualize.py --cases img0038 img0039
    python code/task4/visualize.py --cases img0038 --slices 40 52 61
    python code/task4/visualize.py --cases img0038 --zoom gall pancreas

Figures written to code/task4/outputs/:

    compare_<case>.png      CT | ground truth | Base | Small | Tiny on the same axial slices
    errors_<case>.png       per-model error maps (missed / extra / wrong organ) on the
                            slices with the most errors
    projection_<case>.png   coronal projection of the whole label volume, which shows
                            far-away false-positive islands (the cause of large Hausdorff)
    zoom_<case>_<organ>.png close-up of one organ, prediction filled + ground-truth contour

Dependencies: numpy, matplotlib, nibabel.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
IMAGES = REPO / "data" / "imagesTr"
LABELS = REPO / "data" / "labelsTr"
PREDS = REPO / "code" / "task2" / "outputs"
METRICS = REPO / "code" / "task3" / "outputs" / "per_case_per_organ.csv"
OUT = Path(__file__).resolve().parent / "outputs"

MODELS = ["base", "small", "tiny"]
CASES = ["img0035", "img0036", "img0037", "img0038", "img0039", "img0040"]

# Label ids 1..13, same order as data/dataset_0.json.
ORGANS = ["spleen", "rkid", "lkid", "gall", "eso", "liver", "sto", "aorta", "IVC", "veins", "pancreas", "rad", "lad"]
ORGAN_NAMES = [
    "Spleen", "R kidney", "L kidney", "Gallbladder", "Esophagus", "Liver", "Stomach",
    "Aorta", "IVC", "Portal/splenic veins", "Pancreas", "R adrenal", "L adrenal",
]
ORGAN_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#ffe119", "#f58231", "#bf7c4a", "#f032e6",
    "#ff4d4d", "#42d4f4", "#ffffff", "#bfef45", "#fabed4", "#469990",
]  # veins are white on purpose: a purple would look like stomach magenta drawn over dark air
ERROR_NAMES = ["Missed (false negative)", "Extra (false positive)", "Wrong organ"]
ERROR_COLORS = ["#2f7bff", "#ff3b30", "#ffd60a"]

# Same soft-tissue window the model sees (ScaleIntensityRanged a_min/a_max in task2).
HU_MIN, HU_MAX = -175.0, 250.0


def make_lut(colors, alpha):
    lut = np.zeros((len(colors) + 1, 4), dtype=np.float32)
    for i, color in enumerate(colors, start=1):
        lut[i] = to_rgba(color, alpha)
    return lut


ORGAN_LUT = make_lut(ORGAN_COLORS, 0.55)
ERROR_LUT = make_lut(ERROR_COLORS, 0.75)


# ---------------------------------------------------------------- loading

def load_las(path, is_label=False):
    """Load a NIfTI volume as (x->left, y->anterior, z->superior) plus voxel spacing in mm.

    With this axis order, `vol[:, :, z].T` drawn with origin='lower' is the usual
    radiological axial view (anterior at the top, patient's left on the right).
    """
    img = nib.as_closest_canonical(nib.load(str(path)))  # RAS
    data = np.asanyarray(img.dataobj)[::-1]  # flip x: RAS -> LAS
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    if is_label:
        return np.rint(data).astype(np.uint8), spacing
    return data.astype(np.float32), spacing


def load_case(case):
    number = case.replace("img", "")
    ct, spacing = load_las(IMAGES / f"{case}.nii.gz")
    gt, _ = load_las(LABELS / f"label{number}.nii.gz", is_label=True)
    preds = {}
    for model in MODELS:
        path = PREDS / model / f"{case}.nii.gz"
        if path.exists():
            preds[model], _ = load_las(path, is_label=True)
            if preds[model].shape != gt.shape:
                raise ValueError(f"{path} has shape {preds[model].shape}, label has {gt.shape}")
    if not preds:
        raise FileNotFoundError(f"no predictions for {case} under {PREDS}")
    return ct, gt, preds, spacing


def load_metrics():
    """{(model, case): {"hd95": mean HD95, "organ": {organ: dice}}} from the Task 3 CSV, if present."""
    table = {}
    if not METRICS.exists():
        return table
    with open(METRICS, newline="") as handle:
        for row in csv.DictReader(handle):
            entry = table.setdefault((row["model"], row["case"]), {"hd95_values": [], "organ": {}})
            entry["organ"][row["organ"]] = float(row["dice"])
            hd95 = float(row["hd95_mm"]) if row["hd95_mm"] not in ("", "nan") else float("nan")
            if np.isfinite(hd95):
                entry["hd95_values"].append(hd95)
    for entry in table.values():
        values = entry.pop("hd95_values")
        entry["hd95"] = float(np.mean(values)) if values else float("nan")
    return table


def dice(pred, target):
    """Same convention as Task 2/3: Dice is 0 when the reference organ is absent."""
    target_sum = np.count_nonzero(target)
    if target_sum == 0:
        return 0.0
    return 2.0 * np.count_nonzero(pred & target) / (np.count_nonzero(pred) + target_sum)


def mean_dice(pred, gt):
    return float(np.mean([dice(pred == k, gt == k) for k in range(1, 14)]))


def model_title(model, case, pred, gt, metrics):
    title = f"{model.capitalize()}\nDice {mean_dice(pred, gt):.3f}"
    entry = metrics.get((model, case))
    if entry and np.isfinite(entry["hd95"]):
        title += f" · HD95 {entry['hd95']:.1f} mm"
    return title


# ---------------------------------------------------------------- drawing helpers

def window(ct2d):
    return np.clip((ct2d - HU_MIN) / (HU_MAX - HU_MIN), 0.0, 1.0)


def show(ax, ct2d, overlay=None, aspect=1.0):
    ax.imshow(window(ct2d).T, cmap="gray", vmin=0, vmax=1, origin="lower", aspect=aspect, interpolation="bilinear")
    if overlay is not None:
        ax.imshow(np.transpose(overlay, (1, 0, 2)), origin="lower", aspect=aspect, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def error_codes(pred, gt):
    """0 = agree, 1 = missed organ voxel, 2 = extra organ voxel, 3 = wrong organ label."""
    codes = np.zeros(gt.shape, dtype=np.uint8)
    codes[(gt > 0) & (pred == 0)] = 1
    codes[(gt == 0) & (pred > 0)] = 2
    codes[(gt > 0) & (pred > 0) & (gt != pred)] = 3
    return codes


def organ_legend(fig, present):
    handles = [Patch(color=ORGAN_COLORS[k - 1], label=ORGAN_NAMES[k - 1]) for k in range(1, 14) if k in present]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=9)


def spread_slices(gt, count):
    """Axial slices at evenly spaced quantiles of the ground-truth organ volume."""
    per_slice = np.count_nonzero(gt, axis=(0, 1)).astype(np.float64)
    if per_slice.sum() == 0:
        return list(np.linspace(0, gt.shape[2] - 1, count + 2).astype(int)[1:-1])
    cumulative = np.cumsum(per_slice) / per_slice.sum()
    quantiles = (np.arange(count) + 0.5) / count
    return sorted({int(np.searchsorted(cumulative, q)) for q in quantiles})


def top_slices(score, count, min_gap):
    """Indices of the highest-scoring slices, at least `min_gap` slices apart."""
    chosen = []
    for z in np.argsort(score)[::-1]:
        if score[z] <= 0:
            break
        if all(abs(int(z) - c) >= min_gap for c in chosen):
            chosen.append(int(z))
        if len(chosen) == count:
            break
    return sorted(chosen)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {path.relative_to(REPO)}")


# ---------------------------------------------------------------- figures

def figure_compare(case, ct, gt, preds, spacing, metrics, slices):
    """CT | ground truth | each model, one row per axial slice."""
    columns = [("CT", None), ("Ground truth", gt)] + [
        (model_title(m, case, p, gt, metrics), p) for m, p in preds.items()
    ]
    aspect = spacing[1] / spacing[0]
    fig, axes = plt.subplots(len(slices), len(columns), figsize=(3.1 * len(columns), 3.1 * len(slices) + 1.0), squeeze=False)
    present = set()
    for row, z in enumerate(slices):
        for col, (title, labels) in enumerate(columns):
            ax = axes[row, col]
            overlay = None
            if labels is not None:
                overlay = ORGAN_LUT[labels[:, :, z]]
                present.update(np.unique(labels[:, :, z]).tolist())
            show(ax, ct[:, :, z], overlay, aspect)
            if row == 0:
                ax.set_title(title, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"slice z={z}", fontsize=10)
    fig.suptitle(f"{case} — axial slices (patient's left is on the right)", fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    organ_legend(fig, present)
    save(fig, f"compare_{case}.png")


def figure_errors(case, ct, gt, preds, spacing, metrics, count):
    """Per-model error maps on the axial slices where the models disagree most with the label."""
    codes = {m: error_codes(p, gt) for m, p in preds.items()}
    score = sum(np.count_nonzero(c, axis=(0, 1)) for c in codes.values())
    slices = top_slices(score, count, min_gap=max(2, gt.shape[2] // 15))
    if not slices:
        return
    aspect = spacing[1] / spacing[0]
    ncols = len(codes) + 1
    fig, axes = plt.subplots(len(slices), ncols, figsize=(3.1 * ncols, 3.1 * len(slices) + 0.9), squeeze=False)
    for row, z in enumerate(slices):
        show(axes[row, 0], ct[:, :, z], ORGAN_LUT[gt[:, :, z]], aspect)
        axes[row, 0].set_ylabel(f"slice z={z}", fontsize=10)
        if row == 0:
            axes[row, 0].set_title("Ground truth", fontsize=11)
        for col, (model, code) in enumerate(codes.items(), start=1):
            show(axes[row, col], ct[:, :, z], ERROR_LUT[code[:, :, z]], aspect)
            if row == 0:
                axes[row, col].set_title(model_title(model, case, preds[model], gt, metrics), fontsize=11)
    fig.suptitle(f"{case} — error maps on the slices with the most errors", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    handles = [Patch(color=c, label=n) for c, n in zip(ERROR_COLORS, ERROR_NAMES)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10)
    save(fig, f"errors_{case}.png")


def project_front(labels):
    """Coronal projection: for every (x, z) ray, the first organ label seen from the front."""
    flipped = labels[:, ::-1, :]  # anterior first
    first = np.argmax(flipped > 0, axis=1)
    return np.take_along_axis(flipped, first[:, None, :], axis=1)[:, 0, :]


def figure_projection(case, ct, gt, preds, spacing, metrics):
    """Whole-volume coronal projection: isolated blobs far from the organs explain large Hausdorff."""
    background = window(ct).mean(axis=1)
    background = (background - background.min()) / max(float(np.ptp(background)), 1e-6)
    columns = [("Ground truth", gt)] + [(model_title(m, case, p, gt, metrics), p) for m, p in preds.items()]
    aspect = spacing[2] / spacing[0]
    height = 3.2 * (gt.shape[2] * spacing[2]) / (gt.shape[0] * spacing[0])
    fig, axes = plt.subplots(1, len(columns), figsize=(3.2 * len(columns), height + 1.6), squeeze=False)
    present = set()
    for ax, (title, labels) in zip(axes[0], columns):
        projected = project_front(labels)
        present.update(np.unique(projected).tolist())
        ax.imshow(background.T, cmap="gray", origin="lower", aspect=aspect, interpolation="bilinear")
        ax.imshow(np.transpose(ORGAN_LUT[projected], (1, 0, 2)), origin="lower", aspect=aspect, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{case} — coronal projection of all labels (front view)", fontsize=13)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    organ_legend(fig, present)
    save(fig, f"projection_{case}.png")


def figure_zoom(case, ct, gt, preds, spacing, metrics, organ, margin_mm=30.0):
    """Close-up of one organ: prediction filled, ground-truth outline in white."""
    k = ORGANS.index(organ) + 1
    reference = gt == k
    if not reference.any():
        reference = np.any([p == k for p in preds.values()], axis=0)
        if not reference.any():
            print(f"skip zoom {case}/{organ}: organ absent in label and predictions")
            return
    z = int(np.argmax(np.count_nonzero(reference, axis=(0, 1))))
    xs, ys = np.nonzero(reference[:, :, z])
    mx, my = int(round(margin_mm / spacing[0])), int(round(margin_mm / spacing[1]))
    x0, x1 = max(xs.min() - mx, 0), min(xs.max() + mx + 1, gt.shape[0])
    y0, y1 = max(ys.min() - my, 0), min(ys.max() + my + 1, gt.shape[1])
    crop = (slice(x0, x1), slice(y0, y1), z)
    lut = make_lut(ORGAN_COLORS, 0.0)
    lut[k] = to_rgba(ORGAN_COLORS[k - 1], 0.6)
    aspect = spacing[1] / spacing[0]
    gt2d = (gt[crop] == k).astype(float)

    fig, axes = plt.subplots(1, len(preds) + 1, figsize=(3.1 * (len(preds) + 1), 3.9), squeeze=False)
    show(axes[0, 0], ct[crop], lut[np.where(gt[crop] == k, k, 0)], aspect)
    axes[0, 0].set_title("Ground truth", fontsize=11)
    for ax, (model, pred) in zip(axes[0, 1:], preds.items()):
        show(ax, ct[crop], lut[np.where(pred[crop] == k, k, 0)], aspect)
        if gt2d.any():
            ax.contour(gt2d.T, levels=[0.5], colors="white", linewidths=1.2)
        ax.set_title(f"{model.capitalize()}\n{ORGAN_NAMES[k - 1]} Dice {dice(pred == k, gt == k):.3f}", fontsize=11)
    fig.suptitle(f"{case} — {ORGAN_NAMES[k - 1]}, slice z={z} (white outline = ground truth)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, f"zoom_{case}_{organ}.png")


# ---------------------------------------------------------------- entry point

def main():
    parser = argparse.ArgumentParser(description="Qualitative figures for the Swin UNETR BTCV predictions")
    parser.add_argument("--cases", nargs="+", default=CASES, help="case ids, e.g. img0038 img0039")
    parser.add_argument("--slices", nargs="+", type=int, help="axial slice indices for compare_<case>.png")
    parser.add_argument("--rows", type=int, default=3, help="number of slices per figure when chosen automatically")
    parser.add_argument("--zoom", nargs="+", default=[], choices=ORGANS, metavar="ORGAN",
                        help=f"organs to draw close-ups for; one of: {' '.join(ORGANS)}")
    args = parser.parse_args()

    metrics = load_metrics()
    for case in args.cases:
        ct, gt, preds, spacing = load_case(case)
        slices = args.slices or spread_slices(gt, args.rows)
        bad = [z for z in slices if not 0 <= z < gt.shape[2]]
        if bad:
            parser.error(f"{case} has slices 0..{gt.shape[2] - 1}, got {bad}")
        figure_compare(case, ct, gt, preds, spacing, metrics, slices)
        figure_errors(case, ct, gt, preds, spacing, metrics, args.rows)
        figure_projection(case, ct, gt, preds, spacing, metrics)
        for organ in args.zoom:
            figure_zoom(case, ct, gt, preds, spacing, metrics, organ)


if __name__ == "__main__":
    main()
