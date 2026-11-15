"""Task 3: offline evaluation of Swin UNETR BTCV prediction masks.

Run from repository root, after BTCV labels are downloaded:
    python code/task3/evaluate.py

Dependencies: numpy, scipy, nibabel (already required by the repo).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Same empty-reference convention as code/task2/utils/utils.py: 0 if GT empty."""
    target_sum = np.count_nonzero(target)
    if target_sum == 0:
        return 0.0
    pred_sum = np.count_nonzero(pred)
    intersection = np.count_nonzero(pred & target)
    return 2.0 * intersection / (pred_sum + target_sum)


def cropped_masks(pred: np.ndarray, target: np.ndarray):
    """Crop to union bounding box + a 1-voxel margin for correct surfaces."""
    combined = pred | target
    slices = []
    for ax in range(combined.ndim):
        projection = np.any(combined, axis=tuple(i for i in range(combined.ndim) if i != ax))
        indices = np.flatnonzero(projection)
        lo = max(int(indices[0]) - 1, 0)
        hi = min(int(indices[-1]) + 2, combined.shape[ax])
        slices.append(slice(lo, hi))
    return pred[tuple(slices)], target[tuple(slices)]


def hausdorff_mm(pred: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]):
    """Symmetric max Hausdorff distance and HD95, based on 3D surface voxels, in mm.

    A missing predicted or ground-truth organ has undefined Hausdorff distance
    (returns NaN). The CSV includes an explicit status column for these cases.
    """
    if not np.any(pred) or not np.any(target):
        return float('nan'), float('nan')

    pred, target = cropped_masks(pred, target)
    structure = ndi.generate_binary_structure(pred.ndim, 1)
    pred_surface = pred & ~ndi.binary_erosion(pred, structure=structure, border_value=0)
    gt_surface = target & ~ndi.binary_erosion(target, structure=structure, border_value=0)

    distances_to_gt = ndi.distance_transform_edt(~gt_surface, sampling=spacing)
    pred_to_gt = distances_to_gt[pred_surface]
    del distances_to_gt
    distances_to_pred = ndi.distance_transform_edt(~pred_surface, sampling=spacing)
    gt_to_pred = distances_to_pred[gt_surface]
    del distances_to_pred

    hd = max(float(np.max(pred_to_gt)), float(np.max(gt_to_pred)))
    hd95 = max(float(np.percentile(pred_to_gt, 95)), float(np.percentile(gt_to_pred, 95)))
    return hd, hd95


def mean_finite(values):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float('nan')


def write_csv(path: Path, rows: list[dict], headers: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def verify_alignment(pred_img, gt_img, name: str):
    if pred_img.shape != gt_img.shape:
        raise ValueError(f'{name}: prediction shape {pred_img.shape} != ground truth {gt_img.shape}. '
                         'Do not calculate metrics until images are aligned.')
    if not np.allclose(pred_img.affine, gt_img.affine, atol=1e-3, rtol=1e-4):
        raise ValueError(f'{name}: prediction and GT have different NIfTI affines. '
                         'Do not calculate metrics until orientation and voxel space are aligned.')


def main():
    import nibabel as nib

    parser = argparse.ArgumentParser(description='Quantitatively evaluate saved Swin UNETR BTCV predictions')
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output_dir = args.output_dir or (repo / 'code' / 'task3' / 'outputs')
    dataset_path = repo / 'data' / 'dataset_0.json'
    with dataset_path.open(encoding='utf-8') as f:
        dataset = json.load(f)

    labels = {int(k): v for k, v in dataset['labels'].items() if int(k) != 0}
    cases = dataset['validation']
    all_rows = []
    per_case_rows = []
    per_organ_rows = []
    model_rows = []

    for model in ('base', 'small', 'tiny'):
        print(f'\nEvaluating {model} ...')
        model_rows_raw = []
        for case in cases:
            image_name = Path(case['image']).name
            case_id = image_name.replace('.nii.gz', '')
            pred_path = repo / 'code' / 'task2' / 'outputs' / model / image_name
            gt_path = repo / 'data' / case['label']
            for path in (pred_path, gt_path):
                if not path.is_file():
                    raise FileNotFoundError(f'Missing required file: {path}\n'
                                            'Predictions are in the repo, but BTCV GT labels '
                                            'must be downloaded into data/labelsTr/.')

            pred_img = nib.load(str(pred_path))
            gt_img = nib.load(str(gt_path))
            verify_alignment(pred_img, gt_img, f'{model}/{case_id}')
            spacing = tuple(float(v) for v in nib.affines.voxel_sizes(gt_img.affine)[:3])
            prediction = np.asanyarray(pred_img.dataobj)
            ground_truth = np.asanyarray(gt_img.dataobj)
            case_rows = []
            for label_id, organ in labels.items():
                p = (prediction == label_id)
                g = (ground_truth == label_id)
                p_count = int(np.count_nonzero(p))
                g_count = int(np.count_nonzero(g))
                status = ('both_empty' if p_count == 0 and g_count == 0 else
                          'missing_prediction' if p_count == 0 else
                          'missing_reference' if g_count == 0 else 'ok')
                dice = dice_score(p, g)
                hd, hd95 = hausdorff_mm(p, g, spacing)
                row = dict(model=model, case=case_id, organ_id=label_id, organ=organ,
                           dice=dice, hausdorff_mm=hd, hd95_mm=hd95, status=status,
                           pred_voxels=p_count, gt_voxels=g_count)
                case_rows.append(row)
                model_rows_raw.append(row)
            all_rows.extend(case_rows)
            row = dict(model=model, case=case_id,
                       mean_dice=float(np.mean([r['dice'] for r in case_rows])),
                       mean_hausdorff_mm=mean_finite([r['hausdorff_mm'] for r in case_rows]),
                       mean_hd95_mm=mean_finite([r['hd95_mm'] for r in case_rows]),
                       valid_hd_organs=sum(r['status'] == 'ok' for r in case_rows),
                       undefined_hd_organs=sum(r['status'] != 'ok' for r in case_rows))
            per_case_rows.append(row)
            print(f"  {case_id}: mean Dice={row['mean_dice']:.4f}, "
                  f"mean HD={row['mean_hausdorff_mm']:.2f} mm, "
                  f"undefined HD organs={row['undefined_hd_organs']}")

        for label_id, organ in labels.items():
            organ_rows = [r for r in model_rows_raw if r['organ_id'] == label_id]
            per_organ_rows.append(dict(model=model, organ_id=label_id, organ=organ,
                                       mean_dice=float(np.mean([r['dice'] for r in organ_rows])),
                                       mean_hausdorff_mm=mean_finite([r['hausdorff_mm'] for r in organ_rows]),
                                       mean_hd95_mm=mean_finite([r['hd95_mm'] for r in organ_rows]),
                                       valid_hd_cases=sum(r['status'] == 'ok' for r in organ_rows),
                                       undefined_hd_cases=sum(r['status'] != 'ok' for r in organ_rows)))
        model_case_rows = [r for r in per_case_rows if r['model'] == model]
        model_rows.append(dict(model=model,
                               mean_dice=float(np.mean([r['mean_dice'] for r in model_case_rows])),
                               mean_hausdorff_mm=mean_finite([r['hausdorff_mm'] for r in model_rows_raw]),
                               mean_hd95_mm=mean_finite([r['hd95_mm'] for r in model_rows_raw]),
                               valid_hd_organ_cases=sum(r['status'] == 'ok' for r in model_rows_raw),
                               undefined_hd_organ_cases=sum(r['status'] != 'ok' for r in model_rows_raw)))

    write_csv(output_dir / 'per_case_per_organ.csv', all_rows,
              ['model', 'case', 'organ_id', 'organ', 'dice', 'hausdorff_mm', 'hd95_mm',
               'status', 'pred_voxels', 'gt_voxels'])
    write_csv(output_dir / 'per_case_summary.csv', per_case_rows,
              ['model', 'case', 'mean_dice', 'mean_hausdorff_mm', 'mean_hd95_mm',
               'valid_hd_organs', 'undefined_hd_organs'])
    write_csv(output_dir / 'per_organ_summary.csv', per_organ_rows,
              ['model', 'organ_id', 'organ', 'mean_dice', 'mean_hausdorff_mm',
               'mean_hd95_mm', 'valid_hd_cases', 'undefined_hd_cases'])
    write_csv(output_dir / 'per_model_summary.csv', model_rows,
              ['model', 'mean_dice', 'mean_hausdorff_mm', 'mean_hd95_mm',
               'valid_hd_organ_cases', 'undefined_hd_organ_cases'])

    ranked = sorted(per_case_rows, key=lambda r: r['mean_dice'])
    summary = {'highest_dice_case': ranked[-1], 'lowest_dice_case': ranked[0],
               'note': ('Case selection is by mean Dice across all 13 organs, '
                        'not by Hausdorff distance. Report both metrics. '
                        'Undefined Hausdorff values are excluded from means and explicitly counted.')}
    (output_dir / 'suggested_examples.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print('\nOverall model summary:')
    for row in model_rows:
        print(f"  {row['model']:5s} Dice={row['mean_dice']:.4f}, "
              f"HD={row['mean_hausdorff_mm']:.2f} mm, HD95={row['mean_hd95_mm']:.2f} mm")
    print(f'\nSaved CSVs and good/bad case suggestions to: {output_dir}')


if __name__ == '__main__':
    main()