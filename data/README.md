# data/

The **BTCV (Beyond the Cranial Vault)** multi-organ abdominal CT dataset. **Raw CT volumes and
labels are git-ignored** (not committed) because they total ~1.5 GB — download them yourself as
described below.

## Structure

```text
data/
├── dataset_0.json   # MONAI decathlon-style split (tracked in git — small, no raw data)
├── imagesTr/        # 30 training CT volumes, imgNNNN.nii.gz (git-ignored)
├── imagesTs/        # 20 held-out test CT volumes, imgNNNN.nii.gz (git-ignored)
└── labelsTr/        # 30 training organ labels, labelNNNN.nii.gz (git-ignored)
```

- `imagesTr`/`labelsTr` pairs are split by `dataset_0.json` into:
  - **training** (24 cases): `img0001`–`img0010`, `img0021`–`img0034`
  - **validation** (6 cases): `img0035`–`img0040` — this is the split used for inference/evaluation
    in Task 2–4.
- `imagesTs` (20 cases, `img0061`–`img0080`) has no public labels (official BTCV test/leaderboard
  split) and is not used by the current scripts.
- Labels: 13 foreground organs + background (see `dataset_0.json` → `"labels"`):
  `spleen, rkid, lkid, gall, eso, liver, sto, aorta, IVC, veins, pancreas, rad, lad`.

## Download

1. Register for a free account on [Synapse](https://www.synapse.org/) and request access to the
   BTCV challenge project: [Synapse:syn3193805](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752).
2. Download `Abdomen/RawData.zip` (training + test CT volumes and training labels) from the
   project's **Files** section.
3. Unzip and arrange the files to match the structure above (rename as needed so that image/label
   filenames match `dataset_0.json`, e.g. `img0001.nii.gz` / `label0001.nii.gz`).
4. `dataset_0.json` is already provided in this folder (same split file used by the official
   [Swin UNETR BTCV](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV)
   code) — no need to re-download it.