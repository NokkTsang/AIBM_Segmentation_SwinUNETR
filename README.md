# Project 1 — Segmentation - Swin UNETR

**Course:** Foundations of AI in Biomedicine — Project 1
**Topic:** Reproducing BTCV multi-organ segmentation results with **Swin UNETR** (Tang et al., CVPR 2022).

This repository reproduces the BTCV abdominal multi-organ segmentation results of
[Swin UNETR](https://openaccess.thecvf.com/content/CVPR2022W/DVMM/html/Tang_Self-Supervised_Pre-Training_of_Swin_Transformers_for_3D_Medical_Image_Analysis_CVPRW_2022_paper.html)
using the official pretrained weights (Base / Small / Tiny) and evaluates them quantitatively and
qualitatively, following the project task sheet (`Tasks 1–5` + bonus OOD task).

## Team members

| Name | Matriculation number |
|------|----------------------|
| Baichuan Zeng | 03832071 |
| _TBD_ | _TBD_ |
| _TBD_ | _TBD_ |
| _TBD_ | _TBD_ |

## Repository structure

```text
AIBM_Segmentation_SwinUNETR/
├── README.md                 # project overview & setup
├── requirements.txt          # Python dependencies (pip install -r requirements.txt)
├── .gitignore
│
├── code/
│   ├── task1/
│   │   └── README.md
│   ├── task2/
│   │   ├── README.md
│   │   ├── test.py           # inference + per-case Dice entry point (MONAI SwinUNETR)
│   │   ├── utils/
│   │   │   ├── data_utils.py # MONAI dataloaders/transforms + distributed sampler
│   │   │   └── utils.py      # dice(), resample_3d(), AverageMeter, distributed helpers
│   │   └── outputs/          # predicted segmentation masks (.nii.gz), per model config
│   │       ├── base/
│   │       ├── small/
│   │       └── tiny/
│   ├── task3/
│   │   └── README.md         # quantitative evaluation (Dice + Hausdorff), reuses task2/outputs
│   ├── task4/
│   │   └── README.md         # qualitative evaluation (visual inspection), reuses task2/outputs
│   ├── task5/
│   │   └── README.md
│   └── bonus/
│       └── README.md
│
├── data/                     # BTCV dataset
│   ├── README.md
│   ├── dataset_0.json        # MONAI decathlon-style train/val/test split (tracked)
│   ├── imagesTr/              # training CT volumes
│   ├── imagesTs/              # test CT volumes
│   └── labelsTr/              # training organ labels
│
└── weights/                  # pretrained Swin UNETR checkpoints
    ├── README.md
    ├── swinunetr_base.pt
    ├── swinunetr_small.pt
    └── swinunetr_tiny.pt
```

## Setup

1. **Clone the repository**
2. **Create an environment and install dependencies**
3. **Get the data and weights** (both are git-ignored):
   - Follow [data/README.md](data/README.md) to download/place the BTCV dataset under `data/`.
   - Follow [weights/README.md](weights/README.md) to download the pretrained checkpoints under `weights/`.
4. **Run inference** (Task 2) — see [code/task2/README.md](code/task2/README.md) for full commands.

## Task 2 results

Overall mean Dice on the 6 BTCV validation cases (`infer_overlap=0.7`):

| Model | Overall mean Dice |
|-------|:---:|
| Swin UNETR / Base | 0.8169 |
| Swin UNETR / Small | 0.7895 |
| Swin UNETR / Tiny | 0.6662 |

Per-case numbers and exact commands: [code/task2/README.md](code/task2/README.md). Per-organ
Dice/Hausdorff distance is Task 3's job (not computed yet).

## Dataset & model source

- **Paper:** Tang et al., *Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis*, CVPR 2022. [[paper]](https://openaccess.thecvf.com/content/CVPR2022W/DVMM/html/Tang_Self-Supervised_Pre-Training_of_Swin_Transformers_for_3D_Medical_Image_Analysis_CVPRW_2022_paper.html)
- **Official code:** [Project-MONAI/research-contributions/SwinUNETR](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV)
- **MONAI BTCV tutorial:** [swin_unetr_btcv_segmentation_3d.ipynb](https://colab.research.google.com/github/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb)
- **BTCV dataset:** [Synapse:syn3193805](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752)
- **TotalSegmentator (bonus OOD task):** [TotalSegmentator dataset](https://zenodo.org/records/10047292)