# weights/

Pretrained **Swin UNETR** checkpoints, fine-tuned on BTCV by the paper authors. **Not committed**
to git (total ~335 MB, and `swinunetr_base.pt` alone exceeds GitHub's 100 MB per-file limit) —
download them yourself and place them here.

## Structure

```text
weights/
├── swinunetr_base.pt   # feature_size=48, 62.1M params  (~245 MB)
├── swinunetr_small.pt  # feature_size=24, 15.7M params  (~67 MB)
└── swinunetr_tiny.pt   # feature_size=12,  4.0M params  (~17 MB)
```

## Download

Official checkpoints from the
[Project-MONAI/research-contributions SwinUNETR/BTCV](https://github.com/Project-MONAI/research-contributions/blob/main/SwinUNETR/BTCV/README.md#models)
model table (reported Dice @ overlap=0.7 / 0.5):

| Model | Feature size | Dice (ov=0.7) | Dice (ov=0.5) | Download | Save as |
|-------|:---:|:---:|:---:|----------|---------|
| Swin UNETR / Base | 48 | 82.25 | 81.86 | [swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt) | `swinunetr_base.pt` |
| Swin UNETR / Small | 24 | 79.79 | 79.34 | [swin_unetr.small_5000ep_f24_lr2e-4_pretrained.pt](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.small_5000ep_f24_lr2e-4_pretrained.pt) | `swinunetr_small.pt` |
| Swin UNETR / Tiny | 12 | 72.05 | 70.35 | [swin_unetr.tiny_5000ep_f12_lr2e-4_pretrained.pt](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.tiny_5000ep_f12_lr2e-4_pretrained.pt) | `swinunetr_tiny.pt` |

Rename each downloaded file as shown above (or adjust `--pretrained_model_name` when running
`code/task2/test.py` to match whatever filename you used).

Each checkpoint is a `dict` with a `"state_dict"` key holding the `monai.networks.nets.SwinUNETR`
weights — load with `--feature_size` matching the table above (this must match the checkpoint,
otherwise `load_state_dict` will fail with a shape mismatch).