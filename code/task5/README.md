# Task 5 — Reproducibility Analysis

<!--
  Note from Task 2 (inference) to the Task 5 team — summary for the reproducibility writeup:

  - Dice vs. official report (overlap=0.7): Base 0.8169 vs 0.8225 (-0.56%),
    Small 0.7895 vs 0.7979 (-0.84%), Tiny 0.6662 vs 0.7205 (-5.43%).
    Base/Small reproduce closely; Tiny has a much bigger gap.
  - Env: Windows 11, RTX 5080, Python 3.10, PyTorch 2.11+cu128, latest MONAI
    (official repo has no pinned requirements/Dockerfile).
  - Had to patch official test.py/utils for newer MONAI/PyTorch, since the code
    targets an older version and some details aren't documented:
      * transforms.AddChanneld (deprecated) -> EnsureChannelFirstd
      * torch.load on PyTorch 2.6+ needs weights_only=False (else UnpicklingError)
      * meta access moved from batch["label_meta_dict"]["affine"] to
        batch["label"].meta["affine"][0] (same for image filename)
  - Likely cause of remaining gap: official numbers use 5-fold CV + ensembling,
    we ran a single pretrained checkpoint once. Tiny is extra sensitive because
    only Base uses self-supervised pretraining, so Tiny/Small generalize less
    reliably and are more affected by undocumented training hyperparameters.
  - Takeaway: reproduction is feasible and fairly close for Base/Small, but
    required real effort to resolve version-compatibility issues, and the CV/
    ensembling protocol for the reported numbers isn't documented.
-->