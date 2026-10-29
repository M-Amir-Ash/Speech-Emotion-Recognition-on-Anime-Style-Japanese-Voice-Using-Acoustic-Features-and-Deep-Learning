# Reproducibility

This package supports static verification and lightweight metric recomputation. It does not bundle audio, checkpoints, model caches, or feature caches.

Create a Python environment with `requirements.txt`. Review configuration paths before running. The historical labeling scripts may create or relocate large workspaces; use them only against legally obtained media in a disposable workspace. The audit calculation is reproducible with `python docs/bias_leakage_recompute.py` after adapting its local input aliases.

The canonical ten-model table is archive-verifiable but not prediction-recomputable because its row-level predictions are missing. Selection-sensitivity and JVNV metrics are recomputable from released sanitized prediction rows. No experiment was rerun to build this release.
