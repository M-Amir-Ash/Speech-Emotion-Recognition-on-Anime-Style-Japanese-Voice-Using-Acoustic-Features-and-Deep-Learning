# Ready Results

Start with `evidence.xlsx` to locate proof for each thesis statement, then open
`report.ipynb` for the executed report of the current 1,969-sample experiment.

Quick navigation:

- `evidence.xlsx`: formatted Excel evidence tracker with clickable primary files
- `evidence.csv`: plain CSV copy of the evidence tracker
- `EVIDENCE_GUIDE.md`: proof chain and recommended thesis usage
- `report.ipynb`: consolidated tables, graphs, and interpretation

## Current dataset

- Train: 1,313 clips
- Validation: 261 clips
- Test: 395 clips
- Total: 1,969 clips
- Audio samples remain in `../sample` and are not duplicated here.

## Folder guide

| Folder | Contents |
|---|---|
| `main` | Final model results, class-wise metrics, tuning tables, model weights, and training histories |
| `splits` | Frozen train, validation, and test manifests plus dataset and methodology records |
| `figures` | Current result plots, confusion matrices, MFCC analysis, waveforms, and spectra |
| `text` | Japanese character n-gram TF-IDF experiment |
| `neural` | Class-weighted CNN and CNN-BiLSTM experiment |
| `multiseed` | Group-safe split variability experiment |
| `speaker` | Anonymous speaker-cluster analysis outputs; embeddings and downloaded model cache are excluded |
| `duration` | Duration-policy analysis outputs |
| `audit` | Human pseudo-label audit form and analysis utility |
| `code` | Python implementation, notebook builder, requirements, and documentation |

## Most important files

- `report.ipynb`: current report and tables
- `main/anime_only_final_results.csv`: canonical model comparison
- `main/classwise_results.csv`: canonical per-emotion metrics
- `splits/train_manifest.csv`: frozen training membership
- `splits/validation_manifest.csv`: frozen validation membership
- `splits/test_manifest.csv`: frozen test membership
- `text/final_results.csv`: Japanese character n-gram result
- `neural/final_results.csv`: class-weighted neural results
- `multiseed/aggregate_results.csv`: multi-seed aggregate results
- `speaker/speaker_split_overlap.csv`: estimated speaker-overlap summary
- `duration/duration_summary.csv`: duration statistics
- `audit/audit_review_form.csv`: blank human-review worksheet

## Important warning

The original `../anime_ser.ipynb` is intentionally not copied here because its
saved display output still contains the old 993-sample execution. Its source code
points to the newer sample directory, but the cached cells are stale. Use
`report.ipynb` and the files in this directory when reading or reporting current
results.

This directory contains copies for easier access. The original frozen artifacts
were not moved, renamed, overwritten, or deleted.
