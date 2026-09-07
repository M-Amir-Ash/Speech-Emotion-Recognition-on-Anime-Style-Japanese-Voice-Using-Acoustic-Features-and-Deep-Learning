# Anime SER: 1,969-sample experiment

This directory contains the model implementation, frozen experiment manifests,
canonical saved results, separate improvement experiments, analysis utilities,
and organized reporting evidence.

## Start here

- `ready/evidence.xlsx` — thesis evidence tracker.
- `ready/report.ipynb` — executed read-only final report.
- `output/anime_only_final_results.csv` — canonical ten-method result table.
- `output/classwise_results.csv` — per-emotion metrics.
- `anime_ser.py` — training and evaluation implementation.
- `requirements.txt` — Python dependencies.

## Training versus reporting

`anime_ser.py` and `anime_ser.ipynb` implement feature extraction, tuning,
training, and evaluation. Running them can take a long time and can rewrite the
configured output directory.

`anime_ser_final_report.ipynb` and `ready/report.ipynb` only read saved artifacts,
verify consistency, and create tables and figures. They do not train models.

The repository copy of `anime_ser.ipynb` has its stale 993-sample cell outputs
cleared. Its code is preserved, but the authoritative reported evidence is the
frozen 1,969-sample CSV output and final report.

## Local dataset requirement

Audio is excluded from Git. To run training, place the seven emotion directories
and `_cluster_clean_manifest.csv` under `sample/`. The expected frozen counts are
1,313 training, 261 validation, and 395 test clips.

See the repository-level `DATA_AVAILABILITY.md` for exclusions and reproduction
guidance.
