# Speech Emotion Recognition on Anime-Style Japanese Voice

Research implementation for seven-class speech emotion recognition on
anime-style Japanese voice using acoustic features and deep learning.

## Repository structure

- `Training Data/Auto Split More Sample/` — dataset-construction, segmentation,
  multimodal pseudo-labeling, quality-control, and clean-selection code.
- `Anime_SER_More_Sample/` — model training code, frozen experiment manifests,
  saved results, robustness analyses, and thesis-ready reporting artifacts.
- `DATA_AVAILABILITY.md` — explains which large or restricted artifacts are not
  stored in Git.

## Frozen experiment

- Total selected clips: 1,969
- Train: 1,313
- Validation: 261
- Test: 395
- Emotions: Angry, Disgust, Fear, Happy, Neutral, Pleasant Surprise, and Sad
- Split terminology: source-media/episode group-disjoint; not verified
  speaker-independent

The strongest canonical method in the saved experiment is openSMILE eGeMAPS with
XGBoost (test accuracy 0.7494; test Macro-F1 0.7517).

## Start here

Open `Anime_SER_More_Sample/ready/evidence.xlsx` to map thesis claims to their
supporting artifacts. The consolidated read-only report is
`Anime_SER_More_Sample/ready/report.ipynb`.

The training implementation is `Anime_SER_More_Sample/anime_ser.py`. The
repository copy of `anime_ser.ipynb` contains source code without the stale
993-sample cached output.

## Security and repository scope

Credential files, API tokens, raw anime media, generated audio workspaces,
downloaded model caches, and duplicate feature caches are intentionally excluded.
Never commit a real token or secret to this repository.


## Verified reproducibility release

Start with `docs/PAPER_EVIDENCE_MAP.md`, then read `docs/BIAS_AND_LEAKAGE_AUDIT.md` and `datasets/DATASET_CARD.md`. The organized `src/`, `configs/`, `notebooks/`, `datasets/`, `results/`, and `figures/` directories contain sanitized copies of preserved historical evidence. The canonical table is not described as prediction-level reproduced because its original row-level predictions are absent.
