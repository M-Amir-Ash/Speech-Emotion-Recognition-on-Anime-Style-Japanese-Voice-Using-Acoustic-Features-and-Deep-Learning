# JVNV external validation

This folder publishes the validation source, configuration, tests, and the authoritative sanitized 1,615-record result package from `20261024_historical_multimodal_full_v001_v002`.

## Verified result

- Reference recordings: 1,615
- Successful predictions: 1,587
- Six-reference-class macro-F1: 0.588074
- Seven-model-output-class macro-F1: 0.504064
- Auto-accepted predictions: 218 (13.50% coverage)
- Auto-accepted accuracy: 86.24%

The two macro-F1 values use different averaging label sets and are both retained explicitly. See `results/20261024_historical_multimodal_full_v001_v002/`.

## Excluded artifacts

JVNV audio, the original dataset archive, transcripts, absolute audio paths, model caches, checkpoints, runtime caches, and local virtual environments are intentionally excluded. Obtain JVNV through its authorized source and configure local paths in `jvnv_validation_config.json`.
