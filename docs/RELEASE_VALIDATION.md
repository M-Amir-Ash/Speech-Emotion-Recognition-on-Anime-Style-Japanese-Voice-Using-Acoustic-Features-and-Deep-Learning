# Release validation

Status: **PASS**

Validated 74 release files totaling 76,231,795 bytes. No training, labeling, feature extraction, or model inference was run.

- Manifest rows: 68,860 initial; 66,084 quality-valid; 1,313/261/395 canonical split.
- Canonical source-group, SHA-256, and clip-ID cross-split overlaps: zero.
- Selection-sensitivity XGBoost macro-F1: 0.742107 selected test; 0.247013 full population.
- JVNV macro-F1: 0.588074 over six reference classes; 0.504064 over seven model-output classes.
- Static Python parsing passed; no prohibited audio/model-weight extension or file >=100 MB was found.
- Credential and absolute-path patterns were scanned.

No validation errors were found.
