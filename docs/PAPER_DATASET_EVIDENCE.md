# Paper dataset evidence and publication status

Status as of 2026-10-30: **metadata and checksums published; audio restricted**.

The two verified audio packages exist locally, but no written permission authorizing redistribution of the third-party anime-derived recordings was found. Consequently, this repository does not contain WAV or ZIP payloads and does not present broken or placeholder download links.

## Dataset lineage

```text
68,860 pseudo-labeled recordings
    ↓ quality control
66,084 quality-valid recordings
    ↓ label-conditioned acoustic selection
2,304 clean-separable candidates
    ↓ final eligibility and per-class selection
1,969 experimental recordings
    ↓ source-group-disjoint canonical split
1,313 train / 261 validation / 395 test
    ↓
ten-model archived benchmark
```

| Stage | Records | Evidence | Verification status |
|---|---:|---|---|
| Initial pseudo-labeled collection | 68,860 | [`datasets/full_68860/pseudolabel_manifest_sanitized.csv`](../datasets/full_68860/pseudolabel_manifest_sanitized.csv) | Manifest row count verified |
| Quality-valid pool | 66,084 | [`datasets/quality_valid_66084/group_first_partition_manifest_sanitized.csv`](../datasets/quality_valid_66084/group_first_partition_manifest_sanitized.csv) | Manifest row count and stable IDs verified |
| Clean-separable candidates | 2,304 | Historical selector output summarized below | `separability_status=clean_separable` counted locally |
| Final experimental cohort | 1,969 | [`datasets/selected_1969/`](../datasets/selected_1969/) | Manifest rows, clip IDs, WAV hashes, and local files verified |
| Canonical benchmark | 10 methods | [`results/canonical/anime_only_final_results.csv`](../results/canonical/anime_only_final_results.csv) | Saved metrics only; canonical predictions are absent |

The final cohort contains 229 Angry, 149 Disgust, 400 Fear, 400 Happy, 362 Neutral, 275 Pleasant Surprise, and 154 Sad recordings. Its split contains 1,313 training, 261 validation, and 395 testing records.

## Initial pseudo-label and quality-control composition

| Inherited pseudo-label | Initial rows |
|---|---:|
| Angry | 3,993 |
| Disgust | 2,413 |
| Fear | 8,983 |
| Happy | 19,643 |
| Neutral | 10,641 |
| Pleasant Surprise | 17,012 |
| Sad | 3,399 |
| No retained label / rejected | 2,776 |

Historical review statuses are 65,050 `manual_review`, 1,034 `high_confidence`, and 2,776 `rejected`. These statuses describe an automatic quality-control workflow; they are not human validation of emotion.

The acoustic selector classified 2,304 rows as `clean_separable`, 3,570 as `usable_review`, and 60,210 as `overlap_review`. A later per-class selection retained 1,969 rows. The selector used label-conditioned centroids and standardized the quality-valid pool before the final experimental split, so the canonical cohort carries a confirmed cohort-construction leakage risk. See [`BIAS_AND_LEAKAGE_AUDIT.md`](BIAS_AND_LEAKAGE_AUDIT.md).

## Exact-byte identification without audio redistribution

[`datasets/full_68860/audio_checksums_sanitized.csv`](../datasets/full_68860/audio_checksums_sanitized.csv) provides 68,860 path-free records containing a stable release record ID, WAV SHA-256, byte size, verified archive part, inherited pseudo-label and quality status, hashed source-group key, duration, and historical confidence fields.

The selected split manifests publish SHA-256, stable clip ID, assigned class, split, source-group metadata, and quality fields. Local verification found 1,969 unique selected hashes, no missing WAVs, no manifest/hash mismatch, and no cross-split duplicate audio hashes.

| Component | Files | Audio bytes | Publication status |
|---|---:|---:|---|
| Selected experimental WAVs | 1,969 | 127,497,356 | Restricted locally |
| Complete pseudo-labeled WAV collection | 68,860 | 4,637,188,592 | Restricted locally |
| Five lossless stored-mode ZIP parts | 5 | 4,666,045,782 | Restricted locally |

All five local archives passed member-count, CRC, member-size, member-path, and SHA-256 read-back verification. Their checksums are retained in the private research audit. No archive has been uploaded.

## Splitting, labels, and known limitations

- The canonical split is disjoint by source group, clip ID, and WAV SHA-256.
- Speaker independence is **not verified**: all selected rows use `UNKNOWN_SPEAKER`, and character identifiers are unavailable.
- Emotion labels are inherited pseudo-labels, not independently human-verified ground truth.
- Label-conditioned selection makes the selected cohort easier than the full quality-valid population.
- Canonical row-level predictions were not preserved. XGBoost macro-F1 0.7517 and SVM macro-F1 0.7474 are verified from archived tables, not independently recomputed.
- Conflicting CNN-BiLSTM results are preserved rather than reconciled by selecting a preferred value. See [`PAPER_EVIDENCE_MAP.md`](PAPER_EVIDENCE_MAP.md).

## Code and experiment links

- Canonical training: [`src/training/anime_ser.py`](../src/training/anime_ser.py)
- Output-bearing historical notebook: [`notebooks/canonical/anime_ser.ipynb`](../notebooks/canonical/anime_ser.ipynb)
- Canonical configuration: [`configs/canonical_experiment_config.json`](../configs/canonical_experiment_config.json)
- Pseudo-labeling and selection: [`src/labeling/`](../src/labeling/)
- Selection-sensitivity results: [`results/selection_sensitivity/`](../results/selection_sensitivity/)
- JVNV sanitized predictions: [`results/external_validation/`](../results/external_validation/)

The selection-sensitivity experiment recomputes XGBoost macro-F1 as 0.7421 on the 455 selected-test records and 0.2470 on the complete 13,462-record test population. JVNV macro-F1 is 0.5881 over the six reference emotions and 0.5041 when all seven model-output classes are included.

## Rights and access

The WAVs were extracted from third-party anime media. Academic use, a private repository, or manuscript publication does not by itself authorize redistribution. No written license covering all source recordings and GitHub/dataset-host distribution was located.

Audio publication is blocked pending written authorization covering the underlying audiovisual soundtrack, extracted dialogue, applicable performer or neighbouring rights, derivative clips, hosting, territory, duration, sublicensing, and required attribution. Until then, authorized review must use institution-approved controlled access to the local packages.

## Actual publication status

| Dataset component | Exists locally | Verified | Uploaded | Download URL | Reason if excluded |
|---|---|---|---|---|---|
| 1,969 selected WAVs | Yes | Yes | **No** | None | Redistribution permission not established |
| 68,860 WAVs / five ZIPs | Yes | Yes | **No** | None | Redistribution permission not established |
| Dataset manifests | Yes | Yes | Yes | [`datasets/`](../datasets/) | Sanitized research metadata |
| SHA-256 checksums | Yes | Yes | Yes | [`audio_checksums_sanitized.csv`](../datasets/full_68860/audio_checksums_sanitized.csv) | Paths and filenames removed |
| Training and labeling code | Yes | Static checks passed | Yes | [`src/`](../src/) | Source code is release-safe |
| Experimental results | Yes | Scope documented | Yes | [`results/`](../results/) | Canonical prediction-level limitation retained |

Actual published audio: **0 WAV files, 0 ZIP files, 0 audio bytes**. No dataset release was created because that would incorrectly imply authorized audio availability.
