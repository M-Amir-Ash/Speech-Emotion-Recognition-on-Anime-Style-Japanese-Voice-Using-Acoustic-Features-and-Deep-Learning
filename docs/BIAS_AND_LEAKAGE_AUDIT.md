# Independent adversarial audit: bias, leakage, and metric reproducibility

Audit date: 2026-10-29. This review is read-only: no model was retrained, no pipeline was executed, and no historical artifact was modified. Calculations are preserved in `bias_leakage_recompute.py` and `bias_leakage_recompute.json`.

## Executive conclusion

The saved artifacts support the selection-sensitivity and JVNV counts, but they do **not** support an unqualified claim that every paper metric was independently reproduced.

- **Verified fact:** the canonical 1,969 split has no cross-split overlap by source group, SHA-256, clip ID, or file path.
- **Confirmed leakage in cohort construction:** the historical acoustic selector fits a scaler and label-specific centroids on the entire 66,084-row quality-valid population before the eventual train/validation/test split. This is transductive preprocessing and label-conditioned selection. It does not prove that test labels entered the final classifiers, but it makes the selected test cohort non-independent of full-population labels and features.
- **No confirmed classifier-fit leakage:** canonical MFCC feature filtering uses training data only, scikit-learn scaling is inside train-fitted pipelines, and hyperparameters are selected on validation data in the inspected code.
- **Unresolved speaker leakage:** every canonical row has `speaker=UNKNOWN_SPEAKER`; character is null. Source-media separation is verified, but speaker-independent generalization is not.
- **Selection bias is large:** the selected historical dataset retains 1.62%–6.18% of each pseudo-class and changes the maximum/minimum class ratio from 8.14 to 2.68. The sensitivity test retains only 455/13,462 rows (3.38%) with class coverage from 1.28% to 7.78%.
- **JVNV metric scope matters:** the reported 0.5881 macro-F1 is correct only when averaged over the six JVNV reference classes. Including Neutral, which the model predicts 134 times but JVNV never labels, produces macro-F1 **0.5041**.
- **Canonical ten-model recomputation is blocked by missing predictions:** the folder contains result tables and confusion-matrix images, not row-level predictions or numeric confusion matrices for the ten canonical methods. Their reported values are preserved and internally repeated, but cannot all be independently recomputed from saved predictions.

## Evidence and calculation scope

| Evidence | Grain | Independent use |
|---|---|---|
| Folder E `output/{train,validation,test}_manifest.csv` | one selected clip per row | split overlap, class balance, speaker/character availability |
| Folder D `phase3d_leakage_aware/group_first_partition_manifest.csv` | one quality-valid clip per row | group-first split counts and overlap |
| Folder D `stage3_final_test_v001/test_predictions.csv` | model × cohort × clip | direct recomputation of sensitivity metrics |
| Folder F full run `raw_predictions/jvnv_predictions.csv` | one JVNV recording per row | direct recomputation of JVNV metrics |
| Folder A `_multimodal_emotion_manifest.csv` and `_cluster_clean_manifest.csv` | classifier decision / quality-valid clip | label-status composition and historical selection coverage |
| Folder E result CSVs | one method/result per row | conflict/version comparison, not prediction-level recomputation |

The sensitivity prediction file intentionally repeats each of the 455 selector-passing clips in both its full-population and passing-subset cohorts. It contains 27,834 rows: 13,462 + 455 for each of two models. There are 910 duplicate `(model, clip_id)` occurrences across cohort labels. Metrics were recomputed after filtering `evaluation_cohort`; pooling all rows would give incorrect results.

## 1. Recomputed results

### Selection-sensitivity predictions

Directly recomputed from Folder D `test_predictions.csv` with the seven fixed emotion labels:

| Model and cohort | Support | Coverage | Accuracy | Macro-F1 | Weighted-F1 | Verification |
|---|---:|---:|---:|---:|---:|---|
| eGeMAPS + XGBoost, full test | 13,462 | 100% | 0.261997 | 0.247013 | 0.260672 | Matches saved table |
| eGeMAPS + XGBoost, label-conditioned passing test | 455 | 3.3799% | 0.753846 | 0.742107 | 0.752263 | Matches saved table |
| MFCC + SVM, full test | 13,462 | 100% | 0.221661 | 0.211540 | 0.223749 | Matches saved table |
| MFCC + SVM, label-conditioned passing test | 455 | 3.3799% | 0.707692 | 0.691709 | 0.706789 | Matches saved table |

**Verified fact:** the saved metrics are arithmetically correct for their stated cohorts. **Potential bias:** the large selected/full gap is conditional association, not evidence that the selector causally improves general emotion recognition. The subset is chosen using inherited target labels.

### JVNV predictions

Directly recomputed from 1,615 unique `clip_id` rows:

| Metric | Recomputed value | Interpretation |
|---|---:|---|
| Successful predictions | 1,587 | 98.27% of references |
| Full-population raw agreement | 57.6471% | Matches reported 57.65% |
| Accuracy among returned labels | 58.6641% | 1,587-row denominator |
| Six-reference-class macro-F1 | 0.588074 | Matches reported 0.5881 |
| Seven-model-class macro-F1 | 0.504064 | Includes unsupported Neutral class |
| Automatically accepted | 218 | 13.4985% coverage |
| Accepted-only agreement | 86.2385% | Conditional on acceptance |
| Needs review | 1,369 | 84.768% of all references |
| Quality rejected | 28 | 1.734% of all references |

JVNV reference counts are relatively balanced across six classes (249–306), but returned predictions are not: Happy is predicted 567 times, while Disgust is predicted 71 times; Neutral is predicted 134 times despite no Neutral reference class.

**Verified fact:** the paper-aligned JVNV values are reproducible. **Potential bias:** 86.24% is a selective-acceptance accuracy at only 13.50% coverage, not overall performance. **Metric-definition limitation:** excluding Neutral from macro-F1 raises the score from 0.5041 to 0.5881. The six-class value is defensible for a six-class reference taxonomy only if explicitly disclosed.

### Canonical ten-model results

No row-level canonical prediction file was found. `output/anime_only_final_results.csv` records the ten reported values, while confusion matrices are PNG images. Therefore:

- the result CSV can be checked for consistency with notebook output and duplicate copies;
- accuracy can sometimes be visually approximated from figures, but this is not an independent exact recomputation;
- macro-F1, class weighting, and label ordering cannot be fully rederived from PNGs;
- the ten canonical numbers remain **not independently reproducible from saved predictions**.

This is a provenance deficiency, not evidence that the numbers are false.

## 2. Contradictory and alternative results

Multiple valid-looking outputs report materially different values. They use different datasets, seeds, weighting, or experiment versions and must not be merged into one canonical table.

| Method | Main historical CSV | Alternative saved result | Difference/evidence |
|---|---:|---:|---|
| eGeMAPS + XGBoost | 0.751744 | old-993: 0.744604; five-seed mean: 0.759234 | multiseed range 0.751744–0.769538 |
| MFCC + SVM | 0.747407 | old-993: 0.643775; five-seed mean: 0.751106 | multiseed range 0.729066–0.764326 |
| Log-Mel CNN | 0.578654 | ready/main: 0.548719; weighted run: 0.576175; old-993: 0.165819 | protocol/version sensitive |
| CNN-BiLSTM | 0.329617 | ready/main: 0.297446; weighted run: 0.461322; old-993: 0.591562 | direction and magnitude conflict strongly |
| Late fusion | 0.564381 | ready/main: 0.551847; old-993: 0.194080 | materially different histories |
| Character TF-IDF | 0.243254 | char-ngram run: 0.243893; ready/main: 0.052746; old-993: 0.042807 | analyzer/protocol differs |

Sources are Folder E `output/`, `output_old_993_20260831/`, `ready/main/`, and `experiments/{grouped_multiseed,neural_class_weighted,text_char_ngram}/`.

**Verified fact:** these files disagree. **Unresolved limitation:** without immutable run IDs linking the submitted manuscript to code, predictions, and environment, matching the manuscript is not the same as independently proving that it is the authoritative execution.

## 3. Split overlap and identity leakage

### Canonical 1,969 split

| Check | Train–validation | Train–test | Validation–test |
|---|---:|---:|---:|
| Source-group overlap | 0 | 0 | 0 |
| SHA-256 overlap | 0 | 0 | 0 |
| Clip-ID overlap | 0 | 0 | 0 |
| File-path overlap | 0 | 0 | 0 |
| Speaker-value overlap | 1 | 1 | 1 |

Within each split there are zero duplicate hashes and zero duplicate paths. The single overlapping speaker value is `UNKNOWN_SPEAKER`, covering all 1,969 rows. Character is null for all rows.

**Verified fact:** no file/source-group overlap is detected. **Unresolved limitation:** speaker and character separation cannot be evaluated; source group represents media provenance, not a verified human speaker.

### Group-first 66,084 partition

Independent checks on `group_first_partition_manifest.csv` confirm:

- 44,628 train, 7,994 validation, and 13,462 test rows;
- 277 unique source groups/source-media identities;
- zero source-group, SHA-256, clip-ID, sample-ID, resolved-path, and source-media overlap across all split pairs;
- all 66,084 hashes, clip IDs, sample IDs, and resolved paths are unique.

No speaker or character column exists in this manifest. Thus **file/source-media leakage is not found**, while **speaker leakage remains untestable**.

## 4. Label-conditioned selection and preprocessing order

### Confirmed transductive selection

Folder A `cluster_clean_target_selector.py` provides direct code evidence:

1. Lines 701–723 read the entire character-emotion manifest, retain all quality-pass rows, extract each row's inherited `target_emotion`, and build features.
2. Lines 728–733 impute medians and call `StandardScaler().fit_transform(arr)` on the complete usable population.
3. `classify_separability` constructs centroids by `target_emotion`; each row is compared with its own label centroid and other label centroids.
4. Lines 738–740 build a separate clean pool for each emotion before per-class selection.
5. Only afterward does Folder E `anime_ser.py` split the selected 1,969 rows at lines 321–331.

This is **confirmed evaluation-cohort leakage/transductive preprocessing**: eventual validation/test features and inherited labels influence scaling, centroids, thresholds, and membership before splitting. It is also **confirmed label-conditioned selection** because the inherited target label determines the own centroid, competing centroids, per-class thresholds, and quotas.

It should not be described as ordinary random sampling or as a deployable label-free selector.

### Classifier preprocessing after splitting

The inspected canonical training code is more defensible downstream:

- group-disjoint splitting occurs before feature-model fitting;
- MFCC correlation filtering uses `X_train_mfcc_full` only (lines 542–560);
- SVM and k-NN scaling is inside scikit-learn pipelines fitted on `X_train_mfcc` (lines 580–630);
- eGeMAPS/XGBoost candidates fit training data and select hyperparameters using validation predictions (lines 753–797);
- test predictions are generated after the selected model is fitted.

No direct test-set fit was found in these inspected paths. The earlier cohort-construction leakage remains and cannot be cured by clean classifier fitting.

The split generator also tests up to 300 group assignments and selects the one closest to the full dataset's label distribution. This is label-informed split construction, but not performance-based test tuning. It improves stratification while making the exact test composition algorithmically selected rather than arbitrary.

## 5. Class imbalance, coverage, and label status

### Historical population versus selected dataset

| Pseudo-class | Quality-valid rows | Quality-valid share | Selected rows | Selected share | Retention |
|---|---:|---:|---:|---:|---:|
| Angry | 3,993 | 6.04% | 229 | 11.63% | 5.74% |
| Disgust | 2,413 | 3.65% | 149 | 7.57% | 6.18% |
| Fear | 8,983 | 13.59% | 400 | 20.31% | 4.45% |
| Happy | 19,643 | 29.72% | 400 | 20.31% | 2.04% |
| Neutral | 10,641 | 16.10% | 362 | 18.38% | 3.40% |
| Pleasant Surprise | 17,012 | 25.74% | 275 | 13.97% | 1.62% |
| Sad | 3,399 | 5.14% | 154 | 7.82% | 4.53% |

The selector reduces imbalance but does so with strongly class-dependent retention. This changes the evaluation distribution and preferentially keeps acoustically separable pseudo-label examples.

### Label-status composition of the 68,860 manifest

| Review status | Rows | Share |
|---|---:|---:|
| `manual_review` | 65,050 | 94.47% |
| `high_confidence` | 1,034 | 1.50% |
| `rejected` | 2,776 | 4.03% |

`pred_folder=Needs_Review` accounts for 55,867 rows (81.13%). The status field indicates routing, not proof that a human reviewed and confirmed those rows. Only 1.50% are explicitly marked high-confidence in this manifest.

### Selection-sensitivity test coverage

| Class | Test denominator | Passing | Coverage |
|---|---:|---:|---:|
| Angry | 814 | 46 | 5.65% |
| Disgust | 514 | 40 | 7.78% |
| Fear | 1,879 | 111 | 5.91% |
| Happy | 3,989 | 104 | 2.61% |
| Neutral | 2,134 | 65 | 3.05% |
| Pleasant Surprise | 3,434 | 44 | 1.28% |
| Sad | 698 | 45 | 6.45% |

Coverage is explicitly recorded as `LABEL_CONDITIONED_BY_INHERITED_PSEUDO_LABEL`. The selective score is therefore not comparable to an unconditioned test score without the coverage and conditioning caveat.

## 6. Finding classification

### Verified facts

- Row-level sensitivity and JVNV arithmetic matches saved outputs under their stated cohort/label definitions.
- Canonical and group-first partitions have no cross-split source-group/hash/file overlap.
- Historical selection is label-conditioned and fitted on the full quality-valid population before splitting.
- Canonical speaker and character identity are unavailable.
- Result families disagree materially across saved experiment versions.

### Potential bias

- Acoustic separability filtering enriches easy, centroid-consistent pseudo-labels.
- Per-class caps alter natural prevalence and improve apparent balance.
- Accepted-only JVNV performance is subject to selective abstention.
- Pseudo-labels are used both to construct cohorts and to score agreement, so model performance is not independent emotion validity.
- Source-group separation does not guarantee speaker independence.

### Confirmed leakage

- Full-population scaling, imputation, label-centroid calculation, and selection occur before the historical train/validation/test split.
- Eventual validation/test labels and features therefore affect whether samples enter the selected experiment.

No duplicate-file/source-group leakage or direct classifier fitting on the test set was found.

### Unresolved limitations

- Missing canonical row-level predictions prevent independent recomputation of all ten paper metrics.
- Historical per-run model activation and checkpoint revisions remain unknown.
- Speaker leakage cannot be tested from `UNKNOWN_SPEAKER`/missing character identifiers.
- Human-review completion is not evidenced by `manual_review` status.
- Different saved result families are not tied to an immutable manuscript commit/run ID.
- JVNV is a reconstructed workflow with a six-class reference taxonomy, not exact replay of the historical anime labeling run.

## Readiness judgment

**Needs revision before strong reproducibility claims.** The release can accurately publish the methodology and saved evidence if it explicitly labels the historical selected-test metrics as label-conditioned pseudo-label agreement, discloses full-corpus pre-split selection, reports JVNV macro-F1 taxonomy, and states that the canonical ten-model table lacks saved row-level predictions. It should not claim speaker-independent evaluation or complete independent reproduction.
