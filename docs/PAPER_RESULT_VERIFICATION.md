# Paper result verification

## Canonical ten-model table

Source: Folder E `output/anime_only_final_results.csv` (10 rows). All target values match the saved CSV to rounding.

| Model | Saved macro-F1 | Verification |
|---|---:|---|
| eGeMAPS + XGBoost | 0.751744 | **CONFIRMED** 0.7517 |
| MFCC + SVM | 0.747407 | **CONFIRMED** 0.7474 |
| MFCC + Random Forest | 0.723397 | **CONFIRMED** 0.7234 |
| MFCC + Gaussian NB | 0.696168 | **CONFIRMED** 0.6962 |
| MFCC + k-NN | 0.650124 | **CONFIRMED** 0.6501 |
| Log-Mel CNN | 0.578654 | **CONFIRMED** 0.5787 |
| Late fusion CNN + text | 0.564381 | **CONFIRMED** 0.5644 |
| MFCC + Decision Tree | 0.482182 | **CONFIRMED** 0.4822 |
| Log-Mel CNN-BiLSTM | 0.329617 | **CONFIRMED** 0.3296 |
| Character TF-IDF + Logistic Regression | 0.243254 | **CONFIRMED** 0.2433 |

The supporting notebook has saved numerical outputs and figures. Saved histories, hyperparameter tables, classwise results, confusion matrices, and selected hyperparameters are present under E `output/` and mirrored into the evidence package.

## Split and dataset values

- 1,969 total, 1,313 train, 261 validation, 395 test: **CONFIRMED** from manifests.
- Seven-class distribution 229/149/400/400/362/275/154: **CONFIRMED** from `dataset_summary.csv`.
- Zero cross-split source-group/SHA overlap: **CONFIRMED**.

## Selection-sensitivity values

Source: Folder D `phase3d_b2_results/stage3_final_test_v001`.

- 66,084 quality-valid; 44,628/7,994/13,462 group-first split: **CONFIRMED by frozen manifests/reports**.
- 1,549 selected training and 455 passing test: **CONFIRMED**.
- Coverage 455/13,462 = 3.3799%: **CONFIRMED**.
- XGBoost selected-test macro-F1 0.742107 and full-pool macro-F1 0.247013: **CONFIRMED**.
- SVM selected-test macro-F1 0.691709 and full-pool macro-F1 0.211540: **CONFIRMED**.

These are pseudo-label agreement results on a label-conditioned selected subset and a full-pool stress test, not direct replacements for the historical paper table.

## Conflicts that must remain visible

The evidence tree contains older, newer, multiseed, neural-only, text-only, and reconstructed result packages. They differ because protocols, splits, seeds, weighting, or correction status differ. The manuscript-aligned file above is authoritative for the historical selected-dataset table, but the existence of conflicting artifacts must be disclosed. Do not rewrite or delete them.

The historical pseudo-label model activation is still **UNVERIFIED**: configuration and later reconstruction establish plausible implementation, not the exact model branch used for the original 68,860 classifications.
