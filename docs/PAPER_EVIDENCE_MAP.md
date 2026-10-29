# Paper evidence map

Local source aliases below are rooted at `<PROJECT_ROOT>`; they identify the authoritative research folders without publishing machine-specific absolute paths.

| Paper claim | Source folder | Exact evidence file | Verification method | Status |
|---|---|---|---|---|
| 68,860 pseudo-labeled recordings | A: `.../Training Data/Auto Split More Sample` | `output/pipeline_workspace/multimodal_emotion_classified/_multimodal_emotion_manifest.csv` | Counted manifest rows; status composition independently profiled | VERIFIED_FROM_MANIFEST |
| 66,084 quality-valid recordings | A / D | A `_cluster_clean_manifest.csv`; D `phase3d_leakage_aware/group_first_partition_manifest.csv` | Counted rows and checked stable IDs | VERIFIED_FROM_MANIFEST |
| 2,304 acoustically selected candidates | A | `.../cluster_clean_target_400_multi_feature/_cluster_clean_manifest.csv` | Counted `clean_separable` rows | VERIFIED_FROM_MANIFEST |
| 1,969 final recordings and 1,313/261/395 split | E: `.../Anime_SER_More_Sample` | `output/{train,validation,test}_manifest.csv` | Counted rows; checked source-group, hash, clip-ID and path overlap | VERIFIED_FROM_MANIFEST |
| XGBoost macro-F1 0.7517; SVM 0.7474 | E | `output/anime_only_final_results.csv` | Compared paper values to saved metric table | VERIFIED_FROM_SAVED_METRICS |
| Canonical ten-model benchmark | E | `output/anime_only_final_results.csv`, `output/confusion_*.png` | Saved metrics/figures only; canonical row-level predictions absent | NOT_REPRODUCIBLE_FROM_ARCHIVE |
| Five-seed XGB mean 0.7592; SVM mean 0.7511 | E | `experiments/grouped_multiseed/aggregate_results.csv` | Compared saved per-seed aggregates | VERIFIED_FROM_SAVED_METRICS |
| CNN-BiLSTM and other neural scores | E | canonical, `output_old_993_20260831`, `ready/main`, `experiments/neural_class_weighted` tables | Results differ by cohort/configuration and cannot be collapsed into one result | CONFLICTING |
| 44,628/7,994/13,462 group-first partition | D: `.../Anime_SER_Experiment_V2` | `phase3d_leakage_aware/group_first_partition_manifest.csv` | Recomputed split counts and overlap checks | VERIFIED_FROM_MANIFEST |
| 1,549 selected train; 455 selected test; 3.3799% coverage | D | `phase3d_b2_results/stage3_final_test_v001/test_predictions.csv`, `test_coverage.csv` | Recomputed from saved predictions | RECOMPUTED_FROM_PREDICTIONS |
| XGB selected/full macro-F1 0.7421/0.2470 | D | same `test_predictions.csv` | Recomputed by evaluation cohort | RECOMPUTED_FROM_PREDICTIONS |
| JVNV 1,615 clips; macro-F1 0.5881 (six reference classes) | F: `JVNV_External_Validation` | `results/runs/20261024_historical_multimodal_full_v001_v002/raw_predictions/jvnv_predictions.csv` | Recomputed from 1,615 saved rows | RECOMPUTED_FROM_PREDICTIONS |
| JVNV macro-F1 0.5041 (seven model-output classes) | F | same predictions | Recomputed with Neutral included as a model-output class | RECOMPUTED_FROM_PREDICTIONS |

## Contradictory results

The canonical table reports CNN-BiLSTM 0.3296 macro-F1. The archived 993-sample run reports 0.5916, `ready/main` reports 0.2974, and the class-weighted neural experiment reports 0.4613. These are distinct cohorts/configurations, not interchangeable replications. The release preserves every table and does not select a value merely because it agrees with the manuscript. Canonical row-level predictions were not archived, so the canonical ten-model scores cannot be independently recomputed.
