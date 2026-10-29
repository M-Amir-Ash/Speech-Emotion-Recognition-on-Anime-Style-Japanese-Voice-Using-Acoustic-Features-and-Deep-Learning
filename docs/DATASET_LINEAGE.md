# Dataset lineage

## Canonical historical path

| Stage | Status | Script/evidence | Input | Output/count |
|---|---|---|---|---|
| Anime media to separated/segmented dialogue | **CONFIRMED implementation; UNVERIFIED exact historical invocation** | Folder A `extract_segment_multimodal_pipeline.py`, `segment_audio.py`, adjacent workflow runner | Anime media under Folder A input/workspace | Multiple WAV stage trees under `output/pipeline_workspace` |
| Automatic emotion pseudo-labeling | **CONFIRMED artifacts** | `multimodal_emotion_classifier.py`; `_multimodal_emotion_manifest.csv` | segmented Japanese dialogue | 68,860 rows |
| Quality screening | **CONFIRMED by saved fields/counts** | `labeling_quality_control.py`; `_cluster_clean_manifest.csv` | historical manifest/derived clips | 66,084 rows with `audio_quality_status=pass`; 2,776 rejected from the 68,860 population |
| Acoustic separability candidate identification | **CONFIRMED** | `cluster_clean_target_selector.py`; `_cluster_clean_manifest.csv` | 66,084 quality-valid rows | 2,304 `clean_separable` candidates |
| Final seven-class selection | **CONFIRMED** | same manifest plus Folder E `sample/` | 2,304 candidates | 1,969 `selection_reason=selected` clips |
| Frozen splits | **CONFIRMED** | Folder E `output/{train,validation,test}_manifest.csv` | 1,969 selected clips | 1,313 train; 261 validation; 395 test |
| Canonical ten-model experiment | **CONFIRMED saved outputs** | Folder E `anime_ser.ipynb`, `anime_ser.py`, `output/anime_only_final_results.csv` | frozen split manifests/features | ten paper-result rows |
| Paper reporting | **CONFIRMED values; manuscript source linkage outside this audit** | saved result CSV/notebook and evidence package | canonical experiment | ICAIDES table values |

The three split manifests have zero cross-split SHA-256 overlap and zero cross-split `source_group` overlap. Their `file_path` fields contain private absolute Windows paths and therefore require sanitized release copies.

## Counts and label origin

- 68,860 is the classifier manifest population, not a count of unique files across Folder A.
- 66,084 equals the seven-label population after removal of the 2,776 rejected rows.
- The 66,084 labels remain inherited pseudo-labels; quality validity is not human emotion validity.
- The selected distribution is Angry 229, Disgust 149, Fear 400, Happy 400, Neutral 362, Pleasant Surprise 275, Sad 154.
- The manifest records confidence, gaps, transcript, decision reasons, review states, and quality features. Manual-review status does not prove that a human changed or confirmed every row.

## Leakage-aware sensitivity branch

Folder D creates a later group-first branch from the 66,084 quality-valid pool:

`66,084 -> 44,628 train / 7,994 validation / 13,462 test -> selector trained only from training -> 1,549 selected training / 455 passing test`.

The frozen Stage 3 output explicitly says the test subset is label-conditioned and evaluates agreement with inherited pseudo-labels. This branch is a robustness/sensitivity analysis, not a replacement for the historical 1,969 experiment.

## External validation branch

Folder F separately maps 1,615 JVNV reference recordings through a reconstructed historical multimodal classifier. It does not run acoustic clean clustering. JVNV Surprise is prospectively mapped to `Pleasant_Surprise`; JVNV has no Neutral reference class.

## Unverified connections

- **UNVERIFIED**: exact original shell/menu invocation and exact cached model revision used for the 68,860 historical run.
- **UNVERIFIED**: whether all `manual_review` rows received human review; status is a workflow routing field.
- **UNVERIFIED**: redistribution rights for anime-derived clips and JVNV audio. Metadata can be separated from audio, but even filenames/transcripts should receive privacy/copyright review.
