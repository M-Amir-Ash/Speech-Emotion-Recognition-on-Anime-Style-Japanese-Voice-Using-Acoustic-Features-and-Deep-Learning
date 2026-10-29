# Experiment comparison

| Experiment | Identity | Dataset/protocol | Features/models | Outputs/status |
|---|---|---|---|---|
| Folder C `_Anime_SER` | **DUPLICATE alias** | Same physical files as E | Same as E | Do not publish separately. |
| Folder D Experiment V2 | **Independent later branch** | 66,084 quality-valid pool; group-first 44,628/7,994/13,462; selected-training 1,549; passing-test 455 | eGeMAPS-175 + XGBoost; MFCC-117 + linear SVM in final sensitivity stage; manifest-driven scripts | Canonical for selection-sensitivity only, not canonical paper ten-model results. |
| Folder B outer More Sample | **Report derivative** | No dataset; report notebook only | Reads/presents existing results | Exploratory/presentation copy; not executable source of record. |
| Folder E inner More Sample | **Canonical historical experiment** | 1,969 selected seven-class clips; 1,313/261/395 source-group-disjoint splits; random state 42 | MFCC classical models, openSMILE eGeMAPS XGBoost, Log-Mel CNN/CNN-BiLSTM, character TF-IDF, late fusion | Canonical saved ten-model values in `output/anime_only_final_results.csv`. |

## Notebook evidence

- E `anime_ser.ipynb`: 51 cells, 25 code cells, 24 code cells with saved outputs, 44 embedded images; kernel metadata `anaconda-base`, Python 3.13.5.
- E backup notebook: byte-identical to the main notebook.
- E `anime_ser_final_report.ipynb`: 16/16 code cells have outputs and five embedded images.
- B report notebook: 16/16 code cells have outputs and five embedded images, but its hash differs from E's report notebook.
- D `anime_ser_v2.ipynb`: 13 code cells and no saved outputs; it cannot substantiate completed results by itself.

## Canonical historical protocol

- Labels originate from Folder A pseudo-labeling followed by quality/acoustic selection, not independent human ground truth.
- Seven classes are Angry, Disgust, Fear, Happy, Neutral, Pleasant Surprise, Sad.
- `output/experiment_config.json` records random state 42, 16 kHz audio, four-second duration, macro-F1 selection, training/validation-only tuning, and final held-out test policy.
- Split manifests show no source-group overlap. This supports source-media separation, but `speaker=UNKNOWN_SPEAKER` is common and therefore does not prove speaker-independent evaluation.
- Feature dimensions visible in later documented names are MFCC-117 and eGeMAPS-175. Neural inputs are cached Log-Mel arrays. Publish exact feature code/config rather than relying only on dimensions.

## Status and disagreements

- **CANONICAL**: E `output/anime_only_final_results.csv` exactly matches the ten values supplied as verification targets.
- **CORRECTED/CANONICAL FOR SENSITIVITY**: D `phase3d_b2_results/stage3_final_test_v001` is the frozen leakage-aware sensitivity result.
- **PRELIMINARY/EXPLORATORY**: V2 threshold sweeps, pilots, reconstruction experiments, smoke runs, and outer report copies.
- **CONFLICTING**: multiple historical/newer result CSVs exist under E and the evidence archive. They must remain version-labelled; no file should be silently selected because it better matches the manuscript.
