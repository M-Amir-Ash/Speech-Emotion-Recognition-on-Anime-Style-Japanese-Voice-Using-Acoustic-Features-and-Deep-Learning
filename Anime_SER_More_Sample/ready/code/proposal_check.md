# Proposal implementation check

Sources checked:

- Final Project Proposal PDF, 53 PDF pages, modified 22 June 2026.
- `Anime_SER/anime_ser.ipynb` and its generator/export.
- Original `Python Notebook/Test/ser_dual_training_framework_anime_transfer_only.ipynb`.
- Auto Split source scripts, manifests, quality-control reports, cluster reports, and workflow audit under `Training Data/Auto Split`.
- Physical contents and audio headers of `cluster_clean_target_400_multi_feature`.

## Confirmed implementation

- Anime-only seven-emotion population.
- 13 MFCC coefficients with nine summary statistics.
- Training-only MFCC correlation filtering at an explicit 0.95 threshold.
- SVM, k-NN, Decision Tree, Random Forest, and Gaussian Naive Bayes.
- Normalized 128-band log-Mel CNN and CNN-BiLSTM.
- TF-IDF unigram/bigram Logistic Regression with balanced class weights.
- Weighted late fusion using 0.40 CNN audio and 0.60 text probabilities.
- openSMILE eGeMAPS low-level descriptors summarized for XGBoost.
- Accuracy, macro precision, macro recall, macro F1, confusion matrices, and class-wise analysis.
- Source-media-disjoint split targeting the proposal's effective 68%/12%/20% proportions.
- Common preprocessing is limited to 16 kHz, mono conversion, and peak normalization;
  silence trimming, band-pass filtering, and pre-emphasis are guarded by the optional
  experimental-enhancement flag and are off for the primary experiment.

## Verified current data

- 993 selected WAV files: 22 angry, 59 disgust, 223 fear, 90 happy, 103 neutral, 217 pleasant surprise, and 279 sad.
- All 993 files are 16 kHz, mono, PCM-16.
- All 993 files map one-to-one to selected cluster-manifest rows and have transcripts.
- No exact duplicate audio hashes or cross-label duplicate hashes were found.
- 96 upstream source-media groups are available.
- The group split produced 663 train, 123 validation, and 207 test rows with all emotions and zero source overlap.
- 910/993 clips are within the proposal's planned 1–6 second range; 83 are shorter than one second.

## Blocking limitations

- The upstream quality-control summary explicitly reports `ready_for_final_training = no`.
- The workflow audit reports missing speaker-cluster and character-organization outputs.
- Accepted cluster metadata contains 100% `UNKNOWN_SPEAKER` and `GLOBAL_BASIC`, so character leakage cannot be audited.
- The labels are cluster-selected pseudo-labels, not independently adjudicated ground truth.
- Every emotion is below the cluster target of 400; shortage is largest for angry and disgust.

Current model scores are suitable for development diagnostics, not final thesis claims, until the blocking limitations are resolved or explicitly accepted and defended by the supervisor.
