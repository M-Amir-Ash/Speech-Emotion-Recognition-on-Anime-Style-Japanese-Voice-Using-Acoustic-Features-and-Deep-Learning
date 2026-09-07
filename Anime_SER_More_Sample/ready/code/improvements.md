# Improvement Checklist

This checklist applies to the completed 1,969-sample anime-only experiment archived with this file.

## Highest priority

1. Independently audit the pseudo-labels.
   - Randomly sample 10-20 clips from each of the seven emotions (70-140 clips total).
   - Have a Japanese-capable human reviewer listen to the audio; do not validate from transcript text alone.
   - Record the original label, reviewer label, agreement, and uncertain cases.
   - Report overall agreement, per-class agreement, and Cohen's kappa when possible.
   - Current risk: 1,613 of 1,969 selected clips originated from `Needs_Review`, and none has a recorded GPT-judge result.

2. Make the Japanese text baseline linguistically appropriate.
   - The current word-level TF-IDF baseline is weak for Japanese text without spaces.
   - Add a character n-gram TF-IDF baseline (for example, `analyzer="char"`, 2-5 grams), or tokenize with Sudachi/MeCab.
   - Select text settings using training and validation only; do not tune against the final test set.
   - Describe the present TF-IDF result (Macro-F1 0.0527) as a limitation of the tested lexical baseline, not proof that Japanese text lacks emotion information.

## Important methodological improvements

3. Add speaker-cluster metadata without reorganizing the emotion folders.
   - All 1,969 selected clips currently use `UNKNOWN_SPEAKER` and `GLOBAL_BASIC`.
   - Use speaker embeddings to assign stable anonymous speaker IDs.
   - Keep source/episode-disjoint terminology until speaker-disjoint splitting is verified; do not claim speaker-independent SER.

4. Make imbalance handling comparable across model families.
   - Classical models use class weighting, while the neural models currently use unweighted cross-entropy.
   - Evaluate class-weighted cross-entropy or a training-only weighted sampler for CNN and CNN-BiLSTM.
   - Keep validation and test distributions natural and unchanged.

5. Measure split variability.
   - The current experiment uses one grouped split with random seed 42.
   - Repeat with several group-safe seeds or grouped cross-validation and report mean, standard deviation, and per-class F1.
   - Preserve a final untouched test policy when selecting configurations.

6. Resolve the duration-policy discrepancy.
   - 1,821 of 1,969 clips (92.5%) are within the planned 1-6 second range.
   - The remaining 148 clips are 0.65-0.99 seconds; none exceeds six seconds.
   - Either exclude them before splitting or document and justify the revised minimum duration.

## Reporting and scope

7. Report class shortages rather than filling them with overlapping clips.
   - Angry 229, Disgust 149, Fear 400, Happy 400, Neutral 362, Pleasant Surprise 275, Sad 154.
   - Use Macro-F1 and class-wise metrics; an equal 400 samples per class is not required.

8. Describe the evaluated domain precisely.
   - The results represent a quality-controlled, cluster-selected anime-style Japanese speech dataset.
   - They do not establish performance on arbitrary raw anime audio.

9. Use conservative comparative language.
   - Current best result: openSMILE eGeMAPS + XGBoost, Macro-F1 0.7517.
   - State that handcrafted acoustic methods outperformed the tested neural configurations under the evaluated conditions.
   - Do not claim that classical ML is inherently superior to deep learning.

10. Preserve the present test set.
    - Do not use the 395-sample final test split for further tuning.
    - If labels or inclusion rules change materially, version the dataset and create a newly documented experiment rather than silently replacing results.

## Current evidence status

- Notebook execution: completed.
- Dataset used: `Anime_SER_More_Sample/sample`.
- Samples: 1,969.
- Split: 1,313 train / 261 validation / 395 test (approximately 68/12/20).
- Transcript coverage: 100%.
- Upstream `ready_for_final_training`: no.
- Appropriate interpretation: preliminary experimental evidence pending independent label validation.
