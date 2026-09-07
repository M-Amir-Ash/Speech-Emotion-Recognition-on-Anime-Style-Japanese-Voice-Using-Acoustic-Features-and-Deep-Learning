# Duration-Policy Analysis

This directory contains Improvement 6, a read-only analysis of all 1,969 frozen
clips. No clips, labels, splits, preprocessing, or model artifacts were changed.

## Overall statistics

| count | minimum_seconds | maximum_seconds | mean_seconds | median_seconds | standard_deviation_seconds | standard_deviation_definition | percentile_25_seconds | percentile_75_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1969 | 0.650000 | 5.340000 | 2.022136 | 1.920000 | 0.795605 | population_ddof_0 | 1.425000 | 2.435000 |

## Duration-policy categories

| duration_policy_category | sample_count | percentage_of_dataset |
| --- | --- | --- |
| below_1.00_second | 148 | 7.516506 |
| within_1.00_to_6.00_seconds | 1821 | 92.483494 |
| above_6.00_seconds | 0 | 0.000000 |

## Per-emotion statistics

| emotion | sample_count | minimum_seconds | maximum_seconds | mean_seconds | median_seconds | below_1_second_count | below_1_second_within_class_percentage | share_of_all_below_1_second_clips_percentage | short_clip_rate_ratio_vs_dataset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| angry | 229 | 0.835000 | 4.725000 | 2.384432 | 2.325000 | 3 | 1.310044 | 2.027027 | 0.174289 |
| disgust | 149 | 0.670000 | 3.125000 | 1.449362 | 1.425000 | 33 | 22.147651 | 22.297297 | 2.946535 |
| fear | 400 | 0.710000 | 5.340000 | 2.289538 | 2.195000 | 18 | 4.500000 | 12.162162 | 0.598682 |
| happy | 400 | 0.790000 | 4.870000 | 2.297913 | 2.147500 | 7 | 1.750000 | 4.729730 | 0.232821 |
| neutral | 362 | 0.680000 | 3.420000 | 1.683978 | 1.615000 | 34 | 9.392265 | 22.972973 | 1.249552 |
| pleasant_surprise | 275 | 0.695000 | 3.875000 | 2.064673 | 2.020000 | 13 | 4.727273 | 8.783784 | 0.628919 |
| sad | 154 | 0.650000 | 2.620000 | 1.345649 | 1.300000 | 40 | 25.974026 | 27.027027 | 3.455598 |

## Conclusion

Short clips are concentrated most strongly in Sad (40/154; 25.974%) and Disgust
(33/149; 22.148%). Sad, Neutral, and Disgust together contribute 107/148
(72.297%) of all sub-one-second clips.

The discrepancy is a **documented implementation refinement**, not a bug. The
proposal allowed its approximate planned range to be adjusted based on segment
quality, and the pipeline intentionally implemented a 0.65-second minimum.

Recommended wording:

> Most selected utterances (92.5%) were within the planned 1–6 s duration interval. A small number of shorter utterances were retained when they satisfied the remaining Japanese-dialogue, audio-quality, and clean-selection criteria.

See `proposal_duration_comparison.md` for the source-level evidence and exact
proposal comparison. `duration_distribution.csv` preserves the row-level audit.
