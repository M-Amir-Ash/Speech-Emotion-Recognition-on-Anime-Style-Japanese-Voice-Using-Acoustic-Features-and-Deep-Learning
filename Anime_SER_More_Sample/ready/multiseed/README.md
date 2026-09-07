# Improvement 5: Group-Safe Split Variability Analysis

This isolated robustness experiment evaluates five predefined source-media/episode-group-disjoint partitions. It does not select a best seed and does not replace the canonical seed-42 result.

Both model configurations were frozen before regenerated test evaluation. MFCC correlation filtering was learned from each seed's training split only. Test labels were used only for final scoring.

## Per-seed results

| seed | model | accuracy | macro_precision | macro_recall | macro_f1 | mfcc_retained_features | hyperparameters_frozen_before_test | test_used_for_tuning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | MFCC + SVM | 0.749367 | 0.740742 | 0.756389 | 0.747407 | 88 | True | False |
| 123 | MFCC + SVM | 0.750630 | 0.761869 | 0.766735 | 0.763231 | 86 | True | False |
| 2026 | MFCC + SVM | 0.713555 | 0.728023 | 0.736834 | 0.729066 | 87 | True | False |
| 3407 | MFCC + SVM | 0.757946 | 0.764033 | 0.769526 | 0.764326 | 87 | True | False |
| 7777 | MFCC + SVM | 0.748052 | 0.760816 | 0.749388 | 0.751503 | 88 | True | False |
| 42 | openSMILE eGeMAPS + XGBoost | 0.749367 | 0.770578 | 0.743820 | 0.751744 |  | True | False |
| 123 | openSMILE eGeMAPS + XGBoost | 0.745592 | 0.766024 | 0.743126 | 0.752548 |  | True | False |
| 2026 | openSMILE eGeMAPS + XGBoost | 0.746803 | 0.770570 | 0.759121 | 0.761502 |  | True | False |
| 3407 | openSMILE eGeMAPS + XGBoost | 0.755501 | 0.768299 | 0.755317 | 0.760840 |  | True | False |
| 7777 | openSMILE eGeMAPS + XGBoost | 0.761039 | 0.788217 | 0.763489 | 0.769538 |  | True | False |

## Aggregate results

| model | seeds | mean_accuracy | std_accuracy | mean_macro_precision | std_macro_precision | mean_macro_recall | std_macro_recall | mean_macro_f1 | std_macro_f1 | min_macro_f1 | max_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MFCC + SVM | 5 | 0.743910 | 0.017396 | 0.751097 | 0.015949 | 0.755775 | 0.013315 | 0.751106 | 0.014330 | 0.729066 | 0.764326 |
| openSMILE eGeMAPS + XGBoost | 5 | 0.751660 | 0.006488 | 0.772737 | 0.008856 | 0.752975 | 0.009147 | 0.759234 | 0.007326 | 0.751744 | 0.769538 |

## Most variable model/class combinations

| model | emotion | seeds | mean_f1 | std_f1 | min_f1 | max_f1 | mean_support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| openSMILE eGeMAPS + XGBoost | angry | 5 | 0.706865 | 0.093357 | 0.623377 | 0.848485 | 44.600000 |
| MFCC + SVM | disgust | 5 | 0.791830 | 0.069285 | 0.692308 | 0.857143 | 31.800000 |
| MFCC + SVM | angry | 5 | 0.714216 | 0.054581 | 0.645161 | 0.767677 | 44.600000 |
| MFCC + SVM | happy | 5 | 0.721709 | 0.052046 | 0.649007 | 0.787097 | 80.800000 |
| MFCC + SVM | neutral | 5 | 0.783142 | 0.039277 | 0.718750 | 0.812030 | 69.400000 |

Every seed passed the 1,969-row uniqueness, SHA-256 uniqueness, all-seven-class, and zero source-group-overlap checks. See the CSV artifacts for full split and class-wise evidence.
