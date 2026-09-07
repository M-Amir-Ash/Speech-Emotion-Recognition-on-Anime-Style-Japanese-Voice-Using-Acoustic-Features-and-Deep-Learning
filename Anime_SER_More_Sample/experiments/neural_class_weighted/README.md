# Improvement 4: Class-Weighted Neural Experiment

This isolated robustness experiment changes only the neural cross-entropy loss by applying inverse-frequency class weights computed from the 1,313 training samples. The frozen dataset, split, preprocessing, architectures, unweighted outputs, text baseline, speaker analysis, pseudo-labeling, and fusion were not changed.

## Selection and test protocol

The three existing documented candidates per architecture were compared using validation Macro-F1 only. The test manifest was not loaded during selection. After configurations were frozen, each final weighted model was evaluated once on the unchanged 395-sample test set.

## Selected configurations

- Log-Mel CNN + weighted cross-entropy: `{"dropout": 0.4, "final_training_best_epoch": 22, "final_training_validation_macro_f1": 0.631853852738866, "learning_rate": 0.0005, "tuning_best_epoch": 10, "validation_macro_f1": 0.535186805551683}`
- Log-Mel CNN-BiLSTM + weighted cross-entropy: `{"dropout": 0.3, "final_training_best_epoch": 16, "final_training_validation_macro_f1": 0.492914797635089, "hidden_size": 64, "learning_rate": 0.001, "tuning_best_epoch": 5, "validation_macro_f1": 0.35656603560558114}`

## Final results

| model_family | method | validation_macro_f1 | test_accuracy | test_macro_precision | test_macro_recall | test_macro_f1 | test_size | test_evaluation_number |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn | Log-Mel CNN + weighted cross-entropy | 0.631854 | 0.582278 | 0.578285 | 0.621520 | 0.576175 | 395 | 1 |
| cnn_bilstm | Log-Mel CNN-BiLSTM + weighted cross-entropy | 0.492915 | 0.481013 | 0.464197 | 0.481764 | 0.461322 | 395 | 1 |

## Comparison with frozen unweighted models

| model_family | unweighted_method | weighted_method | unweighted_selection_validation_macro_f1 | weighted_selection_validation_macro_f1 | weighted_final_training_validation_macro_f1 | unweighted_test_accuracy | weighted_test_accuracy | accuracy_difference | unweighted_test_macro_f1 | weighted_test_macro_f1 | macro_f1_difference | unweighted_zero_f1_classes | previously_zero_classes_improved | weighted_is_better_by_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn | Log-mel CNN | Log-Mel CNN + weighted cross-entropy | 0.488761 | 0.535187 | 0.631854 | 0.572152 | 0.582278 | 0.010127 | 0.548719 | 0.576175 | 0.027456 |  |  | True |
| cnn_bilstm | Log-mel CNN-BiLSTM | Log-Mel CNN-BiLSTM + weighted cross-entropy | 0.328620 | 0.356566 | 0.492915 | 0.445570 | 0.481013 | 0.035443 | 0.297446 | 0.461322 | 0.163876 | angry;disgust;pleasant_surprise | angry;disgust;pleasant_surprise | True |

See `classwise_results.csv` and the two confusion-matrix CSV files for class-level behavior.
