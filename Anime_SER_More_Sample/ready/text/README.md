# Japanese character n-gram text baseline

This experiment preserves the frozen 1,969-sample dataset, its 1,313/261/395 split, the original word-level TF-IDF result, and the existing 0.40/0.60 fusion result.

The original word-level TF-IDF representation was poorly suited to unsegmented Japanese transcripts. A character n-gram baseline was therefore evaluated separately.

## Method

- `TfidfVectorizer(analyzer="char")`
- Validation search over n-gram ranges `(2,4)`, `(2,5)`, and `(3,5)`
- Validation search over `min_df` values 1, 2, and 3
- Validation search over Logistic Regression `C` values 0.1, 0.5, 1.0, 2.0, and 5.0
- Balanced Logistic Regression class weights
- Seed 42
- Primary selection metric: validation Macro-F1
- Final model fitting population: frozen 1,313-row training split only, matching the original text baseline
- Final test population: frozen 395-row test split, evaluated once after selection

The runner intentionally separates validation selection from final testing:

```powershell
python experiments\text_char_ngram\run_text_char_ngram.py tune
python experiments\text_char_ngram\run_text_char_ngram.py evaluate
```

The evaluation command refuses to run again after `final_results.csv` exists. Create a new versioned experiment directory instead of replacing completed results.

This experiment does not change the text component of the existing weighted fusion model.
