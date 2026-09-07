# Evidence Guide

Use `evidence.xlsx` as the main tracker. Each row connects a thesis statement to
the exact file that supports it. Paths are relative to this `ready` directory so
the folder can be moved without creating very long spreadsheet links.

## Proof chain

```text
Frozen sample population
  -> splits/train_manifest.csv
  -> splits/validation_manifest.csv
  -> splits/test_manifest.csv

Training and evaluation implementation
  -> code/anime_ser.py
  -> main/experiment_config.json
  -> main/selected_hyperparameters.csv

Canonical test results
  -> main/anime_only_final_results.csv
  -> main/classwise_results.csv
  -> figures/confusion_*.png

Separate robustness analyses
  -> text/
  -> neural/
  -> multiseed/
  -> speaker/
  -> duration/

Consolidated thesis presentation
  -> report.ipynb
```

## Recommended thesis use

1. Use `splits/dataset_summary.csv` and the three split manifests for the dataset
   and experimental-setup chapter.
2. Use `main/anime_only_final_results.csv` as the canonical model-comparison
   table.
3. Use `main/classwise_results.csv` and the files in `figures` for per-emotion
   and confusion analysis.
4. Report `text`, `neural`, and `multiseed` as separate additional experiments;
   do not silently replace the canonical branches.
5. Use `speaker` and `duration` for limitations and implementation details.
6. Treat `audit` as pending until a human completes the review form.

## Important distinction

`report.ipynb` is a read-only reporting notebook. It reads the saved experiment
artifacts and does not retrain models. The training implementation is preserved
in `code/anime_ser.py`, while the frozen model outputs and checkpoints are in
`main`.

The original notebook containing cached 993-sample display output is not part of
this ready package.
