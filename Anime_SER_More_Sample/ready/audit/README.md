# Human pseudo-label audit

This directory audits a deterministic sample of the frozen 1,969-clip dataset. It does not relabel, move, or modify any source audio or frozen experiment output.

## Review procedure

1. Open `audit_review_form.csv`. This is the preferred reviewer file because it does not show the original pseudo-label.
2. For every row, listen to the audio at `file_path`. Judge the audio performance, not the transcript alone.
3. Enter exactly one of these values in `reviewer_emotion`:
   `angry`, `disgust`, `fear`, `happy`, `neutral`, `pleasant_surprise`, or `sad`.
4. Enter `yes` or `no` in `reviewer_uncertain`.
5. Add optional reasoning in `reviewer_notes`.
6. Do not change `audit_id` or other source columns. Save the completed file as CSV UTF-8.

No suggested alternative label or model prediction is included. `audit_samples.csv` is the traceable master containing the original pseudo-label; keep it unchanged and use the blinded form for review.

## Analyze a completed review

From the `Anime_SER_More_Sample` directory, run:

```powershell
python label_audit\analyze_audit.py --reviews label_audit\audit_review_form.csv
```

The command writes separate files under `label_audit/results/`:

- `audit_summary.json` and `audit_summary.csv`
- `per_class_agreement.csv`
- `confusion_matrix.csv`
- `audit_scored.csv`

Agreement uses all rows with a valid, nonblank `reviewer_emotion`. Blank reviews remain incomplete and are not included in the agreement or Cohen's-kappa denominator. Uncertain rows are counted separately but remain in agreement calculations when an emotion was provided.

## Recreate the blank deterministic sample

The committed audit already uses 15 clips per class and seed `20260905`. The generator refuses to overwrite existing review files unless `--overwrite` is explicitly supplied:

```powershell
python label_audit\create_audit_sample.py
```

Do not use `--overwrite` after human review has started.
