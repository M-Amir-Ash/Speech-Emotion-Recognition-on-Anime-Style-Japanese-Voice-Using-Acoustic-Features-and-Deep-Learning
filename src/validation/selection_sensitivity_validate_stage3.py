#!/usr/bin/env python
"""Validate and hash the completed one-time Stage 3 outputs without rescoring."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CLASSES = {"Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant_Surprise", "Sad"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    coverage = pd.read_csv(HERE / "test_coverage.csv")
    metrics = pd.read_csv(HERE / "test_metrics.csv")
    classwise = pd.read_csv(HERE / "classwise_test_metrics.csv")
    predictions = pd.read_csv(HERE / "test_predictions.csv")
    intervals = pd.read_csv(HERE / "bootstrap_intervals.csv")
    integrity = json.loads((HERE / "integrity_validation.json").read_text(encoding="utf-8"))
    expected_counts = {"full_test_population": 13462, "label_conditioned_selector_passing_test": 455}
    observed = predictions.groupby(["model", "evaluation_cohort"]).size().unstack().to_dict("index")
    for model in ["mfcc_svm", "egemaps_xgboost"]:
        assert observed[model] == expected_counts
    assert coverage.loc[coverage.pseudo_class.eq("OVERALL"), "selected_count"].iloc[0] == 455
    assert coverage.loc[coverage.pseudo_class.eq("OVERALL"), "quality_pass_denominator"].iloc[0] == 13462
    assert set(predictions.true_label) == CLASSES and set(predictions.predicted_label) == CLASSES
    assert len(metrics) == 4 and len(classwise) == 28 and len(intervals) == 40
    assert not classwise.support.isna().any()
    for matrix_path in (HERE / "confusion_matrices").glob("*.csv"):
        matrix = pd.read_csv(matrix_path)
        cohort = "label_conditioned_selector_passing_test" if "label_conditioned" in matrix_path.name else "full_test_population"
        assert int(matrix.drop(columns=[matrix.columns[0]]).to_numpy().sum()) == expected_counts[cohort]
    assert integrity["source_group_overlaps"] == {"train_validation": 0, "train_test": 0, "validation_test": 0}
    assert integrity["selector_label_conditioned"] is True
    assert integrity["no_refitting_or_tuning"] is True
    integrity["postrun_validation"] = {
        "status": "PASS", "prediction_rows": len(predictions), "metrics_rows": len(metrics),
        "classwise_rows": len(classwise), "bootstrap_rows": len(intervals),
        "confusion_matrix_totals_reconciled": True, "taxonomy": sorted(CLASSES),
        "test_run_terminal": True,
    }
    output_names = [
        "test_coverage.csv", "test_metrics.csv", "classwise_test_metrics.csv", "test_predictions.csv",
        "bootstrap_intervals.csv", "frozen_artifact_manifest.csv", "methodology_and_limitations.md",
        "historical_comparison.md", "stage3_summary.md", "run_frozen_test_evaluation.py", "validate_stage3.py",
        "confusion_matrices/egemaps_xgboost__full_test_population.csv",
        "confusion_matrices/egemaps_xgboost__label_conditioned_selector_passing_test.csv",
        "confusion_matrices/mfcc_svm__full_test_population.csv",
        "confusion_matrices/mfcc_svm__label_conditioned_selector_passing_test.csv",
    ]
    integrity["output_hashes"] = {name: digest(HERE / name) for name in output_names}
    atomic_json(HERE / "integrity_validation.json", integrity)
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()
