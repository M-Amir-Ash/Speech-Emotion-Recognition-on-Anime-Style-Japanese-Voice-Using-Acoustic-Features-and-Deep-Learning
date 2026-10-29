"""Read-only independent recomputation for BIAS_AND_LEAKAGE_AUDIT.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


TMP = Path(r"<USER_HOME>\AppData\Local\Temp")
OUT = Path(__file__).resolve().parent / "bias_leakage_recompute.json"
LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant_Surprise", "Sad"]


def metrics(y_true: pd.Series, y_pred: pd.Series, labels: list[str] = LABELS) -> dict[str, float | int]:
    return {
        "support": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def canonical_overlap() -> dict:
    base = TMP / "anime_ser_audit"
    frames = {name: pd.read_csv(base / f"{name}_manifest.csv", low_memory=False) for name in ("train", "validation", "test")}
    result: dict[str, object] = {"rows": {k: len(v) for k, v in frames.items()}}
    for field in ("source_group", "sha256", "clip_id", "file_path", "speaker", "character"):
        field_result = {}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if field not in frames[a] or field not in frames[b]:
                continue
            left = set(frames[a][field].dropna().astype(str))
            right = set(frames[b][field].dropna().astype(str))
            field_result[f"{a}__{b}"] = len(left & right)
        result[f"overlap_{field}"] = field_result
    result["speaker_composition"] = {
        k: v["speaker"].fillna("<NULL>").astype(str).value_counts().to_dict() for k, v in frames.items()
    }
    result["character_nonnull"] = {k: int(v["character"].notna().sum()) for k, v in frames.items()}
    result["class_counts"] = {k: v["emotion"].value_counts().sort_index().to_dict() for k, v in frames.items()}
    result["within_split_duplicate_sha"] = {k: int(v["sha256"].duplicated().sum()) for k, v in frames.items()}
    result["within_split_duplicate_path"] = {k: int(v["file_path"].duplicated().sum()) for k, v in frames.items()}
    return result


def sensitivity_metrics() -> dict:
    frame = pd.read_csv(TMP / "anime_ser_test_predictions.csv", low_memory=False)
    result: dict[str, object] = {"shape": list(frame.shape), "columns": list(frame.columns)}
    # Identify columns explicitly and fail closed if the schema differs.
    true_col = next(c for c in ("true_label", "target_emotion", "pseudo_label", "y_true") if c in frame)
    pred_col = next(c for c in ("predicted_label", "prediction", "pred_label", "y_pred") if c in frame)
    model_col = next(c for c in ("model", "model_name", "method") if c in frame)
    selector_col = next(c for c in ("selector_pass", "selected", "passes_selector") if c in frame)
    result["schema_map"] = {"true": true_col, "pred": pred_col, "model": model_col, "selector": selector_col}
    result["rows_per_model"] = frame[model_col].value_counts().to_dict()
    key_col = "clip_id" if "clip_id" in frame else "sample_id"
    result["duplicate_model_clip_rows_across_cohorts"] = int(frame.duplicated([model_col, key_col]).sum())
    recomputed = {}
    for model, group in frame.groupby(model_col):
        for cohort, cohort_rows in group.groupby("evaluation_cohort"):
            recomputed[f"{model}__{cohort}"] = metrics(cohort_rows[true_col], cohort_rows[pred_col])
            recomputed[f"{model}__{cohort}"]["coverage"] = float(len(cohort_rows) / 13462)
    result["metrics"] = recomputed
    result["selector_counts"] = frame.drop_duplicates([c for c in ("sample_id", "clip_id") if c in frame.columns])[selector_col].value_counts(dropna=False).to_dict()
    return result


def jvnv_metrics() -> dict:
    frame = pd.read_csv(TMP / "jvnv_predictions.csv", low_memory=False)
    success = frame["predicted_emotion"].notna()
    accepted = frame["confidence_status"].eq("auto_accept")
    rejected = frame["confidence_status"].eq("reject")
    returned = frame[success]
    accepted_rows = frame[accepted & success]
    reference_labels = sorted(frame["reference_emotion"].dropna().unique().tolist())
    return {
        "rows": int(len(frame)),
        "unique_clip_ids": int(frame["clip_id"].nunique()),
        "duplicate_clip_ids": int(frame["clip_id"].duplicated().sum()),
        "successful": int(success.sum()),
        "full_population_raw_agreement": float(((frame["predicted_emotion"] == frame["reference_emotion"]) & success).sum() / len(frame)),
        "returned_metrics_six_reference_classes": metrics(returned["reference_emotion"], returned["predicted_emotion"], reference_labels),
        "returned_metrics_seven_model_classes": metrics(returned["reference_emotion"], returned["predicted_emotion"], LABELS),
        "reference_label_set": reference_labels,
        "auto_accepted": int(accepted.sum()),
        "acceptance_coverage": float(accepted.sum() / len(frame)),
        "accepted_accuracy": float((accepted_rows["predicted_emotion"] == accepted_rows["reference_emotion"]).mean()),
        "needs_review": int(frame["confidence_status"].eq("needs_review").sum()),
        "rejected": int(rejected.sum()),
        "quality_status": frame["audio_quality_status"].value_counts(dropna=False).to_dict(),
        "reference_counts": frame["reference_emotion"].value_counts().sort_index().to_dict(),
        "prediction_counts_returned": returned["predicted_emotion"].value_counts().sort_index().to_dict(),
    }


def main() -> None:
    result = {
        "canonical_overlap": canonical_overlap(),
        "selection_sensitivity": sensitivity_metrics(),
        "jvnv": jvnv_metrics(),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
