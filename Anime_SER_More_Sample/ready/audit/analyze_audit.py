"""Analyze completed human reviews without changing the frozen data or audit master."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "pleasant_surprise",
    "sad",
]
REVIEW_FIELDS = ["reviewer_emotion", "reviewer_uncertain", "reviewer_notes"]
TRUE_VALUES = {"yes", "y", "true", "1"}
FALSE_VALUES = {"no", "n", "false", "0"}


def normalize_emotion(value: object) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text


def parse_uncertain(value: object) -> bool | None:
    if pd.isna(value) or not str(value).strip():
        return None
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(
        f"Invalid reviewer_uncertain value {value!r}; use yes/no, true/false, 1/0, or leave blank"
    )


def load_reviews(master_path: Path, reviews_path: Path) -> pd.DataFrame:
    master = pd.read_csv(master_path, dtype=str, keep_default_na=False)
    required_master = {
        "audit_id", "file_path", "filename", "original_emotion", "source_media",
        "duration_sec", "transcript",
    }
    missing_master = sorted(required_master - set(master.columns))
    if missing_master:
        raise ValueError(f"Audit master lacks required columns: {missing_master}")
    if master["audit_id"].duplicated().any():
        raise ValueError("Audit master contains duplicate audit_id values")

    reviews = pd.read_csv(reviews_path, dtype=str, keep_default_na=False)
    missing_reviews = sorted(({"audit_id"} | set(REVIEW_FIELDS)) - set(reviews.columns))
    if missing_reviews:
        raise ValueError(f"Review file lacks required columns: {missing_reviews}")
    if reviews["audit_id"].duplicated().any():
        raise ValueError("Review file contains duplicate audit_id values")
    unknown_ids = sorted(set(reviews["audit_id"]) - set(master["audit_id"]))
    if unknown_ids:
        raise ValueError(f"Review file contains unknown audit_id values, for example: {unknown_ids[0]}")

    merged = master.drop(columns=[column for column in REVIEW_FIELDS if column in master], errors="ignore")
    merged = merged.merge(reviews[["audit_id"] + REVIEW_FIELDS], on="audit_id", how="left", validate="one_to_one")
    for column in REVIEW_FIELDS:
        merged[column] = merged[column].fillna("").astype(str)
    return merged


def analyze(master_path: Path, reviews_path: Path, output_dir: Path) -> dict[str, object]:
    data = load_reviews(master_path, reviews_path)
    data["original_emotion"] = data["original_emotion"].map(normalize_emotion)
    if not data["original_emotion"].isin(EMOTIONS).all():
        invalid = sorted(set(data.loc[~data["original_emotion"].isin(EMOTIONS), "original_emotion"]))
        raise ValueError(f"Invalid original emotion value(s): {invalid}")

    normalized_reviews = data["reviewer_emotion"].map(normalize_emotion)
    reviewed = normalized_reviews.ne("")
    invalid_reviews = sorted(set(normalized_reviews[reviewed & ~normalized_reviews.isin(EMOTIONS)]))
    if invalid_reviews:
        raise ValueError(f"Invalid reviewer emotion value(s): {invalid_reviews}; allowed values: {EMOTIONS}")
    data["reviewer_emotion"] = normalized_reviews.where(reviewed, "")
    data["reviewer_uncertain_normalized"] = data["reviewer_uncertain"].map(parse_uncertain)
    data["agreement"] = np.where(
        reviewed,
        np.where(data["original_emotion"].eq(data["reviewer_emotion"]), "yes", "no"),
        "",
    )

    reviewed_data = data.loc[reviewed].copy()
    agreement_count = int(reviewed_data["agreement"].eq("yes").sum())
    disagreement_count = int(reviewed_data["agreement"].eq("no").sum())
    uncertain_count = int(data["reviewer_uncertain_normalized"].eq(True).sum())
    overall_agreement = agreement_count / len(reviewed_data) if len(reviewed_data) else None
    kappa = None
    if len(reviewed_data) >= 2:
        calculated = cohen_kappa_score(
            reviewed_data["original_emotion"],
            reviewed_data["reviewer_emotion"],
            labels=EMOTIONS,
        )
        kappa = None if pd.isna(calculated) else float(calculated)

    summary = {
        "total_audit_samples": int(len(data)),
        "reviewed_count": int(len(reviewed_data)),
        "incomplete_count": int((~reviewed).sum()),
        "agreement_count": agreement_count,
        "disagreement_count": disagreement_count,
        "uncertain_count": uncertain_count,
        "overall_agreement": overall_agreement,
        "cohen_kappa": kappa,
        "agreement_denominator": "rows with a nonblank valid reviewer_emotion",
    }

    per_class_rows = []
    for emotion in EMOTIONS:
        class_all = data[data["original_emotion"].eq(emotion)]
        class_reviewed = class_all[class_all["reviewer_emotion"].ne("")]
        agree = int(class_reviewed["agreement"].eq("yes").sum())
        per_class_rows.append({
            "original_emotion": emotion,
            "audit_samples": int(len(class_all)),
            "reviewed_count": int(len(class_reviewed)),
            "agreement_count": agree,
            "disagreement_count": int(class_reviewed["agreement"].eq("no").sum()),
            "uncertain_count": int(class_all["reviewer_uncertain_normalized"].eq(True).sum()),
            "agreement_rate": agree / len(class_reviewed) if len(class_reviewed) else np.nan,
        })
    per_class = pd.DataFrame(per_class_rows)

    matrix = confusion_matrix(
        reviewed_data["original_emotion"],
        reviewed_data["reviewer_emotion"],
        labels=EMOTIONS,
    ) if len(reviewed_data) else np.zeros((len(EMOTIONS), len(EMOTIONS)), dtype=int)
    confusion = pd.DataFrame(matrix, index=EMOTIONS, columns=EMOTIONS)
    confusion.index.name = "original_emotion"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame([summary]).to_csv(output_dir / "audit_summary.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(output_dir / "per_class_agreement.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    data.drop(columns="reviewer_uncertain_normalized").to_csv(
        output_dir / "audit_scored.csv", index=False, encoding="utf-8-sig"
    )
    return summary


def parse_args() -> argparse.Namespace:
    audit_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=audit_dir / "audit_samples.csv")
    parser.add_argument(
        "--reviews",
        type=Path,
        default=audit_dir / "audit_review_form.csv",
        help="Completed audit_samples.csv or the preferred blinded audit_review_form.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=audit_dir / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze(args.master.resolve(), args.reviews.resolve(), args.output_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
