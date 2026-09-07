"""Run the Japanese character n-gram text baseline in two guarded phases.

The ``tune`` phase reads only the frozen train and validation manifests for
model selection. The ``evaluate`` phase may be called once afterward and uses
the frozen test manifest without changing the original experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "pleasant_surprise",
    "sad",
]
EXPECTED_SPLIT_ROWS = {"train": 1313, "validation": 261, "test": 395}
NGRAM_RANGES = [(2, 4), (2, 5), (3, 5)]
MIN_DF_VALUES = [1, 2, 3]
C_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
METHOD_NAME = "Character TF-IDF + Logistic Regression"

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
FROZEN_OUTPUT_DIR = PROJECT_ROOT / "output"
MANIFEST_PATHS = {
    split: FROZEN_OUTPUT_DIR / f"{split}_manifest.csv"
    for split in ("train", "validation", "test")
}

TUNING_PATH = EXPERIMENT_DIR / "tuning_results.csv"
SELECTED_PATH = EXPERIMENT_DIR / "selected_hyperparameters.csv"
CONFIG_PATH = EXPERIMENT_DIR / "experiment_config.json"
FINAL_PATH = EXPERIMENT_DIR / "final_results.csv"
CLASSWISE_PATH = EXPERIMENT_DIR / "classwise_results.csv"
CONFUSION_PATH = EXPERIMENT_DIR / "confusion_matrix.csv"
COMPARISON_PATH = EXPERIMENT_DIR / "comparison_with_word_baseline.csv"
PER_CLASS_COMPARISON_PATH = EXPERIMENT_DIR / "per_class_f1_comparison.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_split(split: str) -> pd.DataFrame:
    path = MANIFEST_PATHS[split]
    if not path.is_file():
        raise FileNotFoundError(f"Frozen {split} manifest is missing: {path}")
    frame = pd.read_csv(path, low_memory=False)
    required = {"file_path", "filename", "emotion", "label_id", "transcript", "source_group", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Frozen {split} manifest lacks required columns: {missing}")
    if len(frame) != EXPECTED_SPLIT_ROWS[split]:
        raise ValueError(
            f"Frozen {split} membership changed: expected {EXPECTED_SPLIT_ROWS[split]}, found {len(frame)}"
        )
    if not frame["split"].astype(str).eq(split).all():
        raise ValueError(f"Unexpected split marker in {path}")
    if frame["file_path"].duplicated().any():
        raise ValueError(f"Duplicate file paths in frozen {split} manifest")
    if frame["transcript"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError(f"Frozen {split} manifest does not have 100% transcript coverage")
    if set(frame["emotion"].astype(str)) != set(EMOTIONS):
        raise ValueError(f"Frozen {split} manifest does not contain all seven emotions")
    expected_label = frame["emotion"].astype(str).map({emotion: index for index, emotion in enumerate(EMOTIONS)})
    if not expected_label.eq(pd.to_numeric(frame["label_id"], errors="raise")).all():
        raise ValueError(f"Emotion and label_id disagree in frozen {split} manifest")
    return frame


def build_model(ngram_range: tuple[int, int], min_df: int, c_value: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            min_df=min_df,
            lowercase=True,
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
        )),
        ("model", LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            n_jobs=1,
            solver="lbfgs",
        )),
    ])


def metric_values(true_labels: np.ndarray, predicted_labels: np.ndarray) -> dict[str, float]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=np.arange(len(EMOTIONS)),
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
    }


def software_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "scikit-learn"):
        versions[package] = importlib.metadata.version(package)
    return versions


def refuse_existing(paths: list[Path], phase: str) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite {phase} artifact(s): {existing}. "
            "Create a new versioned experiment directory instead."
        )


def tune() -> None:
    refuse_existing([TUNING_PATH, SELECTED_PATH, CONFIG_PATH], "validation-selection")
    train = load_frozen_split("train")
    validation = load_frozen_split("validation")

    train_text = train["transcript"].astype(str)
    train_labels = train["label_id"].to_numpy(dtype=int)
    validation_text = validation["transcript"].astype(str)
    validation_labels = validation["label_id"].to_numpy(dtype=int)

    tuning_rows = []
    candidates = list(product(NGRAM_RANGES, MIN_DF_VALUES, C_VALUES))
    for candidate_index, (ngram_range, min_df, c_value) in enumerate(candidates, start=1):
        model = build_model(ngram_range, min_df, c_value)
        model.fit(train_text, train_labels)
        validation_prediction = model.predict(validation_text)
        metrics = metric_values(validation_labels, validation_prediction)
        tuning_rows.append({
            "candidate_index": candidate_index,
            "analyzer": "char",
            "ngram_range": f"({ngram_range[0]}, {ngram_range[1]})",
            "ngram_min": ngram_range[0],
            "ngram_max": ngram_range[1],
            "min_df": min_df,
            "C": c_value,
            "class_weight": "balanced",
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_precision": metrics["macro_precision"],
            "validation_macro_recall": metrics["macro_recall"],
            "validation_macro_f1": metrics["macro_f1"],
            "feature_count": len(model.named_steps["tfidf"].vocabulary_),
        })
        print(
            f"Candidate {candidate_index:02d}/{len(candidates)} "
            f"ngram={ngram_range} min_df={min_df} C={c_value:g} "
            f"validation_macro_f1={metrics['macro_f1']:.6f}"
        )

    tuning = pd.DataFrame(tuning_rows).sort_values(
        ["validation_macro_f1", "validation_accuracy", "candidate_index"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    tuning.insert(0, "validation_rank", np.arange(1, len(tuning) + 1))
    selected = tuning.head(1).copy()

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(TUNING_PATH, index=False, encoding="utf-8-sig")
    selected.to_csv(SELECTED_PATH, index=False, encoding="utf-8-sig")

    config = {
        "experiment_name": "text_char_ngram",
        "status": "validation_selected_test_not_evaluated",
        "method": METHOD_NAME,
        "purpose": "Japanese character n-gram baseline; original word-level baseline remains unchanged",
        "random_state": RANDOM_STATE,
        "frozen_split_rows": EXPECTED_SPLIT_ROWS,
        "selection_data": ["train", "validation"],
        "selection_metric": "validation_macro_f1",
        "tie_breaking": ["validation_accuracy descending", "candidate_index ascending"],
        "test_used_during_selection": False,
        "final_fit_population": "frozen train split only, matching the original word-level baseline",
        "candidate_grid": {
            "analyzer": ["char"],
            "ngram_range": [list(value) for value in NGRAM_RANGES],
            "min_df": MIN_DF_VALUES,
            "C": C_VALUES,
        },
        "candidate_count": len(candidates),
        "vectorizer_fixed_settings": {
            "lowercase": True,
            "norm": "l2",
            "use_idf": True,
            "smooth_idf": True,
            "sublinear_tf": False,
        },
        "logistic_regression_fixed_settings": {
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
            "solver": "lbfgs",
        },
        "fusion_modified": False,
        "source_manifests": {
            split: {"path": str(path), "sha256": sha256_file(path)}
            for split, path in MANIFEST_PATHS.items()
        },
        "tuning_results_sha256": sha256_file(TUNING_PATH),
        "selected_hyperparameters_sha256": sha256_file(SELECTED_PATH),
        "software_versions": software_versions(),
    }
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nValidation selection complete. The test manifest was not loaded for model selection.")
    print(selected.to_string(index=False))


def validate_frozen_selection() -> tuple[dict[str, object], pd.Series]:
    if not CONFIG_PATH.is_file() or not SELECTED_PATH.is_file() or not TUNING_PATH.is_file():
        raise FileNotFoundError("Run the tune phase before one-time test evaluation")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("status") != "validation_selected_test_not_evaluated":
        raise RuntimeError(f"Experiment is not eligible for first test evaluation: {config.get('status')}")
    if sha256_file(TUNING_PATH) != config["tuning_results_sha256"]:
        raise ValueError("Validation tuning results changed after selection")
    if sha256_file(SELECTED_PATH) != config["selected_hyperparameters_sha256"]:
        raise ValueError("Selected hyperparameters changed after being frozen")
    selected_table = pd.read_csv(SELECTED_PATH)
    if len(selected_table) != 1:
        raise ValueError("Expected exactly one frozen selected configuration")
    return config, selected_table.iloc[0]


def evaluate_once() -> None:
    final_artifacts = [
        FINAL_PATH,
        CLASSWISE_PATH,
        CONFUSION_PATH,
        COMPARISON_PATH,
        PER_CLASS_COMPARISON_PATH,
    ]
    refuse_existing(final_artifacts, "final-test")
    config, selected = validate_frozen_selection()

    # Only this guarded phase loads the final test manifest.
    train = load_frozen_split("train")
    test = load_frozen_split("test")
    if sha256_file(MANIFEST_PATHS["test"]) != config["source_manifests"]["test"]["sha256"]:
        raise ValueError("Frozen test manifest changed after validation selection")

    ngram_range = (int(selected["ngram_min"]), int(selected["ngram_max"]))
    model = build_model(ngram_range, int(selected["min_df"]), float(selected["C"]))
    model.fit(train["transcript"].astype(str), train["label_id"].to_numpy(dtype=int))
    test_probabilities = model.predict_proba(test["transcript"].astype(str))
    test_prediction = model.classes_[test_probabilities.argmax(axis=1)].astype(int)
    test_labels = test["label_id"].to_numpy(dtype=int)
    metrics = metric_values(test_labels, test_prediction)

    final_result = pd.DataFrame([{
        "method": METHOD_NAME,
        "analyzer": "char",
        "ngram_range": selected["ngram_range"],
        "min_df": int(selected["min_df"]),
        "C": float(selected["C"]),
        "validation_macro_f1": float(selected["validation_macro_f1"]),
        "test_accuracy": metrics["accuracy"],
        "test_macro_precision": metrics["macro_precision"],
        "test_macro_recall": metrics["macro_recall"],
        "test_macro_f1": metrics["macro_f1"],
        "test_samples": len(test),
    }])

    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels,
        test_prediction,
        labels=np.arange(len(EMOTIONS)),
        zero_division=0,
    )
    classwise = pd.DataFrame({
        "method": METHOD_NAME,
        "emotion": EMOTIONS,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
    })
    matrix = pd.DataFrame(
        confusion_matrix(test_labels, test_prediction, labels=np.arange(len(EMOTIONS))),
        index=EMOTIONS,
        columns=EMOTIONS,
    )
    matrix.index.name = "actual_emotion"

    original_result_table = pd.read_csv(FROZEN_OUTPUT_DIR / "anime_only_final_results.csv")
    original_result = original_result_table[
        original_result_table["method"].eq("TF-IDF + Logistic Regression")
    ]
    if len(original_result) != 1:
        raise ValueError("Could not identify the single frozen word-level TF-IDF result")
    original_result = original_result.iloc[0]
    original_selected_table = pd.read_csv(FROZEN_OUTPUT_DIR / "selected_hyperparameters.csv")
    original_selected = original_selected_table[
        original_selected_table["method"].eq("TF-IDF + Logistic Regression")
    ]
    if len(original_selected) != 1:
        raise ValueError("Could not identify the frozen word-level validation result")
    original_validation_f1 = float(original_selected.iloc[0]["validation_macro_f1"])

    comparison = pd.DataFrame([
        {
            "method": "Original word TF-IDF + Logistic Regression",
            "representation": "word unigram/bigram",
            "validation_macro_f1": original_validation_f1,
            "test_accuracy": float(original_result["accuracy"]),
            "test_macro_precision": float(original_result["macro_precision"]),
            "test_macro_recall": float(original_result["macro_recall"]),
            "test_macro_f1": float(original_result["macro_f1"]),
        },
        {
            "method": METHOD_NAME,
            "representation": f"character {selected['ngram_range']}",
            "validation_macro_f1": float(selected["validation_macro_f1"]),
            "test_accuracy": metrics["accuracy"],
            "test_macro_precision": metrics["macro_precision"],
            "test_macro_recall": metrics["macro_recall"],
            "test_macro_f1": metrics["macro_f1"],
        },
    ])
    for metric in ("validation_macro_f1", "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1"):
        comparison[f"delta_vs_original_{metric}"] = comparison[metric] - comparison.loc[0, metric]

    original_classwise = pd.read_csv(FROZEN_OUTPUT_DIR / "classwise_results.csv")
    original_classwise = original_classwise[
        original_classwise["method"].eq("TF-IDF + Logistic Regression")
    ].set_index("emotion")
    per_class_comparison = classwise[["emotion", "f1", "support"]].rename(
        columns={"f1": "character_ngram_f1", "support": "test_support"}
    )
    per_class_comparison["original_word_f1"] = per_class_comparison["emotion"].map(original_classwise["f1"])
    per_class_comparison["f1_delta_vs_original"] = (
        per_class_comparison["character_ngram_f1"] - per_class_comparison["original_word_f1"]
    )

    final_result.to_csv(FINAL_PATH, index=False, encoding="utf-8-sig")
    classwise.to_csv(CLASSWISE_PATH, index=False, encoding="utf-8-sig")
    matrix.to_csv(CONFUSION_PATH, encoding="utf-8-sig")
    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")
    per_class_comparison.to_csv(PER_CLASS_COMPARISON_PATH, index=False, encoding="utf-8-sig")

    config.update({
        "status": "completed_one_time_test_evaluation",
        "test_evaluation_count": 1,
        "test_samples": len(test),
        "selected_configuration": {
            "analyzer": "char",
            "ngram_range": [ngram_range[0], ngram_range[1]],
            "min_df": int(selected["min_df"]),
            "C": float(selected["C"]),
        },
        "final_artifact_sha256": {
            path.name: sha256_file(path) for path in final_artifacts
        },
        "original_word_baseline_modified": False,
        "fusion_modified": False,
    })
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print("One-time final test evaluation complete.")
    print(final_result.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("tune", "evaluate"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "tune":
        tune()
    else:
        evaluate_once()


if __name__ == "__main__":
    main()
