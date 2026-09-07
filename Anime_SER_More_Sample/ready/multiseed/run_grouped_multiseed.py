"""Improvement 5: source-group-safe split variability analysis.

Only the predefined group-split seed changes. Model configurations are frozen,
and test labels are used only for one final evaluation per seed/model pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import librosa
import numpy as np
import opensmile
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
FROZEN_OUTPUT = ROOT / "output"
EXPERIMENT_DIR = Path(__file__).resolve().parent
SEEDS = [42, 123, 2026, 3407, 7777]
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]
EXPECTED_TOTAL = 1969
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.12
SPLIT_ATTEMPTS = 300
MODEL_RANDOM_STATE = 42
TARGET_SR = 16_000
MFCC_CORRELATION_THRESHOLD = 0.95
N_JOBS = 1

MODEL_NAMES = {
    "opensmile_xgboost": "openSMILE eGeMAPS + XGBoost",
    "mfcc_svm": "MFCC + SVM",
}
XGB_CONFIG = {
    "max_depth": 5,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
SVM_CONFIG = {"C": 100.0, "gamma": 0.001, "kernel": "rbf", "class_weight": "balanced"}
MFCC_STATISTICS = ["mean", "std", "max", "min", "variance", "q1", "median", "q3", "iqr"]
MFCC_FEATURE_NAMES = [f"{stat}_C{coefficient}" for stat in MFCC_STATISTICS for coefficient in range(13)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_canonical_metadata() -> tuple[pd.DataFrame, dict[str, set[str]]]:
    frames = {}
    required = {"file_path", "filename", "emotion", "label_id", "source_group", "source_media", "sha256", "split"}
    expected_rows = {"train": 1313, "validation": 261, "test": 395}
    for split, expected in expected_rows.items():
        frame = pd.read_csv(FROZEN_OUTPUT / f"{split}_manifest.csv", low_memory=False)
        if len(frame) != expected:
            raise RuntimeError(f"Canonical {split} rows changed: expected {expected}, found {len(frame)}")
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(f"Canonical {split} manifest lacks {sorted(missing)}")
        frames[split] = frame
    metadata = pd.concat(frames.values(), ignore_index=True)
    if len(metadata) != EXPECTED_TOTAL or metadata["file_path"].duplicated().any():
        raise RuntimeError("Canonical dataset is not exactly 1,969 unique file paths")
    if metadata["sha256"].duplicated().any():
        raise RuntimeError("Canonical dataset contains duplicate SHA-256 values")
    mapping = dict(zip(EMOTIONS, range(len(EMOTIONS))))
    expected_ids = metadata["emotion"].map(mapping)
    if expected_ids.isna().any() or not np.array_equal(expected_ids.to_numpy(int), metadata["label_id"].to_numpy(int)):
        raise RuntimeError("Canonical emotion/label_id mapping changed")
    missing_files = [path for path in metadata["file_path"].astype(str) if not Path(path).is_file()]
    if missing_files:
        raise RuntimeError(f"Missing canonical audio file: {missing_files[0]}")
    # This matches the canonical notebook's pre-split row ordering exactly.
    metadata = metadata.sort_values(["emotion", "filename"]).reset_index(drop=True)
    memberships = {split: set(frame["file_path"].astype(str)) for split, frame in frames.items()}
    return metadata, memberships


def best_group_holdout(df: pd.DataFrame, holdout_size: float, seed: int) -> tuple[np.ndarray, np.ndarray, int, float]:
    target = df["emotion"].value_counts(normalize=True).reindex(EMOTIONS, fill_value=0)
    best = None
    for offset in range(SPLIT_ATTEMPTS):
        splitter = GroupShuffleSplit(n_splits=1, test_size=holdout_size, random_state=seed + offset)
        development_idx, holdout_idx = next(splitter.split(df, groups=df["source_group"]))
        counts = df.iloc[holdout_idx]["emotion"].value_counts().reindex(EMOTIONS, fill_value=0)
        if (counts == 0).any():
            continue
        distribution_error = float((target - counts / counts.sum()).abs().sum())
        size_error = abs(len(holdout_idx) / len(df) - holdout_size)
        minority_penalty = float((1.0 / counts.clip(lower=1)).sum()) * 0.01
        score = distribution_error + size_error + minority_penalty
        if best is None or score < best[0]:
            best = (score, development_idx, holdout_idx, offset)
    if best is None:
        raise RuntimeError(f"No valid all-class group holdout for seed={seed}")
    return best[1], best[2], best[3], best[0]


def make_split(metadata: pd.DataFrame, seed: int) -> tuple[dict[str, pd.DataFrame], dict]:
    train_val_idx, test_idx, test_offset, test_score = best_group_holdout(metadata, TEST_SIZE, seed)
    train_val = metadata.iloc[train_val_idx].reset_index(drop=True)
    test = metadata.iloc[test_idx].reset_index(drop=True)
    validation_fraction = VALIDATION_SIZE / (1.0 - TEST_SIZE)
    train_idx, validation_idx, validation_offset, validation_score = best_group_holdout(
        train_val, validation_fraction, seed + 10_000
    )
    frames = {
        "train": train_val.iloc[train_idx].reset_index(drop=True),
        "validation": train_val.iloc[validation_idx].reset_index(drop=True),
        "test": test,
    }
    details = {
        "test_attempt_offset": test_offset,
        "test_split_score": test_score,
        "validation_attempt_offset": validation_offset,
        "validation_split_score": validation_score,
    }
    return frames, details


def audit_split(seed: int, frames: dict[str, pd.DataFrame]) -> dict:
    combined = pd.concat(
        [frame.assign(generated_split=split) for split, frame in frames.items()], ignore_index=True
    )
    group_sets = {split: set(frame["source_group"].astype(str)) for split, frame in frames.items()}
    path_cross = combined.groupby("file_path")["generated_split"].nunique()
    sha_cross = combined.groupby("sha256")["generated_split"].nunique()
    emotion_missing = {
        split: sorted(set(EMOTIONS) - set(frame["emotion"])) for split, frame in frames.items()
    }
    row = {
        "seed": seed,
        "assigned_rows": len(combined),
        "unique_file_paths": combined["file_path"].nunique(),
        "unique_sha256": combined["sha256"].nunique(),
        "duplicate_path_rows": int(combined["file_path"].duplicated().sum()),
        "duplicate_sha256_rows": int(combined["sha256"].duplicated().sum()),
        "paths_crossing_splits": int((path_cross > 1).sum()),
        "sha256_crossing_splits": int((sha_cross > 1).sum()),
        "train_validation_group_overlap": len(group_sets["train"] & group_sets["validation"]),
        "train_test_group_overlap": len(group_sets["train"] & group_sets["test"]),
        "validation_test_group_overlap": len(group_sets["validation"] & group_sets["test"]),
        "missing_emotions_train": ";".join(emotion_missing["train"]),
        "missing_emotions_validation": ";".join(emotion_missing["validation"]),
        "missing_emotions_test": ";".join(emotion_missing["test"]),
    }
    row["all_checks_pass"] = (
        row["assigned_rows"] == EXPECTED_TOTAL
        and row["unique_file_paths"] == EXPECTED_TOTAL
        and row["unique_sha256"] == EXPECTED_TOTAL
        and all(row[key] == 0 for key in [
            "duplicate_path_rows", "duplicate_sha256_rows", "paths_crossing_splits",
            "sha256_crossing_splits", "train_validation_group_overlap",
            "train_test_group_overlap", "validation_test_group_overlap",
        ])
        and all(not row[key] for key in [
            "missing_emotions_train", "missing_emotions_validation", "missing_emotions_test"
        ])
    )
    if not row["all_checks_pass"]:
        raise RuntimeError(f"Split audit failed for seed {seed}: {row}")
    return row


def load_common_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    peak = np.max(np.abs(audio)) if len(audio) else 0.0
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)


def mfcc_statistics(path: str) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=load_common_audio(path), sr=TARGET_SR, n_mfcc=13)
    q1 = np.quantile(mfcc, 0.25, axis=1)
    median = np.quantile(mfcc, 0.50, axis=1)
    q3 = np.quantile(mfcc, 0.75, axis=1)
    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1), mfcc.max(axis=1), mfcc.min(axis=1),
        mfcc.var(axis=1), q1, median, q3, q3 - q1,
    ]).astype(np.float32)


SMILE = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)


def opensmile_features(path: str) -> np.ndarray:
    values = SMILE.process_signal(load_common_audio(path), TARGET_SR).select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    q1 = np.quantile(values, 0.25, axis=0)
    median = np.quantile(values, 0.50, axis=0)
    q3 = np.quantile(values, 0.75, axis=0)
    return np.concatenate([
        values.mean(axis=0), values.std(axis=0), values.min(axis=0), values.max(axis=0),
        q1, median, q3,
    ]).astype(np.float32)


def get_feature_matrix(metadata: pd.DataFrame, kind: str) -> np.ndarray:
    cache_path = EXPERIMENT_DIR / f"{kind}_features.npy"
    index_path = EXPERIMENT_DIR / "feature_index.csv"
    expected_index = metadata[["file_path", "sha256"]].copy()
    if cache_path.exists() and index_path.exists():
        stored_index = pd.read_csv(index_path)
        matrix = np.load(cache_path, allow_pickle=False)
        if stored_index.equals(expected_index) and len(matrix) == len(metadata) and np.isfinite(matrix).all():
            print(f"Reusing {kind} cache: {matrix.shape}", flush=True)
            return matrix
        raise RuntimeError(f"Existing {kind} cache does not match the frozen dataset")
    extractor = mfcc_statistics if kind == "mfcc" else opensmile_features
    print(f"Extracting {kind} features for {len(metadata)} clips ...", flush=True)
    rows = []
    started = time.time()
    for index, path in enumerate(metadata["file_path"].astype(str), start=1):
        rows.append(extractor(path))
        if index == 1 or index % 50 == 0 or index == len(metadata):
            print(f"  {kind}: {index}/{len(metadata)} ({time.time()-started:.1f}s)", flush=True)
    matrix = np.stack(rows)
    np.save(cache_path, matrix, allow_pickle=False)
    if not index_path.exists():
        expected_index.to_csv(index_path, index=False, encoding="utf-8-sig")
    return matrix


def retained_mfcc_indices(training: np.ndarray) -> list[int]:
    correlation = np.nan_to_num(
        np.corrcoef(training, rowvar=False), nan=0.0, posinf=0.0, neginf=0.0
    )
    variance = training.var(axis=0)
    retained = []
    for feature_index in range(training.shape[1]):
        if variance[feature_index] <= 1e-12:
            continue
        if retained:
            candidate = np.abs(correlation[feature_index, retained])
            if float(candidate.max()) > MFCC_CORRELATION_THRESHOLD:
                continue
        retained.append(feature_index)
    return retained


def metric_row(seed: int, model_key: str, truth: np.ndarray, prediction: np.ndarray, retained: int | None) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, prediction, labels=range(len(EMOTIONS)), average="macro", zero_division=0
    )
    return {
        "seed": seed,
        "model": MODEL_NAMES[model_key],
        "accuracy": accuracy_score(truth, prediction),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "mfcc_retained_features": retained if retained is not None else "",
        "hyperparameters_frozen_before_test": True,
        "test_used_for_tuning": False,
    }


def classwise_rows(seed: int, model_key: str, truth: np.ndarray, prediction: np.ndarray) -> list[dict]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=range(len(EMOTIONS)), zero_division=0
    )
    return [
        {
            "seed": seed, "model": MODEL_NAMES[model_key], "emotion": emotion,
            "precision": precision[i], "recall": recall[i], "f1": f1[i], "support": int(support[i]),
        }
        for i, emotion in enumerate(EMOTIONS)
    ]


def markdown_table(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: f"{value:.6f}")
    shown = shown.fillna("").astype(str).map(lambda value: value.replace("|", "\\|").replace("\n", " "))
    return "\n".join([
        "| " + " | ".join(shown.columns) + " |",
        "| " + " | ".join(["---"] * len(shown.columns)) + " |",
        *["| " + " | ".join(row) + " |" for row in shown.to_numpy().tolist()],
    ])


def run_smoke() -> None:
    metadata, canonical = load_canonical_metadata()
    summaries = []
    for seed in SEEDS:
        frames, _ = make_split(metadata, seed)
        audit = audit_split(seed, frames)
        summaries.append((seed, *(len(frames[s]) for s in ["train", "validation", "test"]), audit["all_checks_pass"]))
    # Verify both exact canonical feature implementations on one real clip.
    mfcc = mfcc_statistics(metadata.iloc[0]["file_path"])
    smile = opensmile_features(metadata.iloc[0]["file_path"])
    if mfcc.shape != (117,) or not np.isfinite(mfcc).all() or not np.isfinite(smile).all():
        raise RuntimeError("Feature smoke test failed")
    seed42, _ = make_split(metadata, 42)
    differences = sum(len(set(seed42[s]["file_path"]) ^ canonical[s]) for s in canonical)
    if differences:
        raise RuntimeError(f"Seed 42 does not reproduce canonical membership: {differences} differences")
    print(pd.DataFrame(summaries, columns=["seed", "train", "validation", "test", "audit_pass"]).to_string(index=False))
    print(f"SMOKE PASS: seed42 membership exact; MFCC={mfcc.shape}; openSMILE={smile.shape}")


def run_experiment() -> None:
    final_path = EXPERIMENT_DIR / "seed_results.csv"
    if final_path.exists():
        raise RuntimeError("Results already exist; refusing to rerun predefined test evaluations")
    metadata, canonical_memberships = load_canonical_metadata()
    config = {
        "experiment": "Improvement 5 - Group-Safe Split Variability Analysis",
        "status": "running",
        "dataset_rows": EXPECTED_TOTAL,
        "predefined_split_seeds": SEEDS,
        "canonical_seed": 42,
        "split_method": "two-stage GroupShuffleSplit; best of 300 attempts by canonical distribution/size/minority score",
        "target_split_proportions": {"train": 0.68, "validation": 0.12, "test": 0.20},
        "model_random_state": MODEL_RANDOM_STATE,
        "models": {"required": list(MODEL_NAMES.values()), "optional_random_forest_run": False},
        "fixed_hyperparameters": {"opensmile_xgboost": XGB_CONFIG, "mfcc_svm": SVM_CONFIG},
        "feature_policy": {
            "sample_rate_hz": TARGET_SR,
            "mono": True,
            "peak_normalization": True,
            "experimental_enhancement": False,
            "mfcc": "13 coefficients x 9 canonical statistics; correlation threshold 0.95 learned per-seed training split only",
            "opensmile": "eGeMAPSv02 LowLevelDescriptors x 7 canonical statistics",
        },
        "selection_policy": "No per-seed tuning. Configurations frozen from canonical experiment before any regenerated test evaluation.",
        "test_evaluation_policy": "One evaluation for each predefined seed/model pair; no seed is selected.",
        "aggregate_standard_deviation": "sample standard deviation (pandas ddof=1)",
        "software": {
            "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "librosa": librosa.__version__,
            "opensmile": opensmile.__version__, "xgboost": xgboost.__version__,
        },
        "canonical_manifest_sha256": {
            split: sha256_file(FROZEN_OUTPUT / f"{split}_manifest.csv")
            for split in ["train", "validation", "test"]
        },
    }
    atomic_json(EXPERIMENT_DIR / "experiment_config.json", config)

    mfcc_all = get_feature_matrix(metadata, "mfcc")
    smile_all = get_feature_matrix(metadata, "opensmile")
    if mfcc_all.shape != (EXPECTED_TOTAL, 117) or len(smile_all) != EXPECTED_TOTAL:
        raise RuntimeError("Feature matrix dimensions are inconsistent")

    seed_results = []
    classwise = []
    split_summaries = []
    leakage_audits = []
    assignments = []

    for seed in SEEDS:
        print(f"\n=== Split seed {seed} ===", flush=True)
        frames, details = make_split(metadata, seed)
        audit = audit_split(seed, frames)
        leakage_audits.append({**audit, **details})
        for split, frame in frames.items():
            counts = frame["emotion"].value_counts().reindex(EMOTIONS, fill_value=0)
            split_summaries.append({
                "seed": seed, "split": split, "rows": len(frame),
                "percentage": len(frame) / EXPECTED_TOTAL,
                "source_groups": frame["source_group"].nunique(),
                **{f"count_{emotion}": int(counts[emotion]) for emotion in EMOTIONS},
            })
            assignments.append(frame[["file_path", "filename", "sha256", "emotion", "label_id", "source_media", "source_group"]].assign(seed=seed, generated_split=split))

        # Dataset row indices are stable because frames retain the canonical sorted index.
        index_lookup = pd.Series(np.arange(len(metadata)), index=metadata["file_path"])
        indices = {
            split: index_lookup.loc[frame["file_path"]].to_numpy(int) for split, frame in frames.items()
        }
        y = metadata["label_id"].to_numpy(int)

        retained = retained_mfcc_indices(mfcc_all[indices["train"]])
        svm = Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(**SVM_CONFIG)),
        ])
        svm.fit(mfcc_all[indices["train"]][:, retained], y[indices["train"]])
        svm_pred = svm.predict(mfcc_all[indices["test"]][:, retained]).astype(int)
        seed_results.append(metric_row(seed, "mfcc_svm", y[indices["test"]], svm_pred, len(retained)))
        classwise.extend(classwise_rows(seed, "mfcc_svm", y[indices["test"]], svm_pred))

        xgb = XGBClassifier(
            objective="multi:softprob", num_class=len(EMOTIONS), eval_metric="mlogloss",
            random_state=MODEL_RANDOM_STATE, n_jobs=N_JOBS, **XGB_CONFIG,
        )
        xgb.fit(smile_all[indices["train"]], y[indices["train"]])
        xgb_pred = xgb.predict(smile_all[indices["test"]]).astype(int)
        seed_results.append(metric_row(seed, "opensmile_xgboost", y[indices["test"]], xgb_pred, None))
        classwise.extend(classwise_rows(seed, "opensmile_xgboost", y[indices["test"]], xgb_pred))
        print(pd.DataFrame(seed_results[-2:])[["model", "accuracy", "macro_f1"]].to_string(index=False), flush=True)

    seed_df = pd.DataFrame(seed_results).sort_values(["model", "seed"]).reset_index(drop=True)
    classwise_df = pd.DataFrame(classwise).sort_values(["model", "emotion", "seed"]).reset_index(drop=True)
    aggregate = seed_df.groupby("model", as_index=False).agg(
        seeds=("seed", "count"), mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"),
        mean_macro_precision=("macro_precision", "mean"), std_macro_precision=("macro_precision", "std"),
        mean_macro_recall=("macro_recall", "mean"), std_macro_recall=("macro_recall", "std"),
        mean_macro_f1=("macro_f1", "mean"), std_macro_f1=("macro_f1", "std"),
        min_macro_f1=("macro_f1", "min"), max_macro_f1=("macro_f1", "max"),
    )
    class_aggregate = classwise_df.groupby(["model", "emotion"], as_index=False).agg(
        seeds=("seed", "count"), mean_f1=("f1", "mean"), std_f1=("f1", "std"),
        min_f1=("f1", "min"), max_f1=("f1", "max"), mean_support=("support", "mean"),
    )

    pd.DataFrame(split_summaries).to_csv(EXPERIMENT_DIR / "split_summary.csv", index=False, encoding="utf-8-sig")
    seed_df.to_csv(final_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(EXPERIMENT_DIR / "aggregate_results.csv", index=False, encoding="utf-8-sig")
    classwise_df.to_csv(EXPERIMENT_DIR / "classwise_seed_results.csv", index=False, encoding="utf-8-sig")
    class_aggregate.to_csv(EXPERIMENT_DIR / "classwise_aggregate_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(leakage_audits).to_csv(EXPERIMENT_DIR / "source_group_leakage_audit.csv", index=False, encoding="utf-8-sig")
    pd.concat(assignments, ignore_index=True).to_csv(EXPERIMENT_DIR / "split_assignments.csv", index=False, encoding="utf-8-sig")
    seed_df.pivot(index="seed", columns="model", values="macro_f1").reset_index().to_csv(
        EXPERIMENT_DIR / "macro_f1_by_seed.csv", index=False, encoding="utf-8-sig"
    )

    seed42_frames, _ = make_split(metadata, 42)
    seed42_differences = {
        split: len(set(seed42_frames[split]["file_path"]) ^ canonical_memberships[split])
        for split in canonical_memberships
    }
    config.update({
        "status": "complete", "seed42_membership_symmetric_differences": seed42_differences,
        "test_evaluations_per_seed_model": 1, "all_predefined_seeds_reported": True,
        "zero_source_group_leakage_all_seeds": bool(pd.DataFrame(leakage_audits)["all_checks_pass"].all()),
    })
    atomic_json(EXPERIMENT_DIR / "experiment_config.json", config)

    most_unstable = class_aggregate.sort_values("std_f1", ascending=False).head(5)
    readme = [
        "# Improvement 5: Group-Safe Split Variability Analysis", "",
        "This isolated robustness experiment evaluates five predefined source-media/episode-group-disjoint partitions. It does not select a best seed and does not replace the canonical seed-42 result.", "",
        "Both model configurations were frozen before regenerated test evaluation. MFCC correlation filtering was learned from each seed's training split only. Test labels were used only for final scoring.", "",
        "## Per-seed results", "", markdown_table(seed_df), "",
        "## Aggregate results", "", markdown_table(aggregate), "",
        "## Most variable model/class combinations", "", markdown_table(most_unstable), "",
        "Every seed passed the 1,969-row uniqueness, SHA-256 uniqueness, all-seven-class, and zero source-group-overlap checks. See the CSV artifacts for full split and class-wise evidence.", "",
    ]
    (EXPERIMENT_DIR / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print("\nAGGREGATE RESULTS")
    print(aggregate.to_string(index=False))
    print("Improvement 5 complete. Improvement 6 was not started.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "run"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Improvement 5 phase={args.phase}; root={ROOT}", flush=True)
    if args.phase == "smoke":
        run_smoke()
    else:
        run_experiment()


if __name__ == "__main__":
    main()
