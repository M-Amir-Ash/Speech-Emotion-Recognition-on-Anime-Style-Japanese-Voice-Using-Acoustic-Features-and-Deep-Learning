#!/usr/bin/env python
"""One-time frozen Phase 3D-B2 Stage 3 held-out test evaluation.

No fitting, tuning, model selection, threshold changes, or feature extraction.
The selector is label-conditioned by construction and is reported as such.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
S1 = HERE.parent / "stage1_v001"
S2 = HERE.parent / "stage2_validation_v001"
PARTITION = ROOT / "Anime_SER_Experiment_V2" / "phase3d_leakage_aware" / "group_first_partition_manifest.csv"
A_ROOT = ROOT / "Anime_SER_Experiment_V2" / "phase3d_leakage_aware" / "full_pool_features"
B1_ROOT = ROOT / "Anime_SER_Experiment_V2" / "phase3d_b1_selector_features" / "code_faithful_reconstruction"

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant_Surprise", "Sad"]
SEED = 20261014
BOOTSTRAPS = 2000
EXPECTED = {
    "partition": (PARTITION, "9e06f55cc7dedbbfa27750059c3f0449b0e3aad106380d5d119675d356376c76"),
    "phase3d_a_manifest": (A_ROOT / "feature_manifest.csv", "a9c05c6b95b53b0bd7f30d682175be19f9c2f361cd22813b66dceb39cbc165ef"),
    "phase3d_b1_manifest": (B1_ROOT / "feature_manifest.csv", "ded54233aed71606c95bc168d45bb5ec46267538f1e1de4ba8e63cd033689b2a"),
    "stage1_selector_config": (S1 / "train_only_selector_config.json", "3faec2222274c23b51c0f03365f1eee36c55e4e985e060ac339627609a39844e"),
    "stage1_selector_bundle": (S1 / "train_only_selector_bundle.joblib", "d102252d77bc70af1171274367fecb0547955c731f2ccfc443df67af1f64f707"),
    "stage2_svm_bundle": (S2 / "models" / "mfcc_svm.joblib", "5107227b7bb2e836a855ba1206bdc12bf0ef7fdc94a75dc9b072b7aa3115b283"),
    "stage2_xgb_bundle": (S2 / "models" / "egemaps_xgboost.joblib", "e42731660b4c6fdedf4d9f74c904b0cbf591b6a27be697912a4b170d9cc00119"),
    "stage2_predictions": (S2 / "validation_predictions.csv", "db105b3f1c076f27b375bb28e40a2e559e5838897478007a8ba0b8a70114837a"),
    "stage2_search_log": (S2 / "search_log.csv", "e9e871a19778369039a803c2863c3407670b0836d78b7f4fa127e0140e821e94"),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def preflight() -> tuple[pd.DataFrame, dict, dict, dict, list[dict]]:
    frozen_rows = []
    for name, (path, expected) in EXPECTED.items():
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(f"STOP: frozen hash mismatch for {name}: {actual} != {expected}")
        frozen_rows.append({"artifact": name, "path": str(path), "sha256": actual, "bytes": path.stat().st_size, "status": "VERIFIED"})

    stage2_summary = json.loads((S2 / "run_summary.json").read_text(encoding="utf-8"))
    stage2_integrity = json.loads((S2 / "integrity_validation.json").read_text(encoding="utf-8"))
    if stage2_summary.get("test_predictions_computed") is not False or stage2_integrity.get("test_predictions_computed") is not False:
        raise RuntimeError("STOP: Stage 2 does not certify zero test predictions")
    if stage2_summary.get("scope") != "validation_only" or stage2_integrity.get("prediction_splits_are_validation_only") is not True:
        raise RuntimeError("STOP: Stage 2 scope certification mismatch")

    search = pd.read_csv(S2 / "search_log.csv")
    selected = search[search.selected.astype(str).str.lower().eq("true")].set_index("model")
    if json.loads(selected.loc["mfcc_svm", "parameters_json"]) != {"C": 1.0, "gamma": "scale", "kernel": "linear"}:
        raise RuntimeError("STOP: selected SVM configuration mismatch")
    if json.loads(selected.loc["egemaps_xgboost", "parameters_json"]) != {"learning_rate": 0.05, "max_depth": 3, "n_estimators": 300}:
        raise RuntimeError("STOP: selected XGBoost configuration mismatch")

    selector = joblib.load(S1 / "train_only_selector_bundle.joblib")
    svm = joblib.load(S2 / "models" / "mfcc_svm.joblib")
    xgb = joblib.load(S2 / "models" / "egemaps_xgboost.joblib")
    svm_params = svm["model"].named_steps["classifier"].get_params()
    xgb_params = xgb["model"].named_steps["classifier"].get_params()
    if not (svm_params["kernel"] == "linear" and svm_params["C"] == 1.0):
        raise RuntimeError("STOP: frozen SVM internal parameters mismatch")
    if not (xgb_params["n_estimators"] == 300 and xgb_params["max_depth"] == 3 and xgb_params["learning_rate"] == 0.05):
        raise RuntimeError("STOP: frozen XGBoost internal parameters mismatch")
    if len(svm["training_rows"]) != 1549 or len(xgb["training_rows"]) != 1549 or svm["training_rows"] != xgb["training_rows"]:
        raise RuntimeError("STOP: model training identity mismatch")

    frame = pd.read_csv(PARTITION, low_memory=False)
    if len(frame) != 66084 or frame.partition_row.duplicated().any() or frame.clip_id.duplicated().any() or frame.sha256.duplicated().any():
        raise RuntimeError("STOP: frozen partition grain failure")
    if frame.phase3d_split.value_counts().to_dict() != {"train": 44628, "test": 13462, "validation": 7994}:
        raise RuntimeError("STOP: partition counts mismatch")
    group_sets = {split: set(frame.loc[frame.phase3d_split.eq(split), "source_group"].astype(str)) for split in ["train", "validation", "test"]}
    overlaps = {"train_validation": len(group_sets["train"] & group_sets["validation"]),
                "train_test": len(group_sets["train"] & group_sets["test"]),
                "validation_test": len(group_sets["validation"] & group_sets["test"])}
    if any(overlaps.values()):
        raise RuntimeError(f"STOP: source-group overlap: {overlaps}")
    test = frame[frame.phase3d_split.eq("test")].sort_values("partition_row").reset_index(drop=True)
    if set(test.target_emotion) != set(CLASSES):
        raise RuntimeError("STOP: test taxonomy mismatch")

    preprocessing = {}
    for name, bundle in [("mfcc_svm", svm), ("egemaps_xgboost", xgb)]:
        scaler = bundle["model"].named_steps["scale"]
        payload = {"mean": scaler.mean_.tolist(), "var": scaler.var_.tolist(), "scale": scaler.scale_.tolist(),
                   "n_samples_seen": int(scaler.n_samples_seen_)}
        preprocessing[name] = {"sha256": canonical_hash(payload), "dimensions": len(scaler.mean_), "n_samples_seen": int(scaler.n_samples_seen_)}
    audit = {"stage2_scope_validation_only": True, "stage2_test_predictions": 0, "source_group_overlaps": overlaps,
             "selector_label_conditioned": True, "selector_dependency": "target_emotion chooses own centroid and class thresholds",
             "model_preprocessing": preprocessing}
    return test, selector, svm, xgb, frozen_rows, audit


def load_features(root: Path, test_rows: set[int], keys: tuple[str, ...]) -> dict[str, dict[int, np.ndarray]]:
    result = {key: {} for key in keys}
    for meta_path in sorted((root / "shards").glob("shard_*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        npz = meta_path.with_suffix(".npz")
        if digest(npz) != meta["artifact_sha256"]:
            raise RuntimeError(f"STOP: shard hash mismatch: {npz}")
        with np.load(npz, allow_pickle=False) as data:
            rows = data["partition_row"].astype(int)
            keep = np.fromiter((int(row) in test_rows for row in rows), dtype=bool, count=len(rows))
            for key in keys:
                for row, vector in zip(rows[keep], data[key][keep]):
                    result[key][int(row)] = np.asarray(vector, dtype=np.float64)
    for key in keys:
        if set(result[key]) != test_rows:
            raise RuntimeError(f"STOP: incomplete test feature join for {key}: {len(result[key])}/{len(test_rows)}")
    return result


def apply_label_conditioned_selector(test: pd.DataFrame, features: dict[int, np.ndarray], selector: dict) -> np.ndarray:
    matrix = np.stack([features[int(row)] for row in test.partition_row])
    if matrix.shape != (13462, 16) or not np.isfinite(matrix).all():
        raise RuntimeError("STOP: invalid test selector matrix")
    scaled = selector["scaler"].transform(matrix)
    passed = []
    for vector, inherited_label in zip(scaled, test.target_emotion):
        # LABEL-CONDITIONED: inherited_label chooses both own centroid and thresholds.
        own = float(np.linalg.norm(vector - selector["centroids"][inherited_label]))
        nearest = min(float(np.linalg.norm(vector - center)) for other, center in selector["centroids"].items() if other != inherited_label)
        margin = nearest - own
        threshold = selector["thresholds"][inherited_label]
        passed.append(nearest >= own and margin >= 0 and own <= threshold["clean_own"] and margin >= threshold["clean_margin"])
    return np.asarray(passed, dtype=bool)


def predict(bundle: dict, matrix: np.ndarray) -> np.ndarray:
    values = bundle["model"].predict(matrix)
    encoder = bundle.get("label_encoder")
    return values.astype(str) if encoder is None else encoder.inverse_transform(values.astype(int))


def metric_pair(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    return (float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
            float(f1_score(y, pred, labels=CLASSES, average="weighted", zero_division=0)),
            float(accuracy_score(y, pred)))


def bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for (model, cohort), part in predictions.groupby(["model", "evaluation_cohort"], sort=True):
        part = part.reset_index(drop=True)  # duplicate-index-safe implementation
        groups = part.source_group.unique()
        positions = {group: np.flatnonzero(part.source_group.eq(group).to_numpy()) for group in groups}
        draws = {"macro_f1": [], "weighted_f1": [], "accuracy": []}
        class_draws = {label: [] for label in CLASSES}
        for _ in range(BOOTSTRAPS):
            selected_groups = rng.choice(groups, size=len(groups), replace=True)
            sampled = part.iloc[np.concatenate([positions[group] for group in selected_groups])]
            macro, weighted, accuracy = metric_pair(sampled.true_label.to_numpy(), sampled.predicted_label.to_numpy())
            draws["macro_f1"].append(macro); draws["weighted_f1"].append(weighted); draws["accuracy"].append(accuracy)
            fs = f1_score(sampled.true_label, sampled.predicted_label, labels=CLASSES, average=None, zero_division=0)
            for label, value in zip(CLASSES, fs):
                class_draws[label].append(value)
        points = dict(zip(["macro_f1", "weighted_f1", "accuracy"], metric_pair(part.true_label.to_numpy(), part.predicted_label.to_numpy())))
        for metric, values in draws.items():
            rows.append({"model": model, "evaluation_cohort": cohort, "metric": metric, "pseudo_class": "OVERALL",
                         "point_estimate": points[metric], "ci_lower": np.percentile(values, 2.5), "ci_upper": np.percentile(values, 97.5),
                         "method": f"source_group_cluster_bootstrap_{BOOTSTRAPS}_replicates_seed_{SEED}", "support": len(part),
                         "source_groups": len(groups), "support_warning": ""})
        fs = f1_score(part.true_label, part.predicted_label, labels=CLASSES, average=None, zero_division=0)
        supports = part.true_label.value_counts()
        for label, point in zip(CLASSES, fs):
            n = int(supports.get(label, 0))
            rows.append({"model": model, "evaluation_cohort": cohort, "metric": "class_f1", "pseudo_class": label,
                         "point_estimate": point, "ci_lower": np.percentile(class_draws[label], 2.5),
                         "ci_upper": np.percentile(class_draws[label], 97.5),
                         "method": f"source_group_cluster_bootstrap_{BOOTSTRAPS}_replicates_seed_{SEED}", "support": n,
                         "source_groups": len(groups), "support_warning": "LOW_SUPPORT_LT_30" if n < 30 else ""})
    return pd.DataFrame(rows)


def reports(metrics: pd.DataFrame, classwise: pd.DataFrame, coverage: pd.DataFrame, intervals: pd.DataFrame, audit: dict) -> None:
    def row(model: str, cohort: str) -> pd.Series:
        return metrics[(metrics.model == model) & (metrics.evaluation_cohort == cohort)].iloc[0]
    s_sel = row("mfcc_svm", "label_conditioned_selector_passing_test")
    x_sel = row("egemaps_xgboost", "label_conditioned_selector_passing_test")
    s_all = row("mfcc_svm", "full_test_population")
    x_all = row("egemaps_xgboost", "full_test_population")
    overall_cov = coverage[coverage.pseudo_class.eq("OVERALL")].iloc[0]
    low = classwise[(classwise.evaluation_cohort.eq("label_conditioned_selector_passing_test")) & (classwise.support < 30)]
    low_text = ", ".join(f"{r.pseudo_class} n={int(r.support)} ({r.model})" for r in low.itertuples()) or "None"
    method = f"""# Methodology and limitations

This was the single authorized frozen held-out test evaluation. Input hashes, Stage 1 selector, Stage 2 model bundles, embedded preprocessing, and selected configurations were verified before any test scoring. No fitting, model selection, threshold change, or feature extraction occurred. The frozen linear SVM uses `C=1.0`; the frozen XGBoost model uses 300 trees, depth 3, and learning rate 0.05.

## Label-conditioned selector

The selector is **LABEL-CONDITIONED**. For each test clip, its inherited pseudo-emotion label selects the corresponding class centroid and class-specific distance/margin thresholds. Therefore, the {int(overall_cov.selected_count):,}-clip passing cohort is not produced by a deployable label-free filter. It is a retrospective, label-conditioned subset used to quantify sensitivity to the historical acoustic-selection procedure. Test labels were not used to alter the selector, models, or method.

The second evaluation includes all 13,462 quality-pass test clips. These predictions come from models trained only on the 1,549 selector-passing training clips, so the full-pool result is an out-of-selection-distribution stress test, not performance from full-pool training.

Uncertainty uses a corrected duplicate-index-safe source-group cluster bootstrap with 2,000 replicates and seed {SEED}. Source-group separation is not speaker-disjointness. Targets remain inherited pseudo-labels, not independent human ground truth. The original selector's MFCC-9 statistics were reconstructed code-faithfully but could not be historically numerically verified. Upstream pseudo-label construction may preserve global dependencies across partitions.
"""
    historical = f"""# Historical comparison

The historical eGeMAPS + XGBoost macro-F1 of 0.7517 and MFCC + SVM macro-F1 of 0.7474 came from the original 1,969-clip, globally selected protocol. The frozen train-only-selector test results are XGBoost {x_sel.macro_f1:.4f} and SVM {s_sel.macro_f1:.4f} on a different, label-conditioned passing cohort. Full-test stress results are XGBoost {x_all.macro_f1:.4f} and SVM {s_all.macro_f1:.4f}.

These are descriptive cross-protocol values, not paired improvements or decreases. Population construction, selector fitting, cohort size, and evaluation protocol differ. None of the values measure independent human-label validity.
"""
    summary = f"""# Phase 3D-B2 Stage 3 final test summary

## Manuscript-ready factual findings

The frozen training-only acoustic selector retained {int(overall_cov.selected_count):,} of 13,462 held-out quality-valid clips ({overall_cov.selection_coverage:.2%}); {int(overall_cov.abstained_count):,} clips were outside the label-conditioned passing subset. On this subset, MFCC-117 + linear SVM achieved macro-F1 {s_sel.macro_f1:.4f}, weighted F1 {s_sel.weighted_f1:.4f}, and accuracy {s_sel.accuracy:.4f}; eGeMAPS-175 + XGBoost achieved macro-F1 {x_sel.macro_f1:.4f}, weighted F1 {x_sel.weighted_f1:.4f}, and accuracy {x_sel.accuracy:.4f}. Across all 13,462 held-out clips, the corresponding macro-F1 values were {s_all.macro_f1:.4f} and {x_all.macro_f1:.4f}.

The association between selector passage and higher pseudo-label agreement is large. It supports the conclusion that reported performance is strongly conditioned on acoustic sample selection. It does not by itself establish that selection caused the performance difference: passing and nonpassing clips differ systematically, and labels are inherited pseudo-labels. Full-pool results specifically reflect models trained only on the selected 1,549 training clips.

Low-support warnings: {low_text}.

## Evidence-versus-claims matrix

| Claim | Evidence | Supported wording | Unsupported wording |
|---|---|---|---|
| Frozen evaluation integrity | Verified hashes, unchanged model parameters, zero source-group overlap | One-time evaluation of frozen artifacts | Independently replicated pipeline |
| Selective performance | Label-conditioned passing-test metrics and denominators | Agreement with inherited pseudo-labels on a label-conditioned subset | Deployable label-free filtering performance |
| Full-pool performance | Predictions for all 13,462 test clips | Out-of-selection-distribution stress test | Model trained on the full candidate population |
| Selection conditioning | Large observed selective/full-pool metric difference | Performance is strongly associated with selector passage | Acoustic selection causally explains the entire difference |
| Emotion validity | No independent human truth in this stage | Evaluation targets are inherited pseudo-labels | Human-validated emotion recognition accuracy |
| Generalization | Train/validation/test source groups are disjoint | Source-group-disjoint evaluation | Speaker-independent generalization |
| Historical comparison | Historical 0.7517/0.7474 under another protocol | Descriptive contextual comparison only | Direct paired improvement or decrease |

## Terminal status

The authorized one-time test evaluation is complete. No model or threshold was changed after viewing test outcomes. Further testing or tuning requires separate authorization.
"""
    atomic_text(HERE / "methodology_and_limitations.md", method)
    atomic_text(HERE / "historical_comparison.md", historical)
    atomic_text(HERE / "stage3_summary.md", summary)


def main() -> None:
    started = time.perf_counter()
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "confusion_matrices").mkdir(exist_ok=True)
    test, selector, svm, xgb, frozen_rows, audit = preflight()
    test_rows = set(test.partition_row.astype(int))
    b1 = load_features(B1_ROOT, test_rows, ("features",))["features"]
    a = load_features(A_ROOT, test_rows, ("mfcc", "egemaps"))
    test["selector_pass"] = apply_label_conditioned_selector(test, b1, selector)

    coverage_rows = []
    for label in [*CLASSES, "OVERALL"]:
        part = test if label == "OVERALL" else test[test.target_emotion.eq(label)]
        chosen = part[part.selector_pass]
        coverage_rows.append({"pseudo_class": label, "quality_pass_denominator": len(part), "selected_count": len(chosen),
                              "abstained_count": len(part) - len(chosen), "selection_coverage": len(chosen) / len(part),
                              "denominator_source_groups": part.source_group.nunique(), "selected_source_groups": chosen.source_group.nunique(),
                              "selector_conditioning": "LABEL_CONDITIONED_BY_INHERITED_PSEUDO_LABEL"})
    coverage = pd.DataFrame(coverage_rows)

    ids = test.partition_row.astype(int).tolist()
    mfcc = np.stack([a["mfcc"][row] for row in ids])
    egemaps = np.stack([a["egemaps"][row] for row in ids])
    if mfcc.shape != (13462, 117) or egemaps.shape != (13462, 175) or not np.isfinite(mfcc).all() or not np.isfinite(egemaps).all():
        raise RuntimeError("STOP: invalid test classifier features")
    model_predictions = {"mfcc_svm": predict(svm, mfcc), "egemaps_xgboost": predict(xgb, egemaps)}
    prediction_parts = []
    for model, values in model_predictions.items():
        for cohort, mask in [("label_conditioned_selector_passing_test", test.selector_pass.to_numpy()),
                             ("full_test_population", np.ones(len(test), dtype=bool))]:
            part = test.loc[mask, ["partition_row", "clip_id", "sha256", "source_group", "target_emotion", "selector_pass"]].copy()
            part.insert(0, "model", model); part.insert(1, "evaluation_cohort", cohort)
            part = part.rename(columns={"target_emotion": "true_label"})
            part["predicted_label"] = values[mask]
            prediction_parts.append(part)
    predictions = pd.concat(prediction_parts, ignore_index=True)

    metric_rows, class_rows = [], []
    for (model, cohort), part in predictions.groupby(["model", "evaluation_cohort"], sort=True):
        macro, weighted, accuracy = metric_pair(part.true_label.to_numpy(), part.predicted_label.to_numpy())
        metric_rows.append({"model": model, "evaluation_cohort": cohort, "support": len(part), "test_denominator": 13462,
                            "coverage": len(part) / 13462, "macro_f1": macro, "weighted_f1": weighted, "accuracy": accuracy,
                            "label_basis": "inherited_pseudo_labels"})
        precision, recall, fs, support = precision_recall_fscore_support(part.true_label, part.predicted_label, labels=CLASSES, zero_division=0)
        for label, p, r, f, n in zip(CLASSES, precision, recall, fs, support):
            class_rows.append({"model": model, "evaluation_cohort": cohort, "pseudo_class": label, "support": int(n),
                               "precision": p, "recall": r, "f1": f, "support_warning": "LOW_SUPPORT_LT_30" if n < 30 else ""})
        cm = pd.DataFrame(confusion_matrix(part.true_label, part.predicted_label, labels=CLASSES), index=CLASSES, columns=CLASSES)
        cm.index.name = "true_pseudo_label"
        cm.reset_index().to_csv(HERE / "confusion_matrices" / f"{model}__{cohort}.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    classwise = pd.DataFrame(class_rows)
    intervals = bootstrap(predictions)

    atomic_csv(HERE / "test_coverage.csv", coverage)
    atomic_csv(HERE / "test_metrics.csv", metrics)
    atomic_csv(HERE / "classwise_test_metrics.csv", classwise)
    atomic_csv(HERE / "test_predictions.csv", predictions)
    atomic_csv(HERE / "bootstrap_intervals.csv", intervals)
    atomic_csv(HERE / "frozen_artifact_manifest.csv", pd.DataFrame(frozen_rows))
    audit.update({"status": "PASS", "test_rows": len(test), "test_predictions_per_model": 13462,
                  "selector_pass_rows": int(test.selector_pass.sum()), "selector_label_conditioned": True,
                  "bootstrap_duplicate_index_safe": True, "bootstrap_seed": SEED, "bootstrap_replicates": BOOTSTRAPS,
                  "no_refitting_or_tuning": True, "elapsed_sec": time.perf_counter() - started,
                  "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                               "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__, "joblib": joblib.__version__}})
    atomic_text(HERE / "integrity_validation.json", json.dumps(audit, indent=2, ensure_ascii=False))
    reports(metrics, classwise, coverage, intervals, audit)
    print(json.dumps(audit, indent=2))
    print(coverage.to_string(index=False))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
