import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cluster_clean_target_selector import (
    COLORS,
    EMOTIONS,
    base_feature_vector,
    copy_unique,
    ensure_dir,
    safe_name,
    to_float,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_ENCODING = "utf-8-sig"
REVIEW_STATUSES = {"needs_review", "needs_review_high_priority"}
REJECT_FOLDERS = {"Rejected", "Rejected_LowQuality"}
SCORE_KEYS = [f"final_score_{label.lower()}" for label in EMOTIONS]
PAPER_MFCC_FEATURE_MODES = {"mfcc", "mfcc_paper9", "mfcc_separable"}
ACOUSTIC_MANIFEST_KEYS = [
    "duration_sec",
    "voiced_ratio",
    "harmonic_ratio",
    "flatness_mean",
    "zcr_mean",
    "rms_mean",
    "burst_ratio",
]
AUDIO_ONLY_FEATURE_MODES = PAPER_MFCC_FEATURE_MODES | {"multi_feature"}
MINORITY_RESCUE_LABELS = {"Angry", "Disgust"}
CLUSTER_MIN_SAMPLES = 10
CLUSTER_MAX_CLUSTERS = len(EMOTIONS)
CLUSTER_OUTLIER_PERCENTILE = 95
REVIEW_CLASS_ORDER = {
    "Disgust": 0,
    "Pleasant_Surprise": 1,
    "Fear": 2,
    "Angry": 3,
    "Happy": 4,
    "Sad": 5,
    "Neutral": 6,
}


def read_csv_rows(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding=CSV_ENCODING, newline="") as f:
        return list(csv.DictReader(f))


def status_from_row(row: dict, workflow_type: str) -> str:
    if workflow_type == "character":
        status = str(row.get("confidence_status", "")).strip()
        return status or "reject"
    pred = str(row.get("pred_folder", "")).strip()
    if pred == "Needs_Review":
        return "needs_review"
    if pred in REJECT_FOLDERS:
        return "reject"
    if pred in EMOTIONS:
        return "auto_accept"
    return "reject"


def target_emotion(row: dict, workflow_type: str) -> str:
    for key in ("manual_label", "approved_label", "candidate_emotion", "top_label"):
        value = str(row.get(key, "")).strip()
        if value in EMOTIONS:
            return value
    pred = str(row.get("pred_folder", "")).strip()
    return pred if pred in EMOTIONS else ""


def audio_path_from_row(row: dict) -> Path:
    for key in ("output_file", "global_output_file", "candidate_output_file", "global_candidate_output_file", "all_top_label_output_file"):
        value = str(row.get(key, "")).strip()
        if value:
            path = Path(value)
            if path.is_file():
                return path
    return Path("")


def review_priority(label: str, top_score: float, top_gap: float, evidence_count: int) -> float:
    bonus = {
        "Disgust": 1.0,
        "Pleasant_Surprise": 1.0,
        "Fear": 1.0,
        "Angry": 0.6,
        "Happy": 0.4,
        "Sad": 0.4,
        "Neutral": 0.2,
    }.get(label, 0.0)
    return round(
        0.40 * max(0.0, top_score)
        + 0.20 * min(max(0.0, top_gap) / 0.30, 1.0)
        + 0.20 * min(max(0, evidence_count) / 3.0, 1.0)
        + 0.20 * bonus,
        6,
    )


def normalize_rows(input_emotion_dir: Path, workflow_type: str) -> List[dict]:
    manifest = input_emotion_dir / ("_character_emotion_manifest.csv" if workflow_type == "character" else "_multimodal_emotion_manifest.csv")
    rows = read_csv_rows(manifest)
    if not rows:
        rows = rows_from_folders(input_emotion_dir, workflow_type)
    out = []
    for idx, row in enumerate(rows):
        normalized = dict(row)
        label = target_emotion(normalized, workflow_type)
        status = status_from_row(normalized, workflow_type)
        audio_path = audio_path_from_row(normalized)
        if not label and status == "needs_review":
            label = str(normalized.get("top_label", "")).strip()
        if label not in EMOTIONS:
            label = ""
        normalized["clip_id"] = normalized.get("clip_id") or Path(str(normalized.get("source_file") or audio_path.name)).stem or f"clip_{idx:06d}"
        normalized["source_file"] = normalized.get("source_file") or audio_path.name
        normalized["output_file"] = str(audio_path) if audio_path else normalized.get("output_file", "")
        normalized["character_name"] = normalized.get("character_name") or normalized.get("character") or "GLOBAL_BASIC"
        normalized["speaker_cluster"] = normalized.get("speaker_cluster") or normalized.get("speaker") or "NO_SPEAKER_CLUSTER"
        normalized["confidence_status"] = status
        normalized["target_emotion"] = label
        normalized["candidate_emotion"] = normalized.get("candidate_emotion") or label
        normalized["audio_quality_status"] = normalized.get("audio_quality_status") or ("pass" if status != "reject" else "reject")
        normalized["evidence_count"] = int(to_float(normalized.get("evidence_count"), 1 if label else 0))
        normalized["evidence_sources"] = normalized.get("evidence_sources") or normalized.get("label_source") or normalized.get("reason") or "none"
        normalized["top_score"] = to_float(normalized.get("top_score"))
        normalized["top_gap"] = to_float(normalized.get("top_gap"))
        normalized["review_priority_score"] = to_float(normalized.get("review_priority_score"))
        if normalized["review_priority_score"] <= 0.0:
            normalized["review_priority_score"] = review_priority(label, normalized["top_score"], normalized["top_gap"], normalized["evidence_count"])
        out.append(normalized)
    return out


def rows_from_folders(input_emotion_dir: Path, workflow_type: str) -> List[dict]:
    rows = []
    for folder in EMOTIONS + ["Needs_Review", "Rejected", "Rejected_LowQuality"]:
        folder_path = input_emotion_dir / folder
        if not folder_path.is_dir():
            continue
        for wav in sorted(folder_path.glob("*.wav")):
            rows.append(
                {
                    "source_file": wav.name,
                    "output_file": str(wav),
                    "pred_folder": folder,
                    "top_label": folder if folder in EMOTIONS else "",
                    "character_name": "GLOBAL_BASIC" if workflow_type == "basic" else "",
                    "speaker_cluster": "NO_SPEAKER_CLUSTER" if workflow_type == "basic" else "",
                }
            )
    return rows


def is_review(row: dict) -> bool:
    return str(row.get("confidence_status", "")) in REVIEW_STATUSES


def write_count_consistency(input_dir: Path, rows: List[dict], workflow_type: str) -> int:
    dist = distribution_counts(rows)
    report_rows = []
    mismatches = 0
    labels = EMOTIONS + ["Needs_Review", "Rejected"]
    for label in labels:
        if workflow_type == "character" and (input_dir / "_global").is_dir():
            folder_base = input_dir / "_global"
        else:
            folder_base = input_dir
        folder_name = "Rejected_LowQuality" if workflow_type == "basic" and label == "Rejected" and (folder_base / "Rejected_LowQuality").is_dir() else label
        folder_count = len(list((folder_base / folder_name).glob("*.wav"))) if (folder_base / folder_name).is_dir() else 0
        manifest_count = dist[label]
        difference = abs(folder_count - manifest_count)
        status = "ok" if difference == 0 else "mismatch"
        mismatches += 0 if status == "ok" else 1
        report_rows.append(
            {
                "source": workflow_type,
                "emotion": label,
                "folder_count": folder_count,
                "manifest_count": manifest_count,
                "report_count": manifest_count,
                "status": status,
                "difference": difference,
            }
        )
    write_csv(input_dir / "_count_consistency_report.csv", report_rows, ["source", "emotion", "folder_count", "manifest_count", "report_count", "status", "difference"])
    return mismatches


def distribution_counts(rows: List[dict]) -> Dict[str, int]:
    counts = defaultdict(int)
    for row in rows:
        status = row.get("confidence_status")
        label = row.get("target_emotion")
        if status == "auto_accept" and label in EMOTIONS:
            counts[label] += 1
        elif is_review(row):
            counts["Needs_Review"] += 1
        elif status == "reject":
            counts["Rejected"] += 1
    return counts


def write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in rows])


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def selection_key(row: dict) -> tuple:
    label = row.get("candidate_emotion") or row.get("target_emotion")
    return (
        REVIEW_CLASS_ORDER.get(label, 99),
        -to_float(row.get("review_priority_score")),
        -to_float(row.get("evidence_count")),
        -to_float(row.get("top_score")),
        -to_float(row.get("top_gap")),
    )


def cluster_selection_key(row: dict) -> tuple:
    return (
        0 if row.get("separability_status") == "clean_separable" else 1,
        -to_float(row.get("separation_margin")),
        to_float(row.get("distance_to_own_centroid")),
        -to_float(row.get("review_priority_score")),
        -to_float(row.get("evidence_count")),
        -to_float(row.get("top_gap")),
    )


def write_review_shortlist(input_dir: Path, rows: List[dict], max_total: int, per_class: int) -> Tuple[int, float]:
    root = input_dir / "_review_shortlist"
    clean_dir(root)
    for label in EMOTIONS:
        ensure_dir(root / label)
    review_rows = [
        row for row in rows
        if is_review(row)
        and row.get("target_emotion") in EMOTIONS
        and row.get("audio_quality_status") == "pass"
        and audio_path_from_row(row).is_file()
    ]
    by_label = defaultdict(list)
    for row in review_rows:
        by_label[row["target_emotion"]].append(row)
    selected = []
    for label in sorted(EMOTIONS, key=lambda x: REVIEW_CLASS_ORDER.get(x, 99)):
        selected.extend(sorted(by_label[label], key=selection_key)[:per_class])
    selected = sorted(selected, key=selection_key)[:max_total]
    manifest = []
    for row in selected:
        label = row["target_emotion"]
        copied = copy_unique(audio_path_from_row(row), root / label)
        manifest.append(short_manifest_row(row, copied, label))
    fields = ["clip_id", "source_file", "output_file", "character_name", "speaker_cluster", "candidate_emotion", "final_top", "confidence_status", "review_priority_score", "evidence_count", "evidence_sources", "top_score", "top_gap", "transcript", "clip_type", "audio_quality_status"]
    write_csv(input_dir / "_review_shortlist_manifest.csv", manifest, fields)
    original = len(review_rows)
    reduction = 0.0 if original == 0 else 100.0 * (1.0 - len(manifest) / original)
    workload = []
    selected_counts = Counter(row["target_emotion"] for row in selected)
    original_counts = Counter(row["target_emotion"] for row in review_rows)
    for label in EMOTIONS:
        workload.append({"category": label, "original_count": original_counts[label], "shortlist_count": selected_counts[label], "reduced_by": original_counts[label] - selected_counts[label], "notes": ""})
    workload.append({"category": "TOTAL", "original_count": original, "shortlist_count": len(manifest), "reduced_by": original - len(manifest), "notes": f"reduction_percent={reduction:.1f}"})
    write_csv(input_dir / "_review_workload_report.csv", workload, ["category", "original_count", "shortlist_count", "reduced_by", "notes"])
    return len(manifest), reduction


def short_manifest_row(row: dict, copied: Path, label: str) -> dict:
    return {
        "clip_id": row.get("clip_id", ""),
        "source_file": row.get("source_file", ""),
        "output_file": str(copied),
        "character_name": row.get("character_name", "GLOBAL_BASIC"),
        "speaker_cluster": row.get("speaker_cluster", "NO_SPEAKER_CLUSTER"),
        "candidate_emotion": row.get("candidate_emotion") or label,
        "final_top": row.get("target_emotion") or row.get("top_label", ""),
        "confidence_status": row.get("confidence_status", ""),
        "review_priority_score": row.get("review_priority_score", ""),
        "evidence_count": row.get("evidence_count", ""),
        "evidence_sources": row.get("evidence_sources", ""),
        "top_score": row.get("top_score", ""),
        "top_gap": row.get("top_gap", ""),
        "transcript": row.get("transcript", ""),
        "clip_type": row.get("clip_type", ""),
        "audio_quality_status": row.get("audio_quality_status", ""),
    }


def write_balanced_target(input_dir: Path, rows: List[dict], target: int, max_total: int) -> int:
    root = input_dir / f"balanced_review_target_{target}"
    clean_dir(root)
    for label in EMOTIONS:
        ensure_dir(root / label)
    review_rows = [row for row in rows if is_review(row) and row.get("target_emotion") in EMOTIONS and audio_path_from_row(row).is_file()]
    selected = []
    by_label = defaultdict(list)
    for row in review_rows:
        by_label[row["target_emotion"]].append(row)
    for label in EMOTIONS:
        selected.extend(sorted(by_label[label], key=selection_key)[:target])
    selected = sorted(selected, key=selection_key)[:max_total]
    for row in selected:
        copy_unique(audio_path_from_row(row), root / row["target_emotion"])
    return len(selected)


def feature_vector(row: dict, cache: Dict[str, List[float]], feature_method: str = "mfcc") -> Tuple[List[float], bool]:
    path = audio_path_from_row(row)
    if not path.is_file():
        return [], False
    key = f"{feature_method}:{path.resolve()}"
    if key not in cache:
        try:
            cache[key] = base_feature_vector(path, feature_method)
        except Exception:
            cache[key] = [0.0] * (9 if feature_method == "multi_feature" else 88 if feature_method == "mfcc_rich" else 9 if feature_method in PAPER_MFCC_FEATURE_MODES else 13)
    features = list(cache[key])
    if feature_method == "multi_feature":
        manifest_acoustic = [to_float(row.get(key)) for key in ACOUSTIC_MANIFEST_KEYS]
        if not any(abs(value) > 1e-12 for value in manifest_acoustic):
            try:
                manifest_acoustic = base_feature_vector(path, "acoustic")[: len(ACOUSTIC_MANIFEST_KEYS)]
            except Exception:
                manifest_acoustic = [0.0] * len(ACOUSTIC_MANIFEST_KEYS)
        features.extend(manifest_acoustic)
    if feature_method not in AUDIO_ONLY_FEATURE_MODES:
        features.extend([to_float(row.get(key)) for key in SCORE_KEYS])
        features.extend([to_float(row.get("top_score")), to_float(row.get("top_gap")), to_float(row.get("evidence_count")), to_float(row.get("review_priority_score"))])
    return features, any(abs(x) > 1e-12 for x in features)


def write_cluster_clean(input_dir: Path, rows: List[dict], target: int, allow_overlap_fill: bool = False, allow_usable_fill: bool = False, feature_method: str = "mfcc") -> int:
    suffix = f"{target}_{feature_method}" if feature_method != "acoustic" else str(target)
    root = input_dir / f"cluster_clean_target_{suffix}"
    if root.exists():
        backup = root.with_name(f"{root.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.move(str(root), str(backup))
    for label in EMOTIONS:
        ensure_dir(root / label)
        ensure_dir(root / "_overlap_review" / label)
        ensure_dir(root / "_minority_rescue_review" / label)
    ensure_dir(root / "_hard_ambiguous")
    ensure_dir(root / "cluster_analysis")
    cache: Dict[str, List[float]] = {}
    usable, feats, hard = [], [], []
    for row in rows:
        if row.get("target_emotion") not in EMOTIONS or row.get("audio_quality_status") != "pass":
            hard.append(row)
            continue
        vec, ok = feature_vector(row, cache, feature_method)
        if not ok:
            hard.append(row)
            continue
        usable.append(row)
        feats.append(vec)
    if not usable:
        return 0
    arr = np.asarray(feats, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    scaled = StandardScaler().fit_transform(arr)
    if feature_method in {"mfcc", "mfcc_paper9"}:
        assign_kmeans_outlier_filter(usable, scaled)
    else:
        assign_separability(usable, scaled)
    selected, overlap = [], []
    for label in EMOTIONS:
        label_rows = [row for row in usable if row["target_emotion"] == label]
        pool = [row for row in label_rows if row["separability_status"] == "clean_separable"]
        if allow_usable_fill:
            pool.extend(row for row in label_rows if row["separability_status"] == "usable_review")
        if allow_overlap_fill:
            pool.extend(row for row in label_rows if row["separability_status"] == "overlap_review")
        chosen = sorted(pool, key=cluster_selection_key)[:target]
        chosen_ids = {id(row) for row in chosen}
        selected.extend(chosen)
        overlap.extend(row for row in label_rows if row["separability_status"] == "overlap_review" or id(row) not in chosen_ids)
    manifest = []
    rescue_manifest = []
    for row in selected:
        copied = copy_unique(audio_path_from_row(row), root / row["target_emotion"])
        manifest.append(cluster_manifest_row(row, copied, "selected"))
    selected_counts = Counter(row["target_emotion"] for row in selected)
    rescue_ids = set()
    for label in MINORITY_RESCUE_LABELS:
        shortage = max(0, target - selected_counts[label])
        if shortage <= 0:
            continue
        rescue_pool = [
            row for row in overlap
            if row.get("target_emotion") == label
            and row.get("audio_quality_status") == "pass"
            and to_float(row.get("separation_margin")) >= 0.0
        ]
        for row in sorted(rescue_pool, key=cluster_selection_key)[:shortage]:
            copied = copy_unique(audio_path_from_row(row), root / "_minority_rescue_review" / label)
            rescue_manifest.append(cluster_manifest_row(row, copied, "minority_rescue_review"))
            rescue_ids.add(id(row))
    for row in overlap:
        copied = copy_unique(audio_path_from_row(row), root / "_overlap_review" / row["target_emotion"])
        manifest.append(cluster_manifest_row(row, copied, "overlap_or_not_selected"))
    for row in hard:
        path = audio_path_from_row(row)
        if path.is_file():
            copy_unique(path, root / "_hard_ambiguous")
    write_csv(root / "_cluster_clean_manifest.csv", manifest, ["clip_id", "source_file", "copied_file", "target_emotion", "character_name", "speaker_cluster", "separability_status", "distance_to_own_centroid", "distance_to_nearest_other_centroid", "nearest_other_emotion", "separation_margin", "review_priority_score", "evidence_count", "top_score", "top_gap", "audio_quality_status", "transcript", "selection_reason"])
    write_csv(root / "_minority_rescue_manifest.csv", rescue_manifest, ["clip_id", "source_file", "copied_file", "target_emotion", "character_name", "speaker_cluster", "separability_status", "distance_to_own_centroid", "distance_to_nearest_other_centroid", "nearest_other_emotion", "separation_margin", "review_priority_score", "evidence_count", "top_score", "top_gap", "audio_quality_status", "transcript", "selection_reason"])
    write_cluster_reports_and_graphs(root, usable, selected, overlap, target, feature_method=feature_method)
    return len(selected)


def assign_kmeans_outlier_filter(rows: List[dict], scaled: np.ndarray) -> None:
    n_samples = len(rows)
    label_count = len({row.get("target_emotion") for row in rows if row.get("target_emotion") in EMOTIONS})
    if n_samples < CLUSTER_MIN_SAMPLES or label_count < 2:
        for row in rows:
            row["separability_status"] = "clean_separable"
            row["distance_to_own_centroid"] = 0.0
            row["distance_to_nearest_other_centroid"] = 0.0
            row["nearest_other_emotion"] = "kmeans_not_run"
            row["separation_margin"] = 0.0
        return
    n_clusters = max(2, min(CLUSTER_MAX_CLUSTERS, label_count, n_samples - 1))
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    cluster_ids = km.fit_predict(scaled)
    centers = km.cluster_centers_
    distances = np.linalg.norm(scaled - centers[cluster_ids], axis=1)
    keep = np.zeros(n_samples, dtype=bool)
    thresholds: Dict[int, float] = {}
    outlier_percentile = CLUSTER_OUTLIER_PERCENTILE
    for cluster_id in range(n_clusters):
        idx = np.where(cluster_ids == cluster_id)[0]
        if len(idx) == 0:
            continue
        threshold = float(np.percentile(distances[idx], outlier_percentile))
        thresholds[cluster_id] = threshold
        keep[idx] = distances[idx] <= threshold
    for label in EMOTIONS:
        idx = [i for i, row in enumerate(rows) if row.get("target_emotion") == label]
        if not idx:
            continue
        min_keep = min(len(idx), max(2, int(math.ceil(len(idx) * 0.60))))
        if sum(1 for i in idx if keep[i]) < min_keep:
            for i in sorted(idx, key=lambda j: float(distances[j]))[:min_keep]:
                keep[i] = True
    emotion_centroids: Dict[str, np.ndarray] = {}
    for label in EMOTIONS:
        idx = [i for i, row in enumerate(rows) if row.get("target_emotion") == label]
        if idx:
            emotion_centroids[label] = np.mean(scaled[idx], axis=0)
    own_dists: List[float] = [0.0] * n_samples
    nearest_dists: List[float] = [0.0] * n_samples
    nearest_labels: List[str] = [""] * n_samples
    for i, row in enumerate(rows):
        cluster_id = int(cluster_ids[i])
        own_label = row.get("target_emotion")
        own_dist = float(np.linalg.norm(scaled[i] - centers[cluster_id]))
        own_dists[i] = own_dist
        nearest_label = ""
        nearest_emotion_dist = float("inf")
        for other_label, centroid in emotion_centroids.items():
            if other_label == own_label:
                continue
            d = float(np.linalg.norm(scaled[i] - centroid))
            if d < nearest_emotion_dist:
                nearest_emotion_dist = d
                nearest_label = other_label
        if nearest_emotion_dist == float("inf"):
            nearest_emotion_dist = 0.0
        nearest_dists[i] = nearest_emotion_dist
        nearest_labels[i] = nearest_label or f"kmeans_cluster_{cluster_id}"
    by_label: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_label[row["target_emotion"]].append(i)
    thresholds_by_label: Dict[str, Dict[str, float]] = {}
    for label, idxs in by_label.items():
        own_values = [own_dists[i] for i in idxs]
        margin_values = [nearest_dists[i] - own_dists[i] for i in idxs]
        if label == "Neutral":
            thresholds_by_label[label] = {
                "clean_own": percentile(own_values, 15),
                "usable_own": percentile(own_values, 35),
                "usable_margin": percentile(margin_values, 78),
                "clean_margin": percentile(margin_values, 92),
            }
        else:
            thresholds_by_label[label] = {
                "clean_own": percentile(own_values, 25),
                "usable_own": percentile(own_values, 50),
                "usable_margin": percentile(margin_values, 65),
                "clean_margin": percentile(margin_values, 85),
            }
    for i, row in enumerate(rows):
        own_label = row.get("target_emotion")
        own_dist = own_dists[i]
        nearest_dist = nearest_dists[i]
        nearest_label = nearest_labels[i]
        separation_margin = nearest_dist - own_dist
        row["distance_to_own_centroid"] = round(float(own_dist), 6)
        row["distance_to_nearest_other_centroid"] = round(float(nearest_dist), 6)
        row["nearest_other_emotion"] = nearest_label
        row["separation_margin"] = round(float(separation_margin), 6)
        th = thresholds_by_label.get(own_label, {})
        if row.get("audio_quality_status") != "pass":
            row["separability_status"] = "hard_ambiguous"
        elif nearest_dist < own_dist or separation_margin < 0:
            row["separability_status"] = "overlap_review"
        elif keep[i] and own_dist <= th.get("clean_own", float("inf")) and separation_margin >= th.get("clean_margin", -float("inf")):
            row["separability_status"] = "clean_separable"
        elif keep[i] and own_dist <= th.get("usable_own", float("inf")) and separation_margin >= th.get("usable_margin", -float("inf")):
            row["separability_status"] = "usable_review"
        else:
            row["separability_status"] = "overlap_review"


def assign_separability(rows: List[dict], scaled: np.ndarray) -> None:
    by_label = defaultdict(list)
    for i, row in enumerate(rows):
        by_label[row["target_emotion"]].append(i)
    centroids = {label: np.mean(scaled[idxs], axis=0) for label, idxs in by_label.items() if idxs}
    for i, row in enumerate(rows):
        label = row["target_emotion"]
        own = float(np.linalg.norm(scaled[i] - centroids[label]))
        nearest_label, nearest_dist = "", float("inf")
        for other, centroid in centroids.items():
            if other == label:
                continue
            dist = float(np.linalg.norm(scaled[i] - centroid))
            if dist < nearest_dist:
                nearest_label, nearest_dist = other, dist
        row["distance_to_own_centroid"] = own
        row["distance_to_nearest_other_centroid"] = 0.0 if nearest_dist == float("inf") else nearest_dist
        row["nearest_other_emotion"] = nearest_label
        row["separation_margin"] = row["distance_to_nearest_other_centroid"] - own
    for label, idxs in by_label.items():
        own_values = [rows[i]["distance_to_own_centroid"] for i in idxs]
        margin_values = [rows[i]["separation_margin"] for i in idxs]
        if label == "Neutral":
            clean_own, usable_own = np.percentile(own_values, 15), np.percentile(own_values, 35)
            clean_margin, usable_margin = np.percentile(margin_values, 92), np.percentile(margin_values, 78)
        else:
            clean_own, usable_own = np.percentile(own_values, 25), np.percentile(own_values, 50)
            clean_margin, usable_margin = np.percentile(margin_values, 85), np.percentile(margin_values, 65)
        for i in idxs:
            row = rows[i]
            if row["separation_margin"] < 0:
                row["separability_status"] = "overlap_review"
            elif row["distance_to_own_centroid"] <= clean_own and row["separation_margin"] >= clean_margin:
                row["separability_status"] = "clean_separable"
            elif row["distance_to_own_centroid"] <= usable_own and row["separation_margin"] >= usable_margin:
                row["separability_status"] = "usable_review"
            else:
                row["separability_status"] = "overlap_review"


def cluster_manifest_row(row: dict, copied: Path, reason: str) -> dict:
    return {
        "clip_id": row.get("clip_id", ""),
        "source_file": row.get("source_file", ""),
        "copied_file": str(copied),
        "target_emotion": row.get("target_emotion", ""),
        "character_name": row.get("character_name", ""),
        "speaker_cluster": row.get("speaker_cluster", ""),
        "separability_status": row.get("separability_status", ""),
        "distance_to_own_centroid": row.get("distance_to_own_centroid", ""),
        "distance_to_nearest_other_centroid": row.get("distance_to_nearest_other_centroid", ""),
        "nearest_other_emotion": row.get("nearest_other_emotion", ""),
        "separation_margin": row.get("separation_margin", ""),
        "review_priority_score": row.get("review_priority_score", ""),
        "evidence_count": row.get("evidence_count", ""),
        "top_score": row.get("top_score", ""),
        "top_gap": row.get("top_gap", ""),
        "audio_quality_status": row.get("audio_quality_status", ""),
        "transcript": row.get("transcript", ""),
        "selection_reason": reason,
    }


def write_cluster_reports_and_graphs(root: Path, rows: List[dict], selected: List[dict], overlap: List[dict], target: int, feature_method: str = "multi_feature") -> None:
    count_rows = []
    for label in EMOTIONS:
        data = [row for row in rows if row["target_emotion"] == label]
        selected_count = sum(1 for row in selected if row["target_emotion"] == label)
        count_rows.append({"emotion": label, "target_count": target, "selected_total_count": selected_count, "overlap_review_count": sum(1 for row in data if row.get("separability_status") == "overlap_review"), "shortage_count": max(0, target - selected_count), "status": "pass" if selected_count >= target else "shortage"})
    write_csv(root / "_cluster_clean_count_report.csv", count_rows, ["emotion", "target_count", "selected_total_count", "overlap_review_count", "shortage_count", "status"])
    overlap_counts = Counter((row["target_emotion"], row.get("nearest_other_emotion", "")) for row in overlap if row.get("nearest_other_emotion"))
    write_csv(root / "_overlap_report.csv", [{"target_emotion": a, "nearest_other_emotion": b, "overlap_count": c, "common_confusion_pair": f"{a} vs {b}", "notes": "review_before_training"} for (a, b), c in overlap_counts.most_common()], ["target_emotion", "nearest_other_emotion", "overlap_count", "common_confusion_pair", "notes"])
    analysis = root / "cluster_analysis"
    labels = [row["target_emotion"] for row in selected]
    counts = Counter(labels)
    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    ax.bar(EMOTIONS, [counts[label] for label in EMOTIONS], color=[COLORS[label] for label in EMOTIONS])
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(analysis / "class_count_bar.png")
    plt.close(fig)
    plot_pca(analysis / "pca_7emotion.png", selected, feature_method=feature_method)
    plot_before_after_filter(analysis / "pca_before_after_clustering_filter.png", rows, selected, feature_method=feature_method)
    plot_margin_hist(analysis / "separation_margin_histogram.png", rows)
    plot_box(analysis / "centroid_distance_boxplot.png", rows)


def plot_pca(path: Path, rows: List[dict], feature_method: str = "multi_feature") -> None:
    vectors = []
    cache = {}
    plot_rows = []
    for row in rows:
        vec, ok = feature_vector(row, cache, feature_method=feature_method)
        if ok:
            vectors.append(vec)
            plot_rows.append(row)
    if len(vectors) < 2:
        return
    coords = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(np.asarray(vectors, dtype=np.float32)))
    fig, ax = plt.subplots(figsize=(12, 7), dpi=140)
    for label in EMOTIONS:
        idx = [i for i, row in enumerate(plot_rows) if row["target_emotion"] == label]
        if idx:
            ax.scatter(coords[idx, 0], coords[idx, 1], s=28, c=COLORS[label], label=label, alpha=0.78)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_before_after_filter(path: Path, rows: List[dict], selected: List[dict], feature_method: str = "multi_feature") -> None:
    selected_ids = {id(row) for row in selected}
    vectors = []
    plot_rows = []
    cache = {}
    for row in rows:
        vec, ok = feature_vector(row, cache, feature_method=feature_method)
        if ok:
            vectors.append(vec)
            plot_rows.append(row)
    if len(vectors) < 2:
        return
    arr = np.asarray(vectors, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    scaled = StandardScaler().fit_transform(arr)
    coords = PCA(n_components=2, random_state=0).fit_transform(scaled)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=140, sharex=True, sharey=True)
    plot_sets = [
        ("Before clustering filter", list(range(len(plot_rows)))),
        ("After clustering filter", [i for i, row in enumerate(plot_rows) if id(row) in selected_ids]),
    ]
    for ax, (title, idxs) in zip(axes, plot_sets):
        for label in EMOTIONS:
            label_idxs = [i for i in idxs if plot_rows[i].get("target_emotion") == label]
            if not label_idxs:
                continue
            ax.scatter(
                coords[label_idxs, 0],
                coords[label_idxs, 1],
                s=20,
                c=COLORS[label],
                label=label,
                alpha=0.78,
            )
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("PC2")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Emotion", loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle(f"{feature_method} | cluster-clean target")
    fig.tight_layout(rect=(0, 0, 0.86, 0.95))
    fig.savefig(path)
    plt.close(fig)


def plot_margin_hist(path: Path, rows: List[dict]) -> None:
    vals = [to_float(row.get("separation_margin")) for row in rows]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    ax.hist(vals, bins=40, color="#4c78a8", alpha=0.82)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_box(path: Path, rows: List[dict]) -> None:
    data = [[to_float(row.get("distance_to_own_centroid")) for row in rows if row.get("target_emotion") == label] for label in EMOTIONS]
    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    ax.boxplot(data, tick_labels=EMOTIONS, showfliers=False)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_candidate_summary(input_dir: Path, rows: List[dict]) -> None:
    out = []
    for label in EMOTIONS:
        data = [row for row in rows if row.get("target_emotion") == label and is_review(row)]
        sources = Counter()
        for row in data:
            for src in str(row.get("evidence_sources", "")).split("|"):
                if src:
                    sources[src] += 1
        out.append({"emotion": label, "candidate_count": len(data), "high_priority_count": sum(1 for row in data if row.get("confidence_status") == "needs_review_high_priority"), "average_top_score": round(float(np.mean([to_float(row.get("top_score")) for row in data])) if data else 0.0, 6), "average_evidence_count": round(float(np.mean([to_float(row.get("evidence_count")) for row in data])) if data else 0.0, 6), "top_evidence_sources": "|".join(f"{k}:{v}" for k, v in sources.most_common(5))})
    write_csv(input_dir / "_candidate_summary_report.csv", out, ["emotion", "candidate_count", "high_priority_count", "average_top_score", "average_evidence_count", "top_evidence_sources"])


def write_branch_bias(input_dir: Path, rows: List[dict]) -> None:
    tops = defaultdict(list)
    for row in rows:
        for name, key in (("final", "target_emotion"), ("text", "text_model_top"), ("prosody", "prosody_top"), ("audio", "audio_model_top")):
            val = str(row.get(key, "")).strip()
            if val in EMOTIONS:
                tops[name].append(val)
    out = []
    for branch, vals in tops.items():
        counts = Counter(vals)
        dominant, count = counts.most_common(1)[0] if vals else ("", 0)
        ratio = count / max(1, len(vals))
        out.append({"branch": branch, "dominant_label": dominant, "dominant_label_count": count, "total_predictions": len(vals), "dominant_label_ratio": round(ratio, 6), "biased_for_this_run": str(ratio > 0.70).lower(), "action_taken": "report_only"})
    write_csv(input_dir / "_branch_bias_report.csv", out, ["branch", "dominant_label", "dominant_label_count", "total_predictions", "dominant_label_ratio", "biased_for_this_run", "action_taken"])


def write_label_collapse(input_dir: Path, rows: List[dict]) -> bool:
    accepted = [row for row in rows if row.get("confidence_status") == "auto_accept" and row.get("target_emotion") in EMOTIONS]
    counts = Counter(row["target_emotion"] for row in accepted)
    out = []
    collapse = False
    for label, count in counts.items():
        ratio = count / max(1, len(accepted))
        action = "report_only"
        if ratio > 0.70:
            collapse = True
            action = "label_collapse_detected"
        out.append({"label": label, "count": count, "ratio": round(ratio, 6), "action_taken": action})
    write_csv(input_dir / "_label_collapse_report.csv", out, ["label", "count", "ratio", "action_taken"])
    return collapse


def write_quality_summary(
    input_dir: Path,
    workflow_type: str,
    rows: List[dict],
    target_per_emotion: int,
    max_review_total: int,
    review_sample_per_class: int,
    cluster_clean_feature: str,
    shortlist: int,
    reduction: float,
    balanced: int,
    clean: int,
    mismatches: int,
    collapse: bool,
) -> None:
    counts = distribution_counts(rows)
    suffix = f"{target_per_emotion}_{cluster_clean_feature}" if cluster_clean_feature != "acoustic" else str(target_per_emotion)
    summary = [{
        "workflow_type": workflow_type,
        "input_emotion_dir": str(input_dir),
        "total_clips": len(rows),
        "auto_accept_or_accepted": sum(counts[label] for label in EMOTIONS),
        "needs_review": counts["Needs_Review"],
        "rejected": counts["Rejected"],
        "target_per_emotion": target_per_emotion,
        "review_sample_per_class": review_sample_per_class,
        "max_review_total": max_review_total,
        "cluster_clean_feature": cluster_clean_feature,
        "review_shortlist": shortlist,
        "shortlist_reduction_percent": round(reduction, 2),
        "balanced_target_total": balanced,
        "balanced_target_path": str(input_dir / f"balanced_review_target_{target_per_emotion}"),
        "cluster_clean_target_total": clean,
        "cluster_clean_target_path": str(input_dir / f"cluster_clean_target_{suffix}"),
        "count_mismatches": mismatches,
        "label_collapse_detected": "yes" if collapse else "no",
        "ready_for_manual_review_shortlist": "yes" if shortlist > 0 and mismatches == 0 else "no",
        "ready_for_final_training": "no",
    }]
    write_csv(input_dir / "_quality_control_summary.csv", summary, list(summary[0].keys()))
    print("\nQUALITY CONTROL SUMMARY:")
    for key, value in summary[0].items():
        print(f"- {key}: {value}")
    if sum(counts[label] for label in EMOTIONS) < sum(counts.values()) * 0.25:
        print('Workflow is usable for candidate review, not final automatic labels.')


def write_cluster_analysis(input_dir: Path, rows: List[dict]) -> None:
    analysis = input_dir / "cluster_analysis"
    clean_dir(analysis)
    usable = [row for row in rows if row.get("target_emotion") in EMOTIONS and audio_path_from_row(row).is_file() and row.get("confidence_status") != "reject"]
    cache: Dict[str, List[float]] = {}
    vectors, plot_rows = [], []
    for row in usable:
        vec, ok = feature_vector(row, cache)
        if ok:
            vectors.append(vec)
            plot_rows.append(row)
    counts = Counter(row.get("target_emotion") for row in usable)
    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    ax.bar(EMOTIONS, [counts[label] for label in EMOTIONS], color=[COLORS[label] for label in EMOTIONS])
    ax.tick_params(axis="x", rotation=30)
    ax.set_title("7 Emotion Class Counts")
    fig.tight_layout()
    fig.savefig(analysis / "class_count_bar.png")
    plt.close(fig)
    if len(vectors) < 2:
        write_csv(analysis / "cluster_points.csv", [], ["clip_id", "source_file", "emotion_label", "x_pca", "y_pca"])
        write_csv(analysis / "cluster_summary.csv", [], ["emotion_label", "sample_count"])
        return
    arr = np.asarray(vectors, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    scaled = StandardScaler().fit_transform(arr)
    assign_separability(plot_rows, scaled)
    coords = PCA(n_components=2, random_state=0).fit_transform(scaled)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=140)
    for label in EMOTIONS:
        idx = [i for i, row in enumerate(plot_rows) if row["target_emotion"] == label]
        if idx:
            ax.scatter(coords[idx, 0], coords[idx, 1], s=28, c=COLORS[label], label=f"{label} ({len(idx)})", alpha=0.78)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.set_title("7 Emotion PCA Cluster")
    fig.tight_layout()
    fig.savefig(analysis / "pca_7emotion.png")
    plt.close(fig)
    margins = [to_float(row.get("separation_margin")) for row in plot_rows]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    ax.hist(margins, bins=40, color="#4c78a8", alpha=0.82)
    ax.set_title("Separation Margin Histogram")
    fig.tight_layout()
    fig.savefig(analysis / "separation_margin_histogram.png")
    plt.close(fig)
    data = [[to_float(row.get("distance_to_own_centroid")) for row in plot_rows if row.get("target_emotion") == label] for label in EMOTIONS]
    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    ax.boxplot(data, tick_labels=EMOTIONS, showfliers=False)
    ax.tick_params(axis="x", rotation=30)
    ax.set_title("Centroid Distance by Emotion")
    fig.tight_layout()
    fig.savefig(analysis / "centroid_distance_boxplot.png")
    plt.close(fig)
    point_rows = []
    for i, row in enumerate(plot_rows):
        point_rows.append(
            {
                "clip_id": row.get("clip_id", ""),
                "source_file": row.get("source_file", ""),
                "emotion_label": row.get("target_emotion", ""),
                "confidence_status": row.get("confidence_status", ""),
                "separability_status": row.get("separability_status", ""),
                "x_pca": round(float(coords[i, 0]), 6),
                "y_pca": round(float(coords[i, 1]), 6),
                "distance_to_own_centroid": row.get("distance_to_own_centroid", ""),
                "nearest_other_emotion": row.get("nearest_other_emotion", ""),
                "separation_margin": row.get("separation_margin", ""),
            }
        )
    write_csv(analysis / "cluster_points.csv", point_rows, ["clip_id", "source_file", "emotion_label", "confidence_status", "separability_status", "x_pca", "y_pca", "distance_to_own_centroid", "nearest_other_emotion", "separation_margin"])
    summary = []
    for label in EMOTIONS:
        label_rows = [row for row in plot_rows if row.get("target_emotion") == label]
        summary.append({"emotion_label": label, "sample_count": len(label_rows), "overlap_review_count": sum(1 for row in label_rows if row.get("separability_status") == "overlap_review")})
    write_csv(analysis / "cluster_summary.csv", summary, ["emotion_label", "sample_count", "overlap_review_count"])


def run_post_label_quality_control(
    output_root: Path,
    input_emotion_dir: Path,
    workflow_type: str,
    target_per_emotion: int = 100,
    max_review_total: int = 700,
    review_sample_per_class: int = 50,
    cluster_clean: bool = True,
    allow_overlap_fill: bool = False,
    allow_usable_fill: bool = False,
    cluster_clean_feature: str = "mfcc",
):
    input_emotion_dir = Path(input_emotion_dir)
    if not input_emotion_dir.is_dir():
        print(f"Quality control skipped; emotion directory not found: {input_emotion_dir}")
        return 0
    rows = normalize_rows(input_emotion_dir, workflow_type)
    if not rows:
        print(f"Quality control skipped; no rows found in: {input_emotion_dir}")
        return 0
    mismatches = write_count_consistency(input_emotion_dir, rows, workflow_type)
    shortlist, reduction = write_review_shortlist(input_emotion_dir, rows, max_review_total, review_sample_per_class)
    balanced = write_balanced_target(input_emotion_dir, rows, target_per_emotion, max_review_total)
    clean = write_cluster_clean(input_emotion_dir, rows, target_per_emotion, allow_overlap_fill=allow_overlap_fill, allow_usable_fill=allow_usable_fill, feature_method=cluster_clean_feature) if cluster_clean else 0
    write_candidate_summary(input_emotion_dir, rows)
    write_branch_bias(input_emotion_dir, rows)
    collapse = write_label_collapse(input_emotion_dir, rows)
    write_cluster_analysis(input_emotion_dir, rows)
    write_quality_summary(
        input_emotion_dir,
        workflow_type,
        rows,
        target_per_emotion,
        max_review_total,
        review_sample_per_class,
        cluster_clean_feature,
        shortlist,
        reduction,
        balanced,
        clean,
        mismatches,
        collapse,
    )
    return 0
