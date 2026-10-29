import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import librosa
import numpy as np
from scipy.signal import resample_poly
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = BASE_DIR / "output" / "pipeline_workspace"
CSV_ENCODING = "utf-8-sig"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant_Surprise", "Sad"]
REVIEW_STATUSES = {"needs_review", "needs_review_high_priority"}
SCORE_BRANCHES = [
    "japanese_audio_ser_scores",
    "general_7class_audio_support_scores",
    "prosody_scores",
    "text_scores",
    "final_scores",
]
COLORS = {
    "Angry": "#d73027",
    "Disgust": "#1a9850",
    "Fear": "#4575b4",
    "Happy": "#fdae61",
    "Neutral": "#7f7f7f",
    "Pleasant_Surprise": "#fee08b",
    "Sad": "#5e3c99",
}
CONFUSION_PAIRS = {
    tuple(sorted(("Fear", "Pleasant_Surprise"))): "Fear vs Pleasant_Surprise",
    tuple(sorted(("Happy", "Pleasant_Surprise"))): "Happy vs Pleasant_Surprise",
    tuple(sorted(("Sad", "Neutral"))): "Sad vs Neutral",
    tuple(sorted(("Angry", "Disgust"))): "Angry vs Disgust",
}
MINORITY_RESCUE_LABELS = {"Angry", "Disgust"}
CLUSTER_MIN_SAMPLES = 10
CLUSTER_MAX_CLUSTERS = len(EMOTIONS)
CLUSTER_OUTLIER_PERCENTILE = 95


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(value).strip()) or "Unknown"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_unique(src: Path, dst_dir: Path) -> Path:
    ensure_dir(dst_dir)
    candidate = dst_dir / src.name
    if not candidate.exists():
        shutil.copy2(src, candidate)
        return candidate
    idx = 2
    while True:
        candidate = dst_dir / f"{src.stem}__dup{idx}{src.suffix}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
        idx += 1


def to_float(value, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return default
        return val
    except (TypeError, ValueError):
        return default


def read_csv_rows(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding=CSV_ENCODING, newline="") as f:
        return list(csv.DictReader(f))


def read_debug_scores(path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in (str(item.get("clip_id", "")), str(item.get("source_file", ""))):
                if key:
                    rows[key] = item
    return rows


def resolve_audio_path(row: dict) -> Path:
    for key in ("output_file", "global_output_file", "all_top_label_output_file", "candidate_output_file", "global_candidate_output_file"):
        value = str(row.get(key, "")).strip()
        if value:
            path = Path(value)
            if path.is_file():
                return path
    return Path("")


def target_emotion(row: dict) -> str:
    for key in ("manual_label", "approved_label", "candidate_emotion", "top_label", "folder"):
        value = str(row.get(key, "")).strip()
        if value in EMOTIONS:
            return value
    return ""


def load_audio(path: Path, sr: int = 16000) -> np.ndarray:
    import soundfile as sf

    y, source_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1).astype(np.float32)
    if int(source_sr) != sr and y.size:
        gcd = int(np.gcd(int(source_sr), sr))
        y = resample_poly(y, sr // gcd, int(source_sr) // gcd).astype(np.float32)
    return np.asarray(y, dtype=np.float32)


def fast_acoustic_features(path: Path, sr: int = 16000) -> List[float]:
    y = load_audio(path, sr=sr)
    if y.size == 0:
        return [0.0] * 13
    duration = float(y.size / sr)
    frame = max(256, int(0.025 * sr))
    hop = max(128, int(0.010 * sr))
    if y.size < frame:
        frames = y.reshape(1, -1)
    else:
        idx = np.arange(frame)[None, :] + hop * np.arange(1 + (y.size - frame) // hop)[:, None]
        frames = y[idx]
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    signs = np.signbit(frames)
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1) if frames.shape[1] > 1 else np.zeros(len(frames))
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    rms_peak = float(np.max(rms))
    burst_ratio = float(rms_peak / (rms_mean + 1e-8))
    windowed = frames * np.hanning(frames.shape[1])[None, :]
    spec = np.abs(np.fft.rfft(windowed, axis=1)) + 1e-10
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sr)
    spec_sum = np.sum(spec, axis=1) + 1e-10
    centroid = np.sum(spec * freqs[None, :], axis=1) / spec_sum
    cumsum = np.cumsum(spec, axis=1)
    rolloff_idx = np.argmax(cumsum >= (0.85 * spec_sum[:, None]), axis=1)
    rolloff = freqs[np.clip(rolloff_idx, 0, len(freqs) - 1)]
    flatness = np.exp(np.mean(np.log(spec), axis=1)) / (np.mean(spec, axis=1) + 1e-10)
    # Cheap voiced/proxy pitch diagnostics: enough for separation, not for labeling.
    voiced_ratio = float(np.mean(rms > np.percentile(rms, 40))) if len(rms) else 0.0
    harmonic_ratio = float(np.mean(np.max(spec, axis=1) / spec_sum))
    f0_mean = 0.0
    f0_std = 0.0
    return [
        duration,
        rms_mean,
        rms_std,
        rms_peak,
        burst_ratio,
        float(np.mean(zcr)),
        float(np.mean(flatness)),
        f0_mean,
        f0_std,
        voiced_ratio,
        harmonic_ratio,
        float(np.mean(centroid)),
        float(np.mean(rolloff)),
    ]


def mfcc_paper9_features(path: Path, sr: int = 16000, n_mfcc: int = 13) -> List[float]:
    y, _ = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        return [0.0] * 9
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    vals = np.asarray(mfcc, dtype=np.float32).reshape(-1)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    q1, median, q3 = np.percentile(vals, [25, 50, 75])
    return [
        float(np.mean(vals)),
        float(np.std(vals)),
        float(np.max(vals)),
        float(np.min(vals)),
        float(np.var(vals)),
        float(q1),
        float(median),
        float(q3),
        float(q3 - q1),
    ]


def mfcc_stats_features(path: Path, sr: int = 16000, n_mfcc: int = 20) -> List[float]:
    y, _ = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        return [0.0] * (n_mfcc * 4 + 8)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    return np.concatenate(
        [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.std(delta, axis=1),
            np.asarray(
                [
                    len(y) / float(sr),
                    float(np.mean(rms)),
                    float(np.std(rms)),
                    float(np.mean(zcr)),
                    float(np.std(zcr)),
                    float(np.mean(centroid)),
                    float(np.std(centroid)),
                    float(np.mean(bandwidth)),
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32).tolist()


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


def base_feature_vector(audio_path: Path, feature_method: str) -> List[float]:
    if feature_method in PAPER_MFCC_FEATURE_MODES:
        return mfcc_paper9_features(audio_path)
    if feature_method == "multi_feature":
        return mfcc_paper9_features(audio_path)
    if feature_method == "mfcc_rich":
        return mfcc_stats_features(audio_path)
    return fast_acoustic_features(audio_path)


def build_feature_row(row: dict, debug: dict, audio_path: Path, cache: Dict[str, List[float]], feature_method: str = "mfcc") -> Tuple[List[float], bool]:
    key = f"{feature_method}:{audio_path.resolve()}"
    if key not in cache:
        try:
            cache[key] = base_feature_vector(audio_path, feature_method)
        except Exception:
            cache[key] = [0.0] * (9 if feature_method == "multi_feature" else 88 if feature_method == "mfcc_rich" else 9 if feature_method in PAPER_MFCC_FEATURE_MODES else 13)
    features = list(cache[key])
    if feature_method == "multi_feature":
        manifest_acoustic = [to_float(row.get(key)) for key in ACOUSTIC_MANIFEST_KEYS]
        if not any(abs(value) > 1e-12 for value in manifest_acoustic):
            manifest_acoustic = fast_acoustic_features(audio_path)[: len(ACOUSTIC_MANIFEST_KEYS)]
        features.extend(manifest_acoustic)
    if feature_method not in AUDIO_ONLY_FEATURE_MODES:
        for branch in SCORE_BRANCHES:
            scores = debug.get(branch, {}) if isinstance(debug.get(branch, {}), dict) else {}
            if branch == "final_scores" and not scores:
                scores = {label: to_float(row.get(f"final_score_{label.lower()}")) for label in EMOTIONS}
            features.extend([to_float(scores.get(label)) for label in EMOTIONS])
        features.extend(
            [
                to_float(row.get("top_score")),
                to_float(row.get("top_gap")),
                to_float(row.get("evidence_count")),
                to_float(row.get("review_priority_score")),
            ]
        )
    return features, any(abs(x) > 1e-12 for x in features)


def prepare_output(path: Path) -> None:
    if path.exists():
        backup = path.with_name(f"{path.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.move(str(path), str(backup))
        print(f"Existing cluster-clean target moved to: {backup}", flush=True)
    for label in EMOTIONS:
        ensure_dir(path / label)
        ensure_dir(path / "_overlap_review" / label)
        ensure_dir(path / "_minority_rescue_review" / label)
    ensure_dir(path / "_hard_ambiguous")
    ensure_dir(path / "cluster_analysis")


def percentile(values: List[float], p: float, default: float = 0.0) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float32), p)) if values else default


def classify_separability(rows: List[dict], scaled: np.ndarray) -> None:
    by_label: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_label[row["target_emotion"]].append(i)
    centroids = {label: np.mean(scaled[idxs], axis=0) for label, idxs in by_label.items() if idxs}
    for i, row in enumerate(rows):
        label = row["target_emotion"]
        own = float(np.linalg.norm(scaled[i] - centroids[label]))
        nearest_label = ""
        nearest_dist = float("inf")
        for other, centroid in centroids.items():
            if other == label:
                continue
            dist = float(np.linalg.norm(scaled[i] - centroid))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_label = other
        if nearest_dist == float("inf"):
            nearest_dist = 0.0
        row["distance_to_own_centroid"] = own
        row["distance_to_nearest_other_centroid"] = nearest_dist
        row["nearest_other_emotion"] = nearest_label
        row["separation_margin"] = nearest_dist - own

    thresholds = {}
    for label, idxs in by_label.items():
        own_values = [rows[i]["distance_to_own_centroid"] for i in idxs]
        margin_values = [rows[i]["separation_margin"] for i in idxs]
        if label == "Neutral":
            thresholds[label] = {
                "clean_own": percentile(own_values, 15),
                "usable_own": percentile(own_values, 35),
                "usable_margin": percentile(margin_values, 78),
                "clean_margin": percentile(margin_values, 92),
            }
        else:
            thresholds[label] = {
                "clean_own": percentile(own_values, 25),
                "usable_own": percentile(own_values, 50),
                "usable_margin": percentile(margin_values, 65),
                "clean_margin": percentile(margin_values, 85),
            }
    for row in rows:
        label = row["target_emotion"]
        th = thresholds[label]
        if row["audio_quality_status"] != "pass" or not row["has_features"]:
            row["separability_status"] = "hard_ambiguous"
        elif row["distance_to_nearest_other_centroid"] < row["distance_to_own_centroid"] or row["separation_margin"] < 0:
            row["separability_status"] = "overlap_review"
        elif row["distance_to_own_centroid"] <= th["clean_own"] and row["separation_margin"] >= th["clean_margin"]:
            row["separability_status"] = "clean_separable"
        elif row["distance_to_own_centroid"] <= th["usable_own"] and row["separation_margin"] >= th["usable_margin"]:
            row["separability_status"] = "usable_review"
        else:
            row["separability_status"] = "overlap_review"


def classify_kmeans_outliers(rows: List[dict], scaled: np.ndarray) -> None:
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
    # Run KMeans in the original scaled feature space so this selector matches
    # the MFCC training outlier-filter behavior while avoiding PCA-only distance
    # decisions.
    n_clusters = max(2, min(CLUSTER_MAX_CLUSTERS, label_count, n_samples - 1))
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    cluster_ids = km.fit_predict(scaled)
    centers = km.cluster_centers_
    # distance to cluster center (in scaled feature space)
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

    # Ensure each emotion retains at least a small core of samples
    for label in EMOTIONS:
        idx = [i for i, row in enumerate(rows) if row.get("target_emotion") == label]
        if not idx:
            continue
        min_keep = min(len(idx), max(2, int(math.ceil(len(idx) * 0.60))))
        if sum(1 for i in idx if keep[i]) < min_keep:
            for i in sorted(idx, key=lambda j: float(distances[j]))[:min_keep]:
                keep[i] = True

    # Compute centroids per emotion in scaled space for nearest-emotion comparisons
    emotion_centroids: Dict[str, np.ndarray] = {}
    for label in EMOTIONS:
        idx = [i for i, row in enumerate(rows) if row.get("target_emotion") == label]
        if idx:
            emotion_centroids[label] = np.mean(scaled[idx], axis=0)

    # First pass: compute distances and nearest emotion for every sample
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

    # Compute per-emotion thresholds similar to classify_separability
    by_label: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_label[row["target_emotion"]].append(i)
    thresholds: Dict[str, Dict[str, float]] = {}
    for label, idxs in by_label.items():
        own_values = [own_dists[i] for i in idxs]
        margin_values = [nearest_dists[i] - own_dists[i] for i in idxs]
        if label == "Neutral":
            thresholds[label] = {
                "clean_own": percentile(own_values, 15),
                "usable_own": percentile(own_values, 35),
                "usable_margin": percentile(margin_values, 78),
                "clean_margin": percentile(margin_values, 92),
            }
        else:
            thresholds[label] = {
                "clean_own": percentile(own_values, 25),
                "usable_own": percentile(own_values, 50),
                "usable_margin": percentile(margin_values, 65),
                "clean_margin": percentile(margin_values, 85),
            }

    # Second pass: assign statuses using both cluster keep and emotion thresholds
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
        th = thresholds.get(own_label, {})
        if row.get("audio_quality_status") != "pass" or not row.get("has_features"):
            row["separability_status"] = "hard_ambiguous"
        elif nearest_dist < own_dist or separation_margin < 0:
            row["separability_status"] = "overlap_review"
        elif keep[i] and own_dist <= th.get("clean_own", float("inf")) and separation_margin >= th.get("clean_margin", -float("inf")):
            row["separability_status"] = "clean_separable"
        elif keep[i] and own_dist <= th.get("usable_own", float("inf")) and separation_margin >= th.get("usable_margin", -float("inf")):
            row["separability_status"] = "usable_review"
        else:
            row["separability_status"] = "overlap_review"


def selection_key(row: dict) -> tuple:
    diversity = f"{row.get('character_name', '')}|{row.get('speaker_cluster', '')}"
    return (
        0 if row["separability_status"] == "clean_separable" else 1,
        -to_float(row.get("separation_margin")),
        to_float(row.get("distance_to_own_centroid")),
        -to_float(row.get("review_priority_score")),
        -to_float(row.get("evidence_count")),
        -to_float(row.get("top_gap")),
        diversity,
    )


def write_manifest(path: Path, rows: List[dict]) -> None:
    fields = [
        "clip_id",
        "source_file",
        "copied_file",
        "target_emotion",
        "character_name",
        "speaker_cluster",
        "separability_status",
        "distance_to_own_centroid",
        "distance_to_nearest_other_centroid",
        "nearest_other_emotion",
        "separation_margin",
        "review_priority_score",
        "evidence_count",
        "top_score",
        "top_gap",
        "audio_quality_status",
        "transcript",
        "selection_reason",
    ]
    with path.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in rows])


def write_count_report(path: Path, selected: List[dict], all_rows: List[dict], hard_rows: List[dict], target: int) -> None:
    fields = [
        "emotion",
        "target_count",
        "clean_selected_count",
        "usable_selected_count",
        "selected_total_count",
        "usable_review_count",
        "overlap_review_count",
        "hard_ambiguous_count",
        "shortage_count",
        "status",
        "notes",
    ]
    with path.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label in EMOTIONS:
            data = [row for row in all_rows if row.get("target_emotion") == label]
            selected_count = sum(1 for row in selected if row.get("target_emotion") == label)
            clean_selected = sum(1 for row in selected if row.get("target_emotion") == label and row.get("separability_status") == "clean_separable")
            usable_selected = sum(1 for row in selected if row.get("target_emotion") == label and row.get("separability_status") == "usable_review")
            shortage = max(0, target - selected_count)
            writer.writerow(
                {
                    "emotion": label,
                    "target_count": target,
                    "clean_selected_count": clean_selected,
                    "usable_selected_count": usable_selected,
                    "selected_total_count": selected_count,
                    "usable_review_count": sum(1 for row in data if row.get("separability_status") == "usable_review"),
                    "overlap_review_count": sum(1 for row in data if row.get("separability_status") == "overlap_review"),
                    "hard_ambiguous_count": sum(1 for row in hard_rows if row.get("target_emotion") == label),
                    "shortage_count": shortage,
                    "status": "pass" if selected_count >= target else "shortage",
                    "notes": "contains_usable_review_samples; inspect manifest before training" if usable_selected else ("do_not_fill_with_overlap" if shortage else ""),
                }
            )


def write_overlap_report(path: Path, rows: List[dict]) -> None:
    counts = Counter()
    for row in rows:
        if row.get("separability_status") == "overlap_review" and row.get("nearest_other_emotion"):
            counts[(row["target_emotion"], row["nearest_other_emotion"])] += 1
    with path.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target_emotion", "nearest_other_emotion", "overlap_count", "common_confusion_pair", "notes"])
        writer.writeheader()
        for (label, other), count in counts.most_common():
            pair = CONFUSION_PAIRS.get(tuple(sorted((label, other))), "")
            writer.writerow(
                {
                    "target_emotion": label,
                    "nearest_other_emotion": other,
                    "overlap_count": count,
                    "common_confusion_pair": pair,
                    "notes": "review_before_training" if pair else "",
                }
            )


def plot_outputs(output_root: Path, selected: List[dict], overlap: List[dict], all_rows: List[dict], scaled_lookup: Dict[str, np.ndarray], feature_method: str = "multi_feature") -> None:
    analysis = output_root / "cluster_analysis"
    ensure_dir(analysis)
    selected_counts = Counter(row["target_emotion"] for row in selected)
    title_counts = ", ".join(f"{label}:{selected_counts[label]}" for label in EMOTIONS)

    def pca_plot(rows: List[dict], name: str, title: str) -> None:
        pts = [scaled_lookup[row["clip_id"]] for row in rows if row["clip_id"] in scaled_lookup]
        if len(pts) < 2:
            return
        coords = PCA(n_components=2, random_state=0).fit_transform(np.vstack(pts))
        fig, ax = plt.subplots(figsize=(12, 7), dpi=140)
        for status, marker in (("clean_separable", "o"), ("usable_review", "^"), ("overlap_review", "x")):
            for label in EMOTIONS:
                idx = [i for i, row in enumerate(rows) if row["target_emotion"] == label and row["separability_status"] == status]
                if not idx:
                    continue
                ax.scatter(coords[idx, 0], coords[idx, 1], s=34, marker=marker, alpha=0.78, c=COLORS[label], label=f"{label} {status}")
        ax.set_title(f"{title} | {title_counts}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
        fig.tight_layout()
        fig.savefig(analysis / name)
        plt.close(fig)

    pca_plot(selected, "pca_clean_target.png", "Cluster-Clean Target PCA")
    pca_plot(overlap, "pca_overlap_review.png", "Overlap Review PCA")
    plot_before_after_filter(analysis / "pca_before_after_clustering_filter.png", all_rows, selected, scaled_lookup, feature_method)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    ax.bar(EMOTIONS, [selected_counts[label] for label in EMOTIONS], color=[COLORS[label] for label in EMOTIONS])
    ax.set_title("Cluster-Clean Target Class Counts")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(analysis / "class_count_bar.png")
    plt.close(fig)

    margins = [to_float(row.get("separation_margin")) for row in all_rows]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    ax.hist(margins, bins=40, color="#4c78a8", alpha=0.82)
    ax.set_title("Separation Margin Histogram")
    fig.tight_layout()
    fig.savefig(analysis / "separation_margin_histogram.png")
    plt.close(fig)

    data = [[to_float(row.get("distance_to_own_centroid")) for row in all_rows if row.get("target_emotion") == label] for label in EMOTIONS]
    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    ax.boxplot(data, tick_labels=EMOTIONS, showfliers=False)
    ax.set_title("Centroid Distance by Emotion")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(analysis / "centroid_distance_boxplot.png")
    plt.close(fig)


def plot_before_after_filter(path: Path, all_rows: List[dict], selected: List[dict], scaled_lookup: Dict[str, np.ndarray], feature_method: str) -> None:
    selected_ids = {str(row.get("clip_id", "")) for row in selected}
    plot_rows = [row for row in all_rows if str(row.get("clip_id", "")) in scaled_lookup]
    if len(plot_rows) < 2:
        return
    pts = np.vstack([scaled_lookup[str(row.get("clip_id", ""))] for row in plot_rows])
    coords = PCA(n_components=2, random_state=0).fit_transform(pts)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=140, sharex=True, sharey=True)
    plot_sets = [
        ("Before clustering filter", list(range(len(plot_rows)))),
        ("After clustering filter", [i for i, row in enumerate(plot_rows) if str(row.get("clip_id", "")) in selected_ids]),
    ]
    for ax, (title, idxs) in zip(axes, plot_sets):
        for label in EMOTIONS:
            label_idxs = [i for i in idxs if plot_rows[i].get("target_emotion") == label]
            if not label_idxs:
                continue
            ax.scatter(coords[label_idxs, 0], coords[label_idxs, 1], s=20, alpha=0.78, c=COLORS[label], label=label)
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


def run(args: argparse.Namespace) -> int:
    workspace = Path(args.output_root or DEFAULT_WORKSPACE).resolve()
    character_root = workspace / "character_emotion_classified"
    manifest_path = character_root / "_character_emotion_manifest.csv"
    if not manifest_path.is_file():
        print(f"Character emotion manifest not found: {manifest_path}", flush=True)
        return 1
    target = int(args.target_per_emotion)
    suffix = f"{target}_{args.cluster_clean_feature}" if args.cluster_clean_feature != "acoustic" else str(target)
    output_root = workspace / f"cluster_clean_target_{suffix}"
    prepare_output(output_root)

    manifest = read_csv_rows(manifest_path)
    debug = read_debug_scores(character_root / "_debug_scores.jsonl")
    cache: Dict[str, List[float]] = {}
    rows: List[dict] = []
    features: List[List[float]] = []
    hard_rows: List[dict] = []

    for raw in manifest:
        row = dict(raw)
        label = target_emotion(row)
        audio_path = resolve_audio_path(row)
        if not label or not audio_path.is_file() or row.get("audio_quality_status") != "pass":
            row["target_emotion"] = label
            row["separability_status"] = "hard_ambiguous"
            hard_rows.append(row)
            continue
        dbg = debug.get(str(row.get("clip_id", ""))) or debug.get(str(row.get("source_file", ""))) or {}
        feature, has_features = build_feature_row(row, dbg, audio_path, cache, args.cluster_clean_feature)
        row["target_emotion"] = label
        row["audio_path"] = str(audio_path)
        row["has_features"] = has_features
        rows.append(row)
        features.append(feature)

    if not rows:
        print("No usable rows found for cluster-clean selection.", flush=True)
        return 1
    arr = np.asarray(features, dtype=np.float32)
    med = np.nanmedian(np.where(np.isfinite(arr), arr, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    arr = np.where(np.isfinite(arr), arr, med)
    scaled = StandardScaler().fit_transform(arr)
    classify_separability(rows, scaled)
    scaled_lookup = {row["clip_id"]: scaled[i] for i, row in enumerate(rows)}

    selected: List[dict] = []
    overlap_rows: List[dict] = []
    for label in EMOTIONS:
        label_rows = [row for row in rows if row["target_emotion"] == label]
        clean_pool = [row for row in label_rows if row["separability_status"] == "clean_separable"]
        if args.allow_usable_fill:
            clean_pool.extend([row for row in label_rows if row["separability_status"] == "usable_review"])
        if args.allow_overlap_fill:
            clean_pool.extend([row for row in label_rows if row["separability_status"] == "overlap_review"])
        clean_pool.sort(key=selection_key)
        chosen = clean_pool[:target]
        chosen_ids = {id(row) for row in chosen}
        selected.extend(chosen)
        overlap_rows.extend([row for row in label_rows if row["separability_status"] == "overlap_review" or id(row) not in chosen_ids])

    copied_rows: List[dict] = []
    rescue_rows: List[dict] = []
    for row in selected:
        copied = copy_unique(Path(row["audio_path"]), output_root / row["target_emotion"])
        row["copied_file"] = str(copied)
        row["selection_reason"] = row["separability_status"]
        copied_rows.append(row)
    selected_counts = Counter(row["target_emotion"] for row in selected)
    for label in MINORITY_RESCUE_LABELS:
        shortage = max(0, target - selected_counts[label])
        if shortage <= 0:
            continue
        rescue_pool = [
            row for row in overlap_rows
            if row.get("target_emotion") == label
            and row.get("audio_quality_status") == "pass"
            and to_float(row.get("separation_margin")) >= 0.0
        ]
        for row in sorted(rescue_pool, key=selection_key)[:shortage]:
            if row.get("audio_path") and Path(row["audio_path"]).is_file():
                copied = copy_unique(Path(row["audio_path"]), output_root / "_minority_rescue_review" / label)
                rescue_copy = dict(row)
                rescue_copy["copied_file"] = str(copied)
                rescue_copy["selection_reason"] = "minority_rescue_review"
                rescue_rows.append(rescue_copy)
    for row in overlap_rows:
        if row.get("audio_path") and Path(row["audio_path"]).is_file():
            copied = copy_unique(Path(row["audio_path"]), output_root / "_overlap_review" / row["target_emotion"])
            row["copied_file"] = str(copied)
            row["selection_reason"] = "overlap_or_not_selected"
    for row in hard_rows:
        src = resolve_audio_path(row)
        if src.is_file():
            copied = copy_unique(src, output_root / "_hard_ambiguous")
            row["copied_file"] = str(copied)
            row["selection_reason"] = "hard_ambiguous"

    write_manifest(output_root / "_cluster_clean_manifest.csv", copied_rows + overlap_rows + hard_rows)
    write_manifest(output_root / "_minority_rescue_manifest.csv", rescue_rows)
    write_count_report(output_root / "_cluster_clean_count_report.csv", copied_rows, rows, hard_rows, target)
    write_overlap_report(output_root / "_overlap_report.csv", overlap_rows)
    plot_outputs(output_root, copied_rows, overlap_rows, rows, scaled_lookup, args.cluster_clean_feature)

    print("Cluster-clean target created:", flush=True)
    counts = Counter(row["target_emotion"] for row in copied_rows)
    for label in EMOTIONS:
        print(f"{label}: {counts[label]} / {target}", flush=True)
    print("\nOverlap warnings:", flush=True)
    overlap_counts = Counter((row["target_emotion"], row.get("nearest_other_emotion", "")) for row in overlap_rows if row.get("nearest_other_emotion"))
    for pair_name in ("Fear vs Pleasant_Surprise", "Happy vs Pleasant_Surprise", "Sad vs Neutral", "Angry vs Disgust"):
        labels = pair_name.split(" vs ")
        count = sum(count for (a, b), count in overlap_counts.items() if set((a, b)) == set(labels))
        print(f"- {pair_name}: {count} clips", flush=True)
    print("\nFinal recommendation:", flush=True)
    print("- Train only with clean_separable samples.", flush=True)
    print("- Review overlap_review samples only if more data is needed.", flush=True)
    print("- Do not force 100 samples if the class is too overlapped.", flush=True)
    print(f"Output: {output_root}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select cluster-separable balanced target candidates.")
    parser.add_argument("--output-root", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--target-per-emotion", type=int, default=100)
    parser.add_argument(
        "--cluster-clean-feature",
        choices=["multi_feature", "mfcc", "mfcc_paper9", "mfcc_separable", "mfcc_rich", "acoustic"],
        default="multi_feature",
        help=(
            "Feature space used for cluster-clean filtering. "
            "multi_feature is recommended for one dataset shared by all training methods; "
            "mfcc is notebook-style KMeans outlier filtering; "
            "mfcc_separable uses the same 9 MFCC stats but selects lower-overlap emotion samples."
        ),
    )
    parser.add_argument("--allow-usable-fill", action="store_true", help="Allow borderline usable_review clips to fill class shortages. Default keeps main target clean_separable only.")
    parser.add_argument("--allow-overlap-fill", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
