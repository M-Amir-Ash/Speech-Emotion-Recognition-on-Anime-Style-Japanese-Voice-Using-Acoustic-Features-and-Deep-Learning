import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = BASE_DIR / "output" / "pipeline_workspace"
LABELING_CONFIG_PATH = BASE_DIR / "character_labeling_config.json"
CSV_ENCODING = "utf-8-sig"

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant_Surprise", "Sad"]
EXTRA_FOLDERS = ["Needs_Review", "Rejected"]
REVIEW_STATUSES = {"needs_review", "needs_review_high_priority"}
CANDIDATE_FOLDERS = {
    "Candidate_Fear": "Fear",
    "Candidate_Disgust": "Disgust",
    "Candidate_Pleasant_Surprise": "Pleasant_Surprise",
}
LOW_COUNT_CLASSES = {"Disgust", "Pleasant_Surprise"}
COLORS = {
    "Angry": "#d73027",
    "Disgust": "#1a9850",
    "Fear": "#4575b4",
    "Happy": "#fdae61",
    "Neutral": "#7f7f7f",
    "Pleasant_Surprise": "#fee08b",
    "Sad": "#5e3c99",
    "Needs_Review": "#8c8c8c",
    "Rejected": "#222222",
}
SSL_EMBEDDER = None


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(text).strip()) or "Unknown"


def read_manifest(root: Path) -> Dict[str, dict]:
    manifest = root / "_character_emotion_manifest.csv"
    if not manifest.is_file():
        return {}
    rows = {}
    with manifest.open("r", encoding=CSV_ENCODING, newline="") as f:
        for row in csv.DictReader(f):
            for key in ("output_file", "global_output_file", "source_file"):
                value = str(row.get(key, "")).strip()
                if value:
                    rows[str(Path(value).resolve())] = row
                    rows[Path(value).name] = row
    return rows


def acoustic_features(path: Path, sr: int = 16000) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        return np.zeros(13, dtype=np.float32)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_ratio = float(np.mean(np.abs(harmonic)) / (np.mean(np.abs(y)) + 1e-8))
    duration_sec = float(len(y) / sr)
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    rms_peak = float(np.max(rms))
    burst_ratio = float(rms_peak / (rms_mean + 1e-8))
    try:
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
        voiced = f0[np.isfinite(f0)]
        f0_mean = float(np.mean(voiced)) if voiced.size else 0.0
        f0_std = float(np.std(voiced)) if voiced.size else 0.0
        voiced_ratio = float(voiced.size / max(1, f0.size))
    except Exception:
        f0_mean = 0.0
        f0_std = 0.0
        voiced_ratio = 0.0
    return np.asarray(
        [
            duration_sec,
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
        ],
        dtype=np.float32,
    )


def load_ssl_embedder():
    global SSL_EMBEDDER
    if SSL_EMBEDDER is not None:
        return SSL_EMBEDDER
    if not LABELING_CONFIG_PATH.is_file():
        raise RuntimeError("character_labeling_config.json not found")
    import json
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    with LABELING_CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    ssl_cfg = cfg.get("models", {}).get("japanese_ssl_embedding", {})
    device = str(ssl_cfg.get("device", "cuda")).lower()
    actual_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    errors = []
    for model_name in ssl_cfg.get("preferred_models", []):
        try:
            extractor = AutoFeatureExtractor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(actual_device).eval()
            SSL_EMBEDDER = {
                "model_name": model_name,
                "extractor": extractor,
                "model": model,
                "torch": torch,
                "device": actual_device,
            }
            print(f"Cluster analysis Japanese SSL model loaded: {model_name} ({actual_device})", flush=True)
            return SSL_EMBEDDER
        except Exception as exc:
            errors.append(f"{model_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors) if errors else "no Japanese SSL models configured")


def ssl_embedding(path: Path, sr: int = 16000) -> np.ndarray:
    embedder = load_ssl_embedder()
    y, _ = librosa.load(path, sr=sr, mono=True)
    inputs = embedder["extractor"](y, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(embedder["device"]) for k, v in inputs.items()}
    with embedder["torch"].no_grad():
        out = embedder["model"](**inputs)
    emb = out.last_hidden_state.mean(dim=1).detach().cpu().numpy()[0].astype(np.float32)
    norm = float(np.linalg.norm(emb))
    return emb / norm if norm > 1e-8 else emb


def feature_for_path(path: Path, feature_method: str) -> Tuple[np.ndarray, str]:
    if feature_method == "japanese_ssl_embedding":
        try:
            return ssl_embedding(path), "japanese_ssl_embedding"
        except Exception as exc:
            print(f"Japanese SSL cluster feature unavailable; using acoustic for {path.name}. Reason: {type(exc).__name__}: {exc}", flush=True)
    return acoustic_features(path), "acoustic"


def collect_points(root: Path, character_root: Path, manifest_rows: Dict[str, dict], limit: int, feature_method: str) -> Tuple[List[dict], np.ndarray]:
    rows: List[dict] = []
    features: List[np.ndarray] = []
    folders = EMOTIONS + EXTRA_FOLDERS + list(CANDIDATE_FOLDERS)
    for folder in folders:
        folder_path = root / folder
        if not folder_path.is_dir():
            continue
        wavs = sorted(p for p in folder_path.glob("*.wav") if p.is_file())
        if limit > 0:
            wavs = wavs[:limit]
        for wav in wavs:
            meta = manifest_rows.get(str(wav.resolve())) or manifest_rows.get(wav.name) or {}
            try:
                feature, actual_feature_method = feature_for_path(wav, feature_method)
            except Exception as exc:
                print(f"Skipping analysis feature for {wav}: {type(exc).__name__}: {exc}", flush=True)
                continue
            features.append(feature)
            rows.append(
                {
                    "clip_id": meta.get("clip_id", wav.stem),
                    "source_file": meta.get("source_file", str(wav)),
                    "output_file": str(wav),
                    "character_name": meta.get("character_name", character_root.name if character_root.name != "_global" else "Global"),
                    "speaker_cluster": meta.get("speaker_cluster", ""),
                    "emotion_label": CANDIDATE_FOLDERS.get(folder) or meta.get("candidate_emotion") or meta.get("top_label", folder if folder in EMOTIONS else ""),
                    "confidence_status": meta.get("confidence_status", "auto_accept" if folder in EMOTIONS else folder.lower()),
                    "top_score": meta.get("top_score", ""),
                    "top_gap": meta.get("top_gap", ""),
                    "feature_method": actual_feature_method,
                    "folder": folder,
                }
            )
    if not features:
        return rows, np.empty((0, 0), dtype=np.float32)
    return rows, np.vstack(features).astype(np.float32)


GRAPH_WARNINGS: List[dict] = []


def add_graph_warning(scope: str, message: str) -> None:
    GRAPH_WARNINGS.append({"scope": scope, "warning": message})
    print(f"Warning: {message}", flush=True)


def reduce_features(features: np.ndarray, skip_tsne: bool = False, skip_umap: bool = False, scope: str = "") -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if len(features) < 2:
        return out
    scaled = StandardScaler().fit_transform(features)
    out["pca"] = PCA(n_components=2, random_state=0).fit_transform(scaled)
    if skip_tsne:
        add_graph_warning(scope, "t-SNE skipped by --skip-tsne")
    elif len(features) >= 4:
        perplexity = max(2, min(30, (len(features) - 1) // 3))
        if perplexity < len(features):
            out["tsne"] = TSNE(
                n_components=2,
                perplexity=perplexity,
                metric="euclidean",
                init="pca",
                learning_rate="auto",
                random_state=0,
            ).fit_transform(scaled)
    if skip_umap:
        add_graph_warning(scope, "UMAP skipped by --skip-umap")
    else:
        try:
            import umap

            if len(features) >= 4:
                out["umap"] = umap.UMAP(n_components=2, n_neighbors=min(15, len(features) - 1), min_dist=0.12, random_state=0).fit_transform(scaled)
        except Exception as exc:
            add_graph_warning(scope, f"UMAP unavailable; skipping UMAP plot. Reason: {type(exc).__name__}: {exc}")
    return out


def centroid_distances(rows: List[dict], features: np.ndarray) -> None:
    by_label: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if row["emotion_label"] in EMOTIONS:
            by_label[row["emotion_label"]].append(i)
    for label, idxs in by_label.items():
        if len(idxs) == 1:
            rows[idxs[0]]["distance_to_emotion_centroid"] = 0.0
            rows[idxs[0]]["outlier_status"] = "good_sample"
            continue
        pts = features[idxs]
        centroid = np.mean(pts, axis=0)
        distances = np.linalg.norm(pts - centroid, axis=1)
        p75 = float(np.percentile(distances, 75))
        p90 = float(np.percentile(distances, 90))
        for idx, distance in zip(idxs, distances):
            if distance <= p75:
                status = "good_sample"
            elif distance <= p90 or label in LOW_COUNT_CLASSES:
                status = "needs_review_distance"
            else:
                status = "outlier_sample"
            rows[idx]["distance_to_emotion_centroid"] = float(distance)
            rows[idx]["outlier_status"] = status
    for row in rows:
        row.setdefault("distance_to_emotion_centroid", "")
        row.setdefault("outlier_status", "")


def plot_scatter(coords: np.ndarray, rows: List[dict], output: Path, title: str, method: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=140)
    counts = Counter(row["emotion_label"] for row in rows if row["emotion_label"] in EMOTIONS)
    for label in EMOTIONS:
        idx = [i for i, row in enumerate(rows) if row["emotion_label"] == label]
        if not idx:
            continue
        pts = coords[idx]
        ax.scatter(pts[:, 0], pts[:, 1], s=34, alpha=0.78, c=COLORS[label], label=f"{label} ({counts[label]})", edgecolors="white", linewidths=0.35)
    ax.set_title(f"{title} - 7 Emotion {method.upper()} Cluster | acoustic features")
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Emotion", loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_counts(rows: List[dict], output: Path, title: str) -> None:
    labels = EMOTIONS + EXTRA_FOLDERS
    counts = [sum(1 for row in rows if row["folder"] == label or row["emotion_label"] == label) for label in labels]
    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    ax.bar(labels, counts, color=[COLORS.get(label, "#777777") for label in labels])
    ax.set_title(f"{title} - Class Counts")
    ax.set_ylabel("Clips")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_distances(rows: List[dict], output: Path, title: str) -> None:
    distances = [float(row["distance_to_emotion_centroid"]) for row in rows if row.get("distance_to_emotion_centroid") not in ("", None)]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    if distances:
        ax.hist(distances, bins=min(30, max(5, int(math.sqrt(len(distances))))), color="#4c78a8", alpha=0.82)
    ax.set_title(f"{title} - Distance to Emotion Centroid")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Clips")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def write_points_csv(rows: List[dict], reducers: Dict[str, np.ndarray], output: Path) -> None:
    fields = [
        "clip_id",
        "source_file",
        "output_file",
        "character_name",
        "speaker_cluster",
        "emotion_label",
        "confidence_status",
        "top_score",
        "top_gap",
        "feature_method",
        "x_pca",
        "y_pca",
        "x_tsne",
        "y_tsne",
        "x_umap",
        "y_umap",
        "distance_to_emotion_centroid",
        "outlier_status",
    ]
    for i, row in enumerate(rows):
        for method in ("pca", "tsne", "umap"):
            coords = reducers.get(method)
            row[f"x_{method}"] = f"{float(coords[i, 0]):.6f}" if coords is not None else ""
            row[f"y_{method}"] = f"{float(coords[i, 1]):.6f}" if coords is not None else ""
    with output.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in rows])


def write_summary_csv(rows: List[dict], output: Path) -> None:
    fields = [
        "character_name",
        "speaker_cluster",
        "emotion_label",
        "sample_count",
        "mean_distance_to_emotion_centroid",
        "median_distance_to_emotion_centroid",
        "max_distance_to_emotion_centroid",
        "outlier_count",
        "needs_review_count",
        "auto_accept_count",
        "notes",
    ]
    summary = []
    for label in EMOTIONS:
        data = [row for row in rows if row["emotion_label"] == label]
        distances = np.asarray([float(row["distance_to_emotion_centroid"]) for row in data if row.get("distance_to_emotion_centroid") not in ("", None)], dtype=np.float32)
        summary.append(
            {
                "character_name": data[0]["character_name"] if data else "",
                "speaker_cluster": data[0]["speaker_cluster"] if data else "",
                "emotion_label": label,
                "sample_count": len(data),
                "mean_distance_to_emotion_centroid": round(float(np.mean(distances)) if len(distances) else 0.0, 6),
                "median_distance_to_emotion_centroid": round(float(np.median(distances)) if len(distances) else 0.0, 6),
                "max_distance_to_emotion_centroid": round(float(np.max(distances)) if len(distances) else 0.0, 6),
                "outlier_count": sum(1 for row in data if row.get("outlier_status") == "outlier_sample"),
                "needs_review_count": sum(1 for row in data if row.get("confidence_status") in REVIEW_STATUSES or row.get("outlier_status") == "needs_review_distance"),
                "auto_accept_count": sum(1 for row in data if row.get("confidence_status") == "auto_accept"),
                "notes": "low_count_class" if label in LOW_COUNT_CLASSES and len(data) < 10 else "",
            }
        )
    with output.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def warn_overlap(rows: List[dict], features: np.ndarray) -> None:
    centroids = []
    labels = []
    for label in EMOTIONS:
        idx = [i for i, row in enumerate(rows) if row["emotion_label"] == label]
        if len(idx) >= 2:
            centroids.append(np.mean(features[idx], axis=0))
            labels.append(label)
    if len(centroids) < 2:
        return
    dists = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dists.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    if dists and float(np.mean(dists)) < 1.0:
        print("Warning: emotion clusters are highly overlapping. Dataset may be difficult for training.", flush=True)


def print_counts(rows: List[dict], title: str) -> None:
    print(title, flush=True)
    for label in EMOTIONS:
        print(f"  {label}: {sum(1 for row in rows if row['emotion_label'] == label and row['confidence_status'] == 'auto_accept')}", flush=True)
    print(f"  Needs_Review: {sum(1 for row in rows if row['confidence_status'] in REVIEW_STATUSES or row['folder'] == 'Needs_Review')}", flush=True)
    print(f"  Rejected: {sum(1 for row in rows if row['confidence_status'] == 'reject' or row['folder'] == 'Rejected')}", flush=True)


def analyze_root(root: Path, manifest_rows: Dict[str, dict], per_emotion_limit: int, feature_method: str, skip_tsne: bool = False, skip_umap: bool = False) -> int:
    analysis_dir = root / "cluster_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    rows, features = collect_points(root, root, manifest_rows, per_emotion_limit, feature_method)
    plot_counts(rows, analysis_dir / "class_count_bar.png", root.name)
    if not rows:
        print(f"No clips found for cluster analysis: {root}", flush=True)
        return 1
    if len({row["emotion_label"] for row in rows if row["emotion_label"] in EMOTIONS}) <= 1:
        print("PCA cannot show emotion separation because only one class exists.", flush=True)
    reducers = reduce_features(features, skip_tsne=skip_tsne, skip_umap=skip_umap, scope=root.name)
    centroid_distances(rows, features)
    if "pca" in reducers:
        plot_scatter(reducers["pca"], rows, analysis_dir / "pca_7emotion.png", root.name, "pca")
    if "tsne" in reducers:
        plot_scatter(reducers["tsne"], rows, analysis_dir / "tsne_7emotion.png", root.name, "tsne")
    if "umap" in reducers:
        plot_scatter(reducers["umap"], rows, analysis_dir / "umap_7emotion.png", root.name, "umap")
    plot_distances(rows, analysis_dir / "distance_histogram.png", root.name)
    write_points_csv(rows, reducers, analysis_dir / "cluster_points.csv")
    write_summary_csv(rows, analysis_dir / "cluster_summary.csv")
    warn_overlap(rows, features)
    print_counts(rows, f"Emotion counts: {root.name}")
    print(f"Saved cluster analysis: {analysis_dir}", flush=True)
    return 0


def write_graph_warning_report(character_root: Path) -> None:
    path = character_root / "_graph_warning_report.csv"
    with path.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scope", "warning"])
        writer.writeheader()
        writer.writerows(GRAPH_WARNINGS)


def run(args: argparse.Namespace) -> int:
    workspace = Path(args.output_root or DEFAULT_WORKSPACE).resolve()
    character_root = workspace / "character_emotion_classified"
    if not character_root.is_dir():
        fallback = workspace / "multimodal_emotion_classified"
        if not fallback.is_dir():
            print(f"No analysis input found: {character_root} or {fallback}", flush=True)
            return 1
        character_root = fallback
    manifest_rows = read_manifest(character_root)
    exit_code = 0
    if getattr(args, "plot_shortlist_only", False):
        shortlist_root = character_root / "_review_shortlist"
        if not shortlist_root.is_dir():
            print(f"Review shortlist not found: {shortlist_root}", flush=True)
            return 1
        exit_code = analyze_root(shortlist_root, manifest_rows, args.per_emotion_limit, args.cluster_feature, args.skip_tsne, args.skip_umap)
    elif (character_root / "_global").is_dir():
        if not getattr(args, "plot_global_only", False):
            for child in sorted(p for p in character_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
                exit_code = max(exit_code, analyze_root(child, manifest_rows, args.per_emotion_limit, args.cluster_feature, args.skip_tsne, args.skip_umap))
        exit_code = max(exit_code, analyze_root(character_root / "_global", manifest_rows, args.per_emotion_limit, args.cluster_feature, args.skip_tsne, args.skip_umap))
    else:
        exit_code = analyze_root(character_root, manifest_rows, args.per_emotion_limit, args.cluster_feature, args.skip_tsne, args.skip_umap)
    write_graph_warning_report(character_root)
    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate character/global 7-emotion cluster analysis.")
    parser.add_argument("--output-root", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--per-emotion-limit", type=int, default=500)
    parser.add_argument("--cluster-feature", choices=["acoustic", "japanese_ssl_embedding"], default="acoustic")
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--plot-global-only", action="store_true")
    parser.add_argument("--plot-shortlist-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
