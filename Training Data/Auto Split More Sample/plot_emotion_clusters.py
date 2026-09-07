import argparse
import csv
import os
from typing import Dict, List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import configure_emotion_output as ceo


DEFAULT_ROOT = ceo.DEFAULT_CLASSIFIED_ROOT
DEFAULT_OUTPUT = os.path.join(DEFAULT_ROOT, "_emotion_cluster_plot.png")
DEFAULT_CSV = os.path.join(DEFAULT_ROOT, "_emotion_cluster_points.csv")
EMOTIONS = ceo.EMOTIONS

COLORS = {
    "Angry": "#d73027",
    "Disgust": "#1a9850",
    "Fear": "#4575b4",
    "Happy": "#fdae61",
    "Neutral": "#7f7f7f",
    "Pleasant_Surprise": "#fee08b",
    "Sad": "#5e3c99",
}


def wav_files(folder: str, limit: int) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if name.lower().endswith(".wav") and os.path.isfile(os.path.join(folder, name))
    ]
    return files[:limit] if limit > 0 else files


MODEL_EMBEDDER = None


def acoustic_embedding(path: str, sr: int = 16000) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        return np.zeros(63, dtype=np.float32)

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    duration = np.array([len(y) / float(sr)], dtype=np.float32)
    scalar = np.array(
        [
            float(np.mean(rms)),
            float(np.std(rms)),
            float(np.mean(zcr)),
            float(np.std(zcr)),
            float(np.mean(centroid)),
            float(np.std(centroid)),
            float(np.mean(bandwidth)),
            float(np.std(bandwidth)),
        ],
        dtype=np.float32,
    )
    parts = [
        duration,
        scalar,
        np.mean(mfcc, axis=1).astype(np.float32),
        np.std(mfcc, axis=1).astype(np.float32),
        np.mean(contrast, axis=1).astype(np.float32),
        np.std(contrast, axis=1).astype(np.float32),
    ]
    return np.concatenate(parts).astype(np.float32)


def load_model_embedder(model_name: str, device: str):
    global MODEL_EMBEDDER
    if MODEL_EMBEDDER is not None and MODEL_EMBEDDER["model_name"] == model_name and MODEL_EMBEDDER["device"] == device:
        return MODEL_EMBEDDER

    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    actual_device = device
    if device == "cuda" and not torch.cuda.is_available():
        actual_device = "cpu"

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(actual_device)
    model.eval()
    MODEL_EMBEDDER = {
        "model_name": model_name,
        "feature_extractor": feature_extractor,
        "model": model,
        "torch": torch,
        "device": actual_device,
    }
    print(f"Cluster embedding model runtime: {model_name} on {actual_device}", flush=True)
    return MODEL_EMBEDDER


def model_embedding(path: str, model_name: str, device: str, sr: int = 16000) -> np.ndarray:
    embedder = load_model_embedder(model_name, device)
    torch = embedder["torch"]
    y, _ = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        return np.zeros(768, dtype=np.float32)
    max_samples = sr * 12
    if y.size > max_samples:
        start = max(0, (y.size - max_samples) // 2)
        y = y[start:start + max_samples]

    inputs = embedder["feature_extractor"](
        y,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(embedder["device"]) for k, v in inputs.items()}
    with torch.no_grad():
        output = embedder["model"](**inputs)
        hidden = output.last_hidden_state[0]
        pooled = hidden.mean(dim=0).detach().cpu().numpy().astype(np.float32)
    norm = float(np.linalg.norm(pooled))
    return pooled / norm if norm > 1e-6 else pooled


def embedding_for_path(path: str, embedding_backend: str, embedding_model: str, device: str) -> np.ndarray:
    if embedding_backend == "model":
        return model_embedding(path, embedding_model, device)
    return acoustic_embedding(path)


def load_points(
    root: str,
    per_emotion_limit: int,
    embedding_backend: str,
    embedding_model: str,
    device: str,
) -> Tuple[List[str], List[str], np.ndarray]:
    paths: List[str] = []
    labels: List[str] = []
    features: List[np.ndarray] = []
    for emotion in EMOTIONS:
        for path in wav_files(os.path.join(root, emotion), per_emotion_limit):
            try:
                features.append(embedding_for_path(path, embedding_backend, embedding_model, device))
                paths.append(path)
                labels.append(emotion)
            except Exception as e:
                print(f"Skipping {path}: {type(e).__name__}: {e}", flush=True)
    if not features:
        return [], [], np.empty((0, 0), dtype=np.float32)
    return paths, labels, np.vstack(features)


def load_manifest_score_points(
    manifest_csv: str,
    per_emotion_limit: int,
    embedding_backend: str,
    embedding_model: str,
    device: str,
) -> Tuple[List[str], List[str], np.ndarray]:
    if not manifest_csv or not os.path.isfile(manifest_csv):
        return [], [], np.empty((0, 0), dtype=np.float32)

    with open(manifest_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    paths: List[str] = []
    labels: List[str] = []
    features: List[np.ndarray] = []
    print(f"Using manifest score-candidate plotting: {manifest_csv}", flush=True)
    for emotion in EMOTIONS:
        score_col = f"final_score_{emotion.lower()}"
        ranked = []
        for row in rows:
            path = str(row.get("output_file") or row.get("global_output_file") or row.get("source_file") or "").strip()
            if not path or not os.path.isfile(path):
                continue
            try:
                score = float(row.get(score_col) or 0.0)
            except ValueError:
                score = 0.0
            ranked.append((score, path))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[:per_emotion_limit] if per_emotion_limit > 0 else ranked
        for score, path in selected:
            if score <= 0.0:
                continue
            try:
                features.append(embedding_for_path(path, embedding_backend, embedding_model, device))
                paths.append(path)
                labels.append(emotion)
            except Exception as e:
                print(f"Skipping {path}: {type(e).__name__}: {e}", flush=True)

    if not features:
        return [], [], np.empty((0, 0), dtype=np.float32)
    return paths, labels, np.vstack(features)


def write_points_csv(csv_path: str, paths: List[str], labels: List[str], coords: np.ndarray) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["emotion", "pc1", "pc2", "source_file"])
        writer.writeheader()
        for path, label, xy in zip(paths, labels, coords):
            writer.writerow({
                "emotion": label,
                "pc1": f"{float(xy[0]):.6f}",
                "pc2": f"{float(xy[1]):.6f}",
                "source_file": path,
            })


def safe_name(text: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(text))
    return cleaned.strip("_") or "speaker"


def plot_cluster(
    root: str,
    output_png: str,
    output_csv: str,
    per_emotion_limit: int,
    title: str,
    embedding_backend: str,
    embedding_model: str,
    device: str,
    reducer: str,
    manifest_csv: str = "",
) -> None:
    try:
        if manifest_csv:
            paths, labels, features = load_manifest_score_points(
                manifest_csv,
                per_emotion_limit,
                embedding_backend,
                embedding_model,
                device,
            )
        else:
            paths, labels, features = load_points(root, per_emotion_limit, embedding_backend, embedding_model, device)
    except Exception as e:
        if embedding_backend != "model":
            raise
        print(f"Model embeddings failed, falling back to acoustic features. Reason: {type(e).__name__}: {e}", flush=True)
        if manifest_csv:
            paths, labels, features = load_manifest_score_points(
                manifest_csv,
                per_emotion_limit,
                "acoustic",
                embedding_model,
                device,
            )
        else:
            paths, labels, features = load_points(root, per_emotion_limit, "acoustic", embedding_model, device)
        embedding_backend = "acoustic"

    if len(paths) < 2:
        raise RuntimeError("Need at least 2 WAV clips across emotion folders to plot PCA clusters.")

    plotted_counts = {emotion: labels.count(emotion) for emotion in EMOTIONS}
    print("PCA plotted clip counts:", flush=True)
    for emotion in EMOTIONS:
        print(f"  {emotion}: {plotted_counts[emotion]}", flush=True)
    missing = [emotion for emotion in EMOTIONS if plotted_counts[emotion] == 0]
    if missing:
        print(
            "Warning: no clips available for these emotion labels, so no points can be drawn: "
            + ", ".join(missing),
            flush=True,
        )

    scaled = StandardScaler().fit_transform(features)
    explained = None
    reducer_name = reducer
    if reducer == "umap":
        try:
            import umap

            coords = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.12, metric="cosine", random_state=0).fit_transform(scaled)
        except Exception as e:
            print(f"UMAP unavailable, falling back to PCA. Reason: {type(e).__name__}: {e}", flush=True)
            reducer_name = "pca"
            pca = PCA(n_components=2, random_state=0)
            coords = pca.fit_transform(scaled)
            explained = pca.explained_variance_ratio_ * 100.0
    elif reducer == "tsne":
        from sklearn.manifold import TSNE

        perplexity = max(5, min(30, (len(paths) - 1) // 3))
        coords = TSNE(n_components=2, perplexity=perplexity, metric="cosine", init="pca", learning_rate="auto", random_state=0).fit_transform(scaled)
    else:
        pca = PCA(n_components=2, random_state=0)
        coords = pca.fit_transform(scaled)
        explained = pca.explained_variance_ratio_ * 100.0

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=140)
    for emotion in EMOTIONS:
        idx = [i for i, label in enumerate(labels) if label == emotion]
        if not idx:
            continue
        pts = coords[idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=34,
            alpha=0.78,
            c=COLORS.get(emotion, "#333333"),
            label=emotion,
            edgecolors="white",
            linewidths=0.35,
        )

    ax.set_title(f"{title} | {embedding_backend} embedding + {reducer_name.upper()}")
    if explained is not None:
        ax.set_xlabel(f"PC1 ({explained[0]:.1f}% var)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}% var)")
    else:
        ax.set_xlabel(f"{reducer_name.upper()} 1")
        ax.set_ylabel(f"{reducer_name.upper()} 2")
    ax.grid(True, alpha=0.25)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor=COLORS.get(emotion, "#333333"),
            markeredgecolor="white",
            markersize=7,
            linestyle="None",
            label=f"{emotion} ({plotted_counts[emotion]})",
        )
        for emotion in EMOTIONS
    ]
    ax.legend(handles=legend_handles, title="Emotion", loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)
    if output_csv:
        write_points_csv(output_csv, paths, labels, coords)


def plot_speaker_clusters(
    by_speaker_root: str,
    output_root: str,
    per_emotion_limit: int,
    embedding_backend: str,
    embedding_model: str,
    device: str,
    reducer: str,
) -> str:
    if not os.path.isdir(by_speaker_root):
        raise RuntimeError(f"Speaker emotion root not found: {by_speaker_root}")
    os.makedirs(output_root, exist_ok=True)
    summary_csv = os.path.join(output_root, "_speaker_cluster_plots.csv")
    rows: List[Dict[str, str]] = []

    for speaker in sorted(os.listdir(by_speaker_root)):
        speaker_root = os.path.join(by_speaker_root, speaker)
        if not os.path.isdir(speaker_root):
            continue
        wav_count = 0
        for emotion in EMOTIONS:
            wav_count += len(wav_files(os.path.join(speaker_root, emotion), per_emotion_limit))
        if wav_count < 2:
            continue

        speaker_safe = safe_name(speaker)
        output_png = os.path.join(output_root, f"{speaker_safe}_emotion_clusters.png")
        output_csv = os.path.join(output_root, f"{speaker_safe}_emotion_cluster_points.csv")
        try:
            plot_cluster(
                root=speaker_root,
                output_png=output_png,
                output_csv=output_csv,
                per_emotion_limit=per_emotion_limit,
                title=f"{speaker} | emotion clusters",
                embedding_backend=embedding_backend,
                embedding_model=embedding_model,
                device=device,
                reducer=reducer,
            )
            rows.append({
                "speaker": speaker,
                "clip_count": str(wav_count),
                "plot_png": output_png,
                "points_csv": output_csv,
            })
            print(f"Speaker cluster plot saved: {output_png}", flush=True)
        except Exception as e:
            print(f"Skipping speaker plot {speaker}: {type(e).__name__}: {e}", flush=True)

    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker", "clip_count", "plot_png", "points_csv"])
        writer.writeheader()
        writer.writerows(rows)
    return summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PCA emotion clusters from emotion audio folders.")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="multimodal_emotion_classified folder.")
    parser.add_argument(
        "--manifest",
        help="Character emotion manifest. If provided, plot balanced score-ranked candidates for all 7 emotions.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output PNG path.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Output point CSV path.")
    parser.add_argument("--no-csv", action="store_true", help="Only save the graph PNG; do not write point CSV.")
    parser.add_argument("--per-emotion-limit", type=int, default=500, help="Max clips per emotion. 0 = no limit.")
    parser.add_argument("--title", default="Anime emotion clusters", help="Plot title.")
    parser.add_argument(
        "--embedding-backend",
        choices=["model", "acoustic"],
        default="acoustic",
        help="Use local acoustic features or pretrained Hugging Face audio model embeddings. Default: %(default)s",
    )
    parser.add_argument(
        "--embedding-model",
        default="facebook/wav2vec2-base",
        help="Hugging Face audio model used for embeddings. Default: %(default)s",
    )
    parser.add_argument(
        "--embedding-device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for model embeddings. Default: %(default)s",
    )
    parser.add_argument(
        "--reducer",
        choices=["pca", "umap", "tsne"],
        default="pca",
        help="2D projection method. UMAP/TSNE can show clusters better than PCA. Default: %(default)s",
    )
    parser.add_argument(
        "--by-speaker-root",
        help="Root containing <Speaker>/<Emotion> folders. Creates one PCA graph per speaker.",
    )
    parser.add_argument(
        "--speaker-output-root",
        help="Output folder for per-speaker PCA graphs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.by_speaker_root:
        output_root = args.speaker_output_root or os.path.join(os.path.abspath(args.by_speaker_root), "_speaker_cluster_plots")
        summary = plot_speaker_clusters(
            by_speaker_root=os.path.abspath(args.by_speaker_root),
            output_root=os.path.abspath(output_root),
            per_emotion_limit=args.per_emotion_limit,
            embedding_backend=args.embedding_backend,
            embedding_model=args.embedding_model,
            device=args.embedding_device,
            reducer=args.reducer,
        )
        print(f"Speaker cluster plot summary saved: {summary}")
        return

    plot_cluster(
        root=os.path.abspath(args.root),
        output_png=os.path.abspath(args.output),
        output_csv="" if args.no_csv else os.path.abspath(args.csv),
        per_emotion_limit=args.per_emotion_limit,
        title=args.title,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        device=args.embedding_device,
        reducer=args.reducer,
        manifest_csv=os.path.abspath(args.manifest) if args.manifest else "",
    )
    print(f"Cluster plot saved: {os.path.abspath(args.output)}")
    if not args.no_csv:
        print(f"Cluster point CSV saved: {os.path.abspath(args.csv)}")


if __name__ == "__main__":
    main()
