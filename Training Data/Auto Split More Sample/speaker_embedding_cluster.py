import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = BASE_DIR / "output" / "pipeline_workspace"
DEFAULT_INPUT_ROOT = DEFAULT_WORKSPACE / "segmented" / "japanese_dialogue"
DEFAULT_OUTPUT_ROOT = DEFAULT_WORKSPACE
SPEAKER_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
CSV_ENCODING = "utf-8-sig"


def patch_torchaudio_legacy_backend_api() -> None:
    try:
        import torchaudio
    except Exception:
        return

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend=None: None  # type: ignore[attr-defined]


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(value)) or "unnamed"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def materialize_file(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def read_segment_manifest(input_root: Path) -> Dict[str, dict]:
    manifest_path = input_root / "_segments_manifest.csv"
    if not manifest_path.is_file():
        return {}
    try:
        df = pd.read_csv(manifest_path)
    except Exception as exc:
        print(f"Warning: could not read segment manifest {manifest_path}: {type(exc).__name__}: {exc}")
        return {}
    rows = {}
    for _, row in df.iterrows():
        filename = str(row.get("filename", "")).strip()
        if filename:
            rows[filename] = row.to_dict()
    return rows


def iter_dialogue_clips(input_root: Path) -> List[dict]:
    manifest = read_segment_manifest(input_root)
    clips = []
    for wav in sorted(input_root.rglob("*.wav")):
        if wav.name.startswith("_"):
            continue
        meta = manifest.get(wav.name, {})
        clip_id = wav.stem
        clips.append(
            {
                "clip_id": clip_id,
                "file_path": str(wav),
                "source_file": wav.name,
                "transcript": "" if pd.isna(meta.get("transcript", "")) else str(meta.get("transcript", "")),
                "duration_sec": float(meta.get("duration_sec") or 0.0),
                "source_media": "" if pd.isna(meta.get("source_media", "")) else str(meta.get("source_media", "")),
                "start_sec": "" if pd.isna(meta.get("start_sec", "")) else str(meta.get("start_sec", "")),
                "end_sec": "" if pd.isna(meta.get("end_sec", "")) else str(meta.get("end_sec", "")),
            }
        )
    return clips


class SpeakerEmbedder:
    def __init__(self, model_id: str = SPEAKER_MODEL_ID, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or self._default_device()
        self.model = None
        self.error = ""
        self._load()

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load(self) -> None:
        try:
            patch_torchaudio_legacy_backend_api()
            from speechbrain.inference.speaker import EncoderClassifier

            self.model = EncoderClassifier.from_hparams(source=self.model_id, run_opts={"device": self.device})
            print(f"Speaker embedding model loaded: {self.model_id} ({self.device})")
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.model = None
            print(
                "Warning: SpeechBrain ECAPA speaker embedding unavailable; "
                f"falling back to acoustic embeddings. Reason: {self.error}"
            )

    def encode(self, wav_path: str) -> np.ndarray:
        if self.model is not None:
            if hasattr(self.model, "encode_file"):
                emb = self.model.encode_file(wav_path)
            else:
                import torch
                import soundfile as sf
                from scipy.signal import resample_poly

                y, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
                if y.ndim > 1:
                    y = np.mean(y, axis=1).astype(np.float32)
                if int(sample_rate) != 16000:
                    gcd = int(np.gcd(int(sample_rate), 16000))
                    y = resample_poly(y, 16000 // gcd, int(sample_rate) // gcd).astype(np.float32)
                waveform = torch.as_tensor(y, dtype=torch.float32).unsqueeze(0)
                waveform = waveform.to(self.device)
                emb = self.model.encode_batch(waveform)
            arr = emb.squeeze().detach().cpu().numpy().astype(np.float32)
            return normalize_vector(arr)
        return acoustic_embedding(wav_path)


def acoustic_embedding(wav_path: str) -> np.ndarray:
    import librosa

    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    if len(y) == 0:
        return np.zeros(32, dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    stats = []
    for arr in [mfcc.mean(axis=1), mfcc.std(axis=1), np.asarray([zcr.mean(), zcr.std(), rms.mean(), rms.std(), centroid.mean(), centroid.std()])]:
        stats.extend(np.asarray(arr, dtype=np.float32).tolist())
    return normalize_vector(np.asarray(stats, dtype=np.float32))


def normalize_vector(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return arr
    return arr / norm


def save_metadata(clips: List[dict], out_path: Path) -> None:
    fields = [
        "clip_id",
        "file_path",
        "transcript",
        "duration_sec",
        "speaker_embedding_path",
        "source_media",
        "start_sec",
        "end_sec",
    ]
    with out_path.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: clip.get(k, "") for k in fields} for clip in clips])


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "agglomerative",
    expected_speakers: Optional[int] = None,
) -> np.ndarray:
    n = len(embeddings)
    if n == 0:
        return np.asarray([], dtype=int)
    if n == 1:
        return np.asarray([0], dtype=int)

    method = (method or "agglomerative").lower()
    if method == "hdbscan":
        try:
            import hdbscan

            labels = hdbscan.HDBSCAN(min_cluster_size=max(5, min(20, n // 20)), metric="euclidean").fit_predict(embeddings)
            return labels.astype(int)
        except Exception as exc:
            print(f"Warning: HDBSCAN unavailable/failed; falling back to agglomerative. Reason: {type(exc).__name__}: {exc}")
            method = "agglomerative"

    def kmeans_features(values: np.ndarray) -> np.ndarray:
        if values.shape[1] <= 64:
            from sklearn.preprocessing import StandardScaler

            scaled = StandardScaler().fit_transform(values)
            norms = np.linalg.norm(scaled, axis=1, keepdims=True)
            return scaled / np.maximum(norms, 1e-8)
        return values

    def estimate_speaker_count(count: int) -> int:
        return max(2, min(24, int(round(np.sqrt(max(count, 1) / 12.0)))))

    if method == "kmeans" or expected_speakers:
        from sklearn.cluster import KMeans

        k = int(expected_speakers or min(8, n))
        k = max(1, min(k, n))
        return KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(kmeans_features(embeddings)).astype(int)

    from sklearn.cluster import AgglomerativeClustering

    distance_threshold = 0.42
    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    labels = model.fit_predict(embeddings).astype(int)
    unique_labels = {int(label) for label in labels if int(label) >= 0}
    if len(unique_labels) <= 1 and n >= 20:
        from sklearn.cluster import KMeans

        k = estimate_speaker_count(n)
        print(
            "Warning: agglomerative clustering produced one speaker cluster; "
            f"falling back to automatic KMeans with {k} clusters. "
            "Use --expected-speakers to control this number.",
            flush=True,
        )
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(kmeans_features(embeddings)).astype(int)
    return labels


def cluster_distances(embeddings: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, Dict[int, float]]:
    distances = np.zeros(len(labels), dtype=np.float32)
    cluster_avg = {}
    for label in sorted(set(int(x) for x in labels)):
        idx = np.where(labels == label)[0]
        if label < 0 or len(idx) == 0:
            distances[idx] = 1.0
            cluster_avg[label] = 1.0
            continue
        centroid = normalize_vector(np.mean(embeddings[idx], axis=0))
        dist = 1.0 - np.clip(embeddings[idx] @ centroid, -1.0, 1.0)
        distances[idx] = dist.astype(np.float32)
        cluster_avg[label] = float(np.mean(dist)) if len(dist) else 1.0
    return distances, cluster_avg


def status_for_cluster(clip_count: int, avg_distance: float) -> str:
    if clip_count < 10:
        return "small_cluster"
    if avg_distance <= 0.35:
        return "good_cluster"
    if avg_distance <= 0.50:
        return "needs_review_cluster"
    return "bad_cluster"


def warning_for_cluster(status: str) -> str:
    if status == "small_cluster":
        return "clip_count_lt_10"
    if status == "needs_review_cluster":
        return "avg_cluster_distance_gt_0.35"
    if status == "bad_cluster":
        return "avg_cluster_distance_gt_0.50"
    return ""


def copy_unique(src: Path, dst_dir: Path) -> Path:
    ensure_dir(dst_dir)
    candidate = dst_dir / src.name
    if not candidate.exists():
        materialize_file(src, candidate)
        return candidate
    stem, suffix = src.stem, src.suffix
    idx = 2
    while True:
        candidate = dst_dir / f"{stem}__dup{idx}{suffix}"
        if not candidate.exists():
            materialize_file(src, candidate)
            return candidate
        idx += 1


def representative_indices(labels: np.ndarray, distances: np.ndarray, max_samples: int = 8) -> Dict[int, List[int]]:
    out = {}
    for label in sorted(set(int(x) for x in labels)):
        idx = np.where(labels == label)[0]
        ranked = sorted(idx.tolist(), key=lambda i: float(distances[i]))
        out[label] = ranked[:max_samples]
    return out


def run_speaker_clustering(args: argparse.Namespace) -> int:
    input_root = Path(args.input_root or DEFAULT_INPUT_ROOT).resolve()
    output_root = Path(args.output_root or DEFAULT_OUTPUT_ROOT).resolve()
    cluster_root = output_root / "speaker_clusters"
    embedding_root = cluster_root / "_embeddings"
    sample_root = cluster_root / "speaker_cluster_samples"
    template_path = cluster_root / "character_map_template.json"
    generated_template_path = cluster_root / "character_map_template.generated.json"
    preserved_template = None

    if not input_root.is_dir():
        print(f"Input root not found: {input_root}")
        return 1

    clips = iter_dialogue_clips(input_root)
    if not clips:
        print(f"No wav clips found under {input_root}")
        return 1

    if template_path.is_file() and not getattr(args, "force_character_map_template", False):
        preserved_template = template_path.read_text(encoding="utf-8")

    if cluster_root.exists():
        shutil.rmtree(cluster_root)
    ensure_dir(cluster_root)
    ensure_dir(embedding_root)
    ensure_dir(sample_root)

    embedder = SpeakerEmbedder()
    embeddings = []
    for idx, clip in enumerate(clips, start=1):
        wav_path = clip["file_path"]
        emb = embedder.encode(wav_path)
        emb_path = embedding_root / f"{idx:06d}_{safe_name(Path(wav_path).stem)}.npy"
        np.save(emb_path, emb)
        clip["speaker_embedding_path"] = str(emb_path)
        embeddings.append(emb)
        if idx % 100 == 0:
            print(f"Embedded {idx}/{len(clips)} clips")

    matrix = np.vstack(embeddings).astype(np.float32)
    labels = cluster_embeddings(matrix, args.cluster_method, args.expected_speakers)
    distances, cluster_avg = cluster_distances(matrix, labels)

    label_to_name = {}
    next_id = 0
    for label in sorted(set(int(x) for x in labels)):
        if label < 0:
            label_to_name[label] = "SPEAKER_UNKNOWN"
        else:
            label_to_name[label] = f"SPEAKER_{next_id:02d}"
            next_id += 1

    save_metadata(clips, cluster_root / "_speaker_embedding_metadata.csv")

    manifest_rows = []
    for clip, label, distance in zip(clips, labels, distances):
        cluster_name = label_to_name[int(label)]
        dst = copy_unique(Path(clip["file_path"]), cluster_root / cluster_name)
        confidence = max(0.0, min(1.0, 1.0 - float(distance)))
        manifest_rows.append(
            {
                "clip_id": clip["clip_id"],
                "source_file": clip["source_file"],
                "cluster_id": int(label),
                "speaker_cluster": cluster_name,
                "cluster_confidence": round(confidence, 4),
                "nearest_distance": round(float(distance), 4),
                "transcript": clip.get("transcript", ""),
                "duration_sec": clip.get("duration_sec", ""),
                "source_media": clip.get("source_media", ""),
                "start_sec": clip.get("start_sec", ""),
                "end_sec": clip.get("end_sec", ""),
                "speaker_embedding_path": clip.get("speaker_embedding_path", ""),
            }
        )

    manifest_path = cluster_root / "_speaker_cluster_manifest.csv"
    fields = [
        "clip_id",
        "source_file",
        "speaker_cluster",
        "speaker_embedding_path",
        "nearest_distance",
        "cluster_confidence",
        "transcript",
        "duration_sec",
        "start_sec",
        "end_sec",
        "source_media",
        "cluster_id",
    ]
    with manifest_path.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    report_rows = []
    for label, cluster_name in label_to_name.items():
        rows = [r for r in manifest_rows if r["speaker_cluster"] == cluster_name]
        durations = [float(r["duration_sec"] or 0.0) for r in rows]
        row_distances = np.asarray([float(r["nearest_distance"] or 1.0) for r in rows], dtype=np.float32)
        avg_distance = cluster_avg.get(int(label), 1.0)
        status = status_for_cluster(len(rows), avg_distance)
        report_rows.append(
            {
                "speaker_cluster": cluster_name,
                "clip_count": len(rows),
                "avg_duration": round(float(np.mean(durations)) if durations else 0.0, 3),
                "avg_cluster_distance": round(float(avg_distance), 4),
                "median_cluster_distance": round(float(np.median(row_distances)) if len(row_distances) else 1.0, 4),
                "max_cluster_distance": round(float(np.max(row_distances)) if len(row_distances) else 1.0, 4),
                "recommended_status": status,
                "warning_reason": warning_for_cluster(status),
            }
        )

    report_path = cluster_root / "_speaker_cluster_report.csv"
    with report_path.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "speaker_cluster",
                "clip_count",
                "avg_cluster_distance",
                "median_cluster_distance",
                "max_cluster_distance",
                "recommended_status",
                "warning_reason",
                "avg_duration",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    samples = representative_indices(labels, distances)
    for label, idxs in samples.items():
        cluster_name = label_to_name[int(label)]
        for i in idxs:
            copy_unique(Path(clips[i]["file_path"]), sample_root / cluster_name)

    mapping = {row["speaker_cluster"]: f"Character_{idx:02d}" for idx, row in enumerate(report_rows)}
    mapping["SPEAKER_UNKNOWN"] = "Unknown"
    if preserved_template is not None:
        template_path.write_text(preserved_template, encoding="utf-8")
        template_output_path = generated_template_path
    else:
        template_output_path = template_path
    with template_output_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"Speaker clusters saved: {cluster_root}")
    print(f"Speaker manifest saved: {manifest_path}")
    print(f"Cluster report saved: {report_path}")
    if preserved_template is not None:
        print(f"Existing character map kept: {template_path}")
        print(f"Generated character map template saved: {template_output_path}")
    else:
        print(f"Character map template saved: {template_output_path}")
    print(f"Cluster samples saved: {sample_root}")
    print("Edit character_map_template.json and replace Character_00, Character_01, etc. with real character names before running option 7.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster dialogue clips by speaker embedding.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--expected-speakers", type=int)
    parser.add_argument("--cluster-method", choices=["agglomerative", "hdbscan", "kmeans"], default="agglomerative")
    parser.add_argument("--copy-files", action="store_true", default=True)
    parser.add_argument(
        "--force-character-map-template",
        action="store_true",
        help="Overwrite character_map_template.json instead of preserving an existing edited map.",
    )
    return parser


def main() -> int:
    return run_speaker_clustering(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
