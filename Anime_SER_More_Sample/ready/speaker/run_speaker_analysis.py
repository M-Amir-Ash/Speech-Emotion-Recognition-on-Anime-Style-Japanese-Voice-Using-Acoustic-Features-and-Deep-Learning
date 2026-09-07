"""Estimate anonymous speaker/voice clusters without changing the frozen SER experiment.

This analysis uses SpeechBrain ECAPA embeddings only. It deliberately has no
acoustic-feature fallback, character naming, audio copying, split generation,
or SER training behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


RANDOM_STATE = 42
SPEAKER_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
EMBEDDING_DIMENSION = 192
COSINE_DISTANCE_THRESHOLD = 0.42
SENSITIVITY_THRESHOLDS = [0.32, 0.37, 0.42, 0.47, 0.52]
EXPECTED_SPLIT_ROWS = {"train": 1313, "validation": 261, "test": 395}
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
STUDY_ROOT = PROJECT_ROOT.parent
FROZEN_OUTPUT = PROJECT_ROOT / "output"
EMBEDDING_DIR = ANALYSIS_DIR / "embeddings"
MODEL_CACHE_DIR = ANALYSIS_DIR / "model_cache" / "spkrec-ecapa-voxceleb"
MANIFEST_PATHS = {
    split: FROZEN_OUTPUT / f"{split}_manifest.csv"
    for split in ("train", "validation", "test")
}
EXISTING_EMBEDDING_METADATA = (
    STUDY_ROOT / "Training Data" / "Auto Split" / "output" / "pipeline_workspace"
    / "speaker_clusters" / "_speaker_embedding_metadata.csv"
)
LOCAL_MODEL_ROOT = (
    STUDY_ROOT / "Training Data" / "Auto Split" / ".cache" / "huggingface" / "hub"
    / "models--speechbrain--spkrec-ecapa-voxceleb" / "snapshots"
)

FINAL_ARTIFACTS = [
    ANALYSIS_DIR / "speaker_metadata.csv",
    ANALYSIS_DIR / "speaker_cluster_summary.csv",
    ANALYSIS_DIR / "speaker_split_overlap.csv",
    ANALYSIS_DIR / "speaker_analysis_config.json",
    ANALYSIS_DIR / "speaker_cluster_size_distribution.csv",
    ANALYSIS_DIR / "similarity_diagnostics.csv",
    ANALYSIS_DIR / "threshold_sensitivity.csv",
    ANALYSIS_DIR / "embedding_index.csv",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("Invalid zero or non-finite speaker embedding")
    return vector / norm


def load_frozen_population() -> pd.DataFrame:
    frames = []
    for split, expected_rows in EXPECTED_SPLIT_ROWS.items():
        path = MANIFEST_PATHS[split]
        if not path.is_file():
            raise FileNotFoundError(f"Frozen manifest missing: {path}")
        frame = pd.read_csv(path, low_memory=False)
        required = {
            "filename", "file_path", "emotion", "source_media", "source_group",
            "split", "sha256",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Frozen {split} manifest lacks columns: {missing}")
        if len(frame) != expected_rows or not frame["split"].astype(str).eq(split).all():
            raise ValueError(f"Frozen {split} membership does not match the canonical split")
        frames.append(frame)

    population = pd.concat(frames, ignore_index=True)
    if len(population) != 1969:
        raise ValueError(f"Expected 1,969 frozen clips; found {len(population):,}")
    if population["file_path"].duplicated().any() or population["sha256"].duplicated().any():
        raise ValueError("Frozen population contains duplicate paths or audio hashes")
    if set(population["emotion"].astype(str)) != set(EMOTIONS):
        raise ValueError("Frozen population emotion labels changed")
    missing_audio = population.loc[
        ~population["file_path"].astype(str).map(lambda value: Path(value).is_file()), "file_path"
    ]
    if len(missing_audio):
        raise FileNotFoundError(f"Frozen audio missing, for example: {missing_audio.iloc[0]}")
    return population.sort_values(["filename", "sha256"], kind="stable").reset_index(drop=True)


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "_-" else "_" for character in value)


def clustering_domain(source_media: str) -> str:
    """Reuse the project's conservative anime/season grouping convention."""
    text = str(source_media or "Unknown_Source")
    text = re.sub(r"^\d+_+", "", text, flags=re.I)
    text = re.sub(r"^(?:Kusonime|Otakudesu(?:_net)?)_+", "", text, flags=re.I)
    text = re.sub(r"_+(?:Vocals?|Voc)$", "", text, flags=re.I)
    text = re.sub(r"__BD_-_\d+.*$", "", text, flags=re.I)
    text = re.sub(r"--(?:\d+|OVA).*?$", "", text, flags=re.I)
    text = re.sub(r"_+(?:720p|1080p).*$", "", text, flags=re.I)
    text = re.sub(r"_+", "_", text).strip("_-")
    return safe_name(text or "Unknown_Source")


def existing_embedding_index() -> dict[str, list[dict[str, str]]]:
    by_filename: dict[str, list[dict[str, str]]] = defaultdict(list)
    if not EXISTING_EMBEDDING_METADATA.is_file():
        return by_filename
    metadata = pd.read_csv(EXISTING_EMBEDDING_METADATA, low_memory=False)
    for _, row in metadata.iterrows():
        source_path = Path(str(row.get("file_path", "")))
        embedding_path = Path(str(row.get("speaker_embedding_path", "")))
        if source_path.name:
            by_filename[source_path.name].append({
                "audio_path": str(source_path),
                "embedding_path": str(embedding_path),
            })
    return by_filename


def valid_embedding(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        vector = np.load(path, allow_pickle=False)
        if vector.size != EMBEDDING_DIMENSION or not np.isfinite(vector).all():
            return None
        return normalize_vector(vector)
    except (OSError, ValueError):
        return None


def find_local_model_snapshot() -> Path | None:
    if not LOCAL_MODEL_ROOT.is_dir():
        return None
    candidates = sorted(
        path for path in LOCAL_MODEL_ROOT.iterdir()
        if path.is_dir() and (path / "hyperparams.yaml").is_file()
    )
    return candidates[-1] if candidates else None


def patch_torchaudio_legacy_backend_api() -> None:
    import torchaudio

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend=None: None  # type: ignore[attr-defined]


class EcapaEmbedder:
    def __init__(self, device: str | None = None):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        patch_torchaudio_legacy_backend_api()
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        snapshot = find_local_model_snapshot()
        self.model_source = str(snapshot) if snapshot else SPEAKER_MODEL_ID
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # The downloaded SpeechBrain YAML points its loadables back to the Hub.
        # Override that reference when a complete local snapshot is available so
        # this analysis neither needs a token nor writes to the user-wide cache.
        overrides = {"pretrained_path": str(snapshot)} if snapshot else None
        self.model = EncoderClassifier.from_hparams(
            source=self.model_source,
            savedir=str(MODEL_CACHE_DIR),
            run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY,
            overrides=overrides,
        )

    @staticmethod
    def load_waveform(path: str) -> np.ndarray:
        import soundfile as sf
        from scipy.signal import resample_poly

        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1, dtype=np.float32)
        if not len(waveform):
            raise ValueError(f"Empty audio file: {path}")
        if int(sample_rate) != 16000:
            divisor = math.gcd(int(sample_rate), 16000)
            waveform = resample_poly(
                waveform, 16000 // divisor, int(sample_rate) // divisor
            ).astype(np.float32)
        return np.asarray(waveform, dtype=np.float32)

    def encode_paths(self, paths: list[str]) -> np.ndarray:
        import torch

        waveforms = [self.load_waveform(path) for path in paths]
        maximum_length = max(len(waveform) for waveform in waveforms)
        batch = np.zeros((len(waveforms), maximum_length), dtype=np.float32)
        relative_lengths = np.zeros(len(waveforms), dtype=np.float32)
        for index, waveform in enumerate(waveforms):
            batch[index, : len(waveform)] = waveform
            relative_lengths[index] = len(waveform) / maximum_length
        with torch.inference_mode():
            embeddings = self.model.encode_batch(
                torch.from_numpy(batch).to(self.device),
                torch.from_numpy(relative_lengths).to(self.device),
            )
        matrix = embeddings.squeeze(1).detach().cpu().numpy().astype(np.float32)
        return np.vstack([normalize_vector(vector) for vector in matrix])


def prepare_embeddings(population: pd.DataFrame, batch_size: int) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    previous = existing_embedding_index()
    vectors: list[np.ndarray | None] = [None] * len(population)
    sources = [""] * len(population)
    missing_indices = []

    for index, row in population.iterrows():
        cache_path = EMBEDDING_DIR / f"{row['sha256']}.npy"
        vector = valid_embedding(cache_path)
        if vector is not None:
            vectors[index] = vector
            sources[index] = "speaker_analysis_cache"
            continue

        reused = None
        candidates = previous.get(str(row["filename"]), [])
        if len(candidates) == 1:
            candidate_audio = Path(candidates[0]["audio_path"])
            candidate_embedding = Path(candidates[0]["embedding_path"])
            if candidate_audio.is_file() and sha256_file(candidate_audio) == str(row["sha256"]):
                reused = valid_embedding(candidate_embedding)
        if reused is not None:
            np.save(cache_path, reused, allow_pickle=False)
            vectors[index] = reused
            sources[index] = "validated_existing_ecapa"
        else:
            missing_indices.append(index)

    embedder = None
    for start in range(0, len(missing_indices), batch_size):
        indices = missing_indices[start : start + batch_size]
        if embedder is None:
            embedder = EcapaEmbedder()
            print(f"ECAPA model loaded from {embedder.model_source} on {embedder.device}", flush=True)
        paths = population.loc[indices, "file_path"].astype(str).tolist()
        try:
            encoded = embedder.encode_paths(paths)
        except Exception:
            # Isolate any problematic clip while retaining the neural-only rule.
            encoded = np.vstack([embedder.encode_paths([path])[0] for path in paths])
        for row_index, vector in zip(indices, encoded):
            cache_path = EMBEDDING_DIR / f"{population.loc[row_index, 'sha256']}.npy"
            np.save(cache_path, vector, allow_pickle=False)
            vectors[row_index] = vector
            sources[row_index] = "new_ecapa"
        completed = min(start + len(indices), len(missing_indices))
        print(f"New ECAPA embeddings {completed}/{len(missing_indices)}", flush=True)

    if any(vector is None for vector in vectors):
        raise RuntimeError("One or more ECAPA embeddings were not produced")
    matrix = np.vstack(vectors).astype(np.float32)
    if matrix.shape != (len(population), EMBEDDING_DIMENSION):
        raise ValueError(f"Unexpected ECAPA matrix shape: {matrix.shape}")

    embedding_index = population[["filename", "file_path", "sha256", "split"]].copy()
    embedding_index["embedding_path"] = population["sha256"].map(
        lambda value: str(EMBEDDING_DIR / f"{value}.npy")
    )
    embedding_index["embedding_source"] = sources
    details = {
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embedding_source_counts": pd.Series(sources).value_counts().to_dict(),
        "model_source": embedder.model_source if embedder else "not_loaded_all_embeddings_reused",
        "device": embedder.device if embedder else "not_loaded_all_embeddings_reused",
    }
    return matrix, embedding_index, details


def local_agglomerative(values: np.ndarray, threshold: float) -> np.ndarray:
    if len(values) == 1:
        return np.zeros(1, dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold,
        compute_full_tree=True,
    )
    return model.fit_predict(values).astype(int)


def cluster_within_domains(
    population: pd.DataFrame, embeddings: np.ndarray, threshold: float
) -> tuple[np.ndarray, dict[str, dict[str, object]]]:
    raw_keys: list[tuple[str, int] | None] = [None] * len(population)
    for domain, indices in population.groupby("clustering_domain", sort=True).groups.items():
        positions = np.asarray(sorted(indices), dtype=int)
        local_labels = local_agglomerative(embeddings[positions], threshold)
        for position, label in zip(positions, local_labels):
            raw_keys[position] = (str(domain), int(label))
    if any(key is None for key in raw_keys):
        raise RuntimeError("Cluster assignment did not cover every clip")

    members: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, key in enumerate(raw_keys):
        members[key].append(index)  # type: ignore[index]
    ordered_keys = sorted(
        members,
        key=lambda key: (
            key[0].lower(),
            min(str(population.loc[index, "sha256"]) for index in members[key]),
        ),
    )
    stable_names = {key: f"SPK_{number:04d}" for number, key in enumerate(ordered_keys, start=1)}
    names = np.asarray([stable_names[key] for key in raw_keys], dtype=object)
    info = {
        stable_names[key]: {"domain": key[0], "indices": members[key]}
        for key in ordered_keys
    }
    return names, info


def similarity_statistics(
    population: pd.DataFrame,
    embeddings: np.ndarray,
    cluster_names: np.ndarray,
    cluster_info: dict[str, dict[str, object]],
) -> pd.DataFrame:
    centroids = {}
    domain_clusters: dict[str, list[str]] = defaultdict(list)
    for name, info in cluster_info.items():
        indices = np.asarray(info["indices"], dtype=int)
        centroids[name] = normalize_vector(embeddings[indices].mean(axis=0))
        domain_clusters[str(info["domain"])].append(name)

    rows = []
    for index, name in enumerate(cluster_names):
        info = cluster_info[str(name)]
        domain = str(info["domain"])
        own_similarity = float(np.clip(embeddings[index] @ centroids[str(name)], -1.0, 1.0))
        other_names = [candidate for candidate in domain_clusters[domain] if candidate != name]
        nearest_other = max(
            (float(np.clip(embeddings[index] @ centroids[candidate], -1.0, 1.0)) for candidate in other_names),
            default=np.nan,
        )
        margin = own_similarity - nearest_other if np.isfinite(nearest_other) else np.nan
        cluster_size = len(info["indices"])
        if cluster_size == 1:
            reliability = "singleton_unverified"
        elif not np.isfinite(margin):
            reliability = "single_cluster_domain_unverified"
        elif margin < 0.05:
            reliability = "low_margin"
        elif margin < 0.10:
            reliability = "moderate_margin"
        else:
            reliability = "higher_margin"
        rows.append({
            "centroid_cosine_similarity": own_similarity,
            "nearest_other_centroid_similarity": nearest_other,
            "similarity_margin": margin,
            "assignment_reliability": reliability,
            "cluster_size": cluster_size,
        })
    return pd.DataFrame(rows)


def overlap_metrics(metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    cluster_splits = metadata.groupby("anonymous_speaker_cluster")["split"].agg(lambda values: frozenset(values))
    cluster_members = metadata.groupby("anonymous_speaker_cluster").groups

    definitions = {
        "train_only": lambda values: values == {"train"},
        "validation_only": lambda values: values == {"validation"},
        "test_only": lambda values: values == {"test"},
        "train_validation_only": lambda values: values == {"train", "validation"},
        "train_test_only": lambda values: values == {"train", "test"},
        "validation_test_only": lambda values: values == {"validation", "test"},
        "all_three": lambda values: values == {"train", "validation", "test"},
        "cross_train_validation_inclusive": lambda values: {"train", "validation"}.issubset(values),
        "cross_train_test_inclusive": lambda values: {"train", "test"}.issubset(values),
        "cross_validation_test_inclusive": lambda values: {"validation", "test"}.issubset(values),
        "any_cross_split": lambda values: len(values) >= 2,
    }
    notes = {
        "train_only": "mutually exclusive split-presence category",
        "validation_only": "mutually exclusive split-presence category",
        "test_only": "mutually exclusive split-presence category",
        "train_validation_only": "crosses train/validation but not test",
        "train_test_only": "crosses train/test but not validation",
        "validation_test_only": "crosses validation/test but not train",
        "all_three": "appears in train, validation, and test",
        "cross_train_validation_inclusive": "includes all-three clusters",
        "cross_train_test_inclusive": "includes all-three clusters",
        "cross_validation_test_inclusive": "includes all-three clusters",
        "any_cross_split": "unique clusters and clips appearing in at least two splits",
    }
    rows = []
    selected_by_metric = {}
    for metric, predicate in definitions.items():
        selected = [name for name, splits in cluster_splits.items() if predicate(set(splits))]
        clip_indices = sorted({index for name in selected for index in cluster_members[name]})
        selected_by_metric[metric] = selected
        rows.append({
            "overlap_category": metric,
            "cluster_count": len(selected),
            "unique_clip_count": len(clip_indices),
            "clip_percentage_of_all": 100.0 * len(clip_indices) / len(metadata),
            "definition": notes[metric],
        })
    cross_names = selected_by_metric["any_cross_split"]
    return pd.DataFrame(rows), {
        "cross_split_cluster_count": len(cross_names),
        "cross_split_clip_count": int(metadata["anonymous_speaker_cluster"].isin(cross_names).sum()),
        "cross_split_clip_percentage": float(
            100.0 * metadata["anonymous_speaker_cluster"].isin(cross_names).mean()
        ),
    }


def build_cluster_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster, group in metadata.groupby("anonymous_speaker_cluster", sort=True):
        splits = sorted(group["split"].unique())
        rows.append({
            "anonymous_speaker_cluster": cluster,
            "clustering_domain": group["clustering_domain"].iloc[0],
            "clip_count": len(group),
            "train_clips": int(group["split"].eq("train").sum()),
            "validation_clips": int(group["split"].eq("validation").sum()),
            "test_clips": int(group["split"].eq("test").sum()),
            "split_count": len(splits),
            "splits_present": "|".join(splits),
            "emotion_count": group["emotion"].nunique(),
            "mean_centroid_cosine_similarity": group["centroid_cosine_similarity"].mean(),
            "median_centroid_cosine_similarity": group["centroid_cosine_similarity"].median(),
            "minimum_centroid_cosine_similarity": group["centroid_cosine_similarity"].min(),
            "mean_similarity_margin": group["similarity_margin"].mean(),
        })
    return pd.DataFrame(rows).sort_values(
        ["clip_count", "anonymous_speaker_cluster"], ascending=[False, True]
    ).reset_index(drop=True)


def threshold_sensitivity(population: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in SENSITIVITY_THRESHOLDS:
        names, _ = cluster_within_domains(population, embeddings, threshold)
        frame = population[["split"]].copy()
        frame["anonymous_speaker_cluster"] = names
        overlap, overall = overlap_metrics(frame)
        sizes = frame["anonymous_speaker_cluster"].value_counts()
        rows.append({
            "cosine_distance_threshold": threshold,
            "estimated_cluster_count": len(sizes),
            "singleton_cluster_count": int(sizes.eq(1).sum()),
            "singleton_clip_percentage": 100.0 * sizes.eq(1).sum() / len(frame),
            "cross_split_cluster_count": overall["cross_split_cluster_count"],
            "cross_split_clip_count": overall["cross_split_clip_count"],
            "cross_split_clip_percentage": overall["cross_split_clip_percentage"],
        })
    return pd.DataFrame(rows)


def package_versions() -> dict[str, str]:
    result = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "scikit-learn", "torch", "torchaudio", "speechbrain"):
        result[package] = importlib.metadata.version(package)
    return result


def run(batch_size: int) -> None:
    existing = [str(path) for path in FINAL_ARTIFACTS if path.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite completed speaker-analysis artifact(s): {existing}"
        )
    population = load_frozen_population()
    population["clustering_domain"] = population["source_media"].fillna("").map(clustering_domain)

    embeddings, embedding_index, embedding_details = prepare_embeddings(population, batch_size)
    names, cluster_info = cluster_within_domains(
        population, embeddings, COSINE_DISTANCE_THRESHOLD
    )
    similarities = similarity_statistics(population, embeddings, names, cluster_info)

    metadata = population[
        ["filename", "file_path", "emotion", "source_media", "source_group", "split", "sha256", "clustering_domain"]
    ].copy()
    metadata["anonymous_speaker_cluster"] = names
    metadata = pd.concat([metadata, similarities], axis=1)
    metadata["embedding_path"] = embedding_index["embedding_path"]
    metadata["embedding_source"] = embedding_index["embedding_source"]

    cluster_summary = build_cluster_summary(metadata)
    overlap, overlap_overall = overlap_metrics(metadata)
    size_distribution = (
        cluster_summary.groupby("clip_count").size().rename("cluster_count").reset_index()
    )
    size_distribution["clips_represented"] = (
        size_distribution["clip_count"] * size_distribution["cluster_count"]
    )
    sensitivity = threshold_sensitivity(population, embeddings)

    diagnostic_rows = []
    for column in ("centroid_cosine_similarity", "similarity_margin"):
        values = pd.to_numeric(metadata[column], errors="coerce").dropna()
        for statistic, value in {
            "minimum": values.min(),
            "q05": values.quantile(0.05),
            "q25": values.quantile(0.25),
            "median": values.median(),
            "q75": values.quantile(0.75),
            "q95": values.quantile(0.95),
            "maximum": values.max(),
            "mean": values.mean(),
        }.items():
            diagnostic_rows.append({"metric": column, "statistic": statistic, "value": value})
    reliability_counts = metadata["assignment_reliability"].value_counts()
    for status, count in reliability_counts.items():
        diagnostic_rows.append({
            "metric": "assignment_reliability_count", "statistic": status, "value": int(count)
        })
    diagnostics = pd.DataFrame(diagnostic_rows)

    if len(metadata) != 1969 or metadata["file_path"].duplicated().any():
        raise AssertionError("Speaker metadata does not represent all 1,969 clips exactly once")
    reference = population.set_index("sha256")[["file_path", "emotion", "split"]].sort_index()
    produced = metadata.set_index("sha256")[["file_path", "emotion", "split"]].sort_index()
    if not reference.equals(produced):
        raise AssertionError("Frozen paths, emotion labels, or split membership changed")
    if int(cluster_summary["clip_count"].sum()) != 1969:
        raise AssertionError("Cluster counts do not sum to 1,969")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(FINAL_ARTIFACTS[0], index=False, encoding="utf-8-sig")
    cluster_summary.to_csv(FINAL_ARTIFACTS[1], index=False, encoding="utf-8-sig")
    overlap.to_csv(FINAL_ARTIFACTS[2], index=False, encoding="utf-8-sig")
    size_distribution.to_csv(FINAL_ARTIFACTS[4], index=False, encoding="utf-8-sig")
    diagnostics.to_csv(FINAL_ARTIFACTS[5], index=False, encoding="utf-8-sig")
    sensitivity.to_csv(FINAL_ARTIFACTS[6], index=False, encoding="utf-8-sig")
    embedding_index.to_csv(FINAL_ARTIFACTS[7], index=False, encoding="utf-8-sig")

    singleton_clusters = int(cluster_summary["clip_count"].eq(1).sum())
    singleton_cluster_share = singleton_clusters / len(cluster_summary)
    reliability_gate_threshold = 0.25
    reliability_gate_passed = singleton_cluster_share <= reliability_gate_threshold
    config = {
        "analysis_name": "anonymous_speaker_cluster_metadata",
        "status": (
            "completed_exploratory_estimated_identity_analysis"
            if reliability_gate_passed
            else "completed_exploratory_low_reliability"
        ),
        "interpretation": "source-media/episode group-disjoint; not speaker-independent",
        "identity_type": "anonymous estimated voice-similarity cluster; not character recognition",
        "reliable_speaker_identity_assignment": reliability_gate_passed,
        "quality_gate": {
            "metric": "singleton cluster share",
            "maximum_allowed": reliability_gate_threshold,
            "observed": singleton_cluster_share,
            "passed": reliability_gate_passed,
            "consequence_if_failed": "retain IDs only as exploratory similarity groups; do not treat the cluster count or overlap as verified speaker identity",
        },
        "speaker_model": SPEAKER_MODEL_ID,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embedding_details": embedding_details,
        "clustering": {
            "algorithm": "agglomerative",
            "metric": "cosine",
            "linkage": "average",
            "cosine_distance_threshold": COSINE_DISTANCE_THRESHOLD,
            "number_of_clusters_forced": False,
            "scope": "within conservatively derived anime/season domain",
            "random_state": RANDOM_STATE,
        },
        "frozen_split_rows": EXPECTED_SPLIT_ROWS,
        "estimated_cluster_count": int(len(cluster_summary)),
        "singleton_cluster_count": singleton_clusters,
        "singleton_cluster_percentage": 100.0 * singleton_cluster_share,
        **overlap_overall,
        "source_manifests": {
            split: {"path": str(path), "sha256": sha256_file(path)}
            for split, path in MANIFEST_PATHS.items()
        },
        "existing_capability_inspected": {
            "code": str(STUDY_ROOT / "Training Data" / "Auto Split" / "speaker_embedding_cluster.py"),
            "reused": "ECAPA model, 192-D normalized embeddings, anime/season domain convention, cosine average-linkage threshold 0.42",
            "not_reused": "destructive output rebuild, audio copying, acoustic fallback, KMeans speaker-count heuristic, character-map generation",
        },
        "limitations": [
            "No independent speaker ground-truth labels are available for clustering validation.",
            "The VoxCeleb-trained ECAPA model is out of domain for expressive anime-style Japanese acting.",
            "Short clips, emotion-dependent voice changes, music, and effects may split one speaker or merge different speakers.",
            "Clustering is restricted within derived anime/season domains, so cross-domain same-voice overlap is not detected.",
            "Centroid similarity and margins are diagnostics, not calibrated identity probabilities.",
        ],
        "software_versions": package_versions(),
    }
    config["artifact_sha256"] = {
        path.name: sha256_file(path) for path in FINAL_ARTIFACTS if path.exists() and path != FINAL_ARTIFACTS[3]
    }
    FINAL_ARTIFACTS[3].write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "clips": len(metadata),
        "estimated_clusters": len(cluster_summary),
        "singleton_clusters": singleton_clusters,
        **overlap_overall,
        "embedding_source_counts": embedding_details["embedding_source_counts"],
    }, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=24)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.batch_size < 1:
        raise ValueError("batch-size must be positive")
    run(arguments.batch_size)
