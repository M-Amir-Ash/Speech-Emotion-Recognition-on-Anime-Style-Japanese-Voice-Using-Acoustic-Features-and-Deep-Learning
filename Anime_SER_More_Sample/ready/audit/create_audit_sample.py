"""Create a deterministic, non-destructive human label-audit sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "pleasant_surprise",
    "sad",
]
DEFAULT_AUDIT_SEED = 20260905
DEFAULT_SAMPLES_PER_EMOTION = 15
REVIEW_FIELDS = ["reviewer_emotion", "reviewer_uncertain", "reviewer_notes"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_population(project_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    output_dir = project_root / "output"
    manifest_paths = [output_dir / f"{split}_manifest.csv" for split in ("train", "validation", "test")]
    missing = [str(path) for path in manifest_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen split manifest(s) missing: {missing}")

    frames = []
    for expected_split, path in zip(("train", "validation", "test"), manifest_paths):
        frame = pd.read_csv(path, low_memory=False)
        if "split" in frame.columns and not frame["split"].astype(str).eq(expected_split).all():
            raise ValueError(f"Unexpected split value in {path}")
        frames.append(frame)

    population = pd.concat(frames, ignore_index=True)
    required = {"file_path", "filename", "emotion", "source_media", "duration_sec", "transcript"}
    missing_columns = sorted(required - set(population.columns))
    if missing_columns:
        raise ValueError(f"Frozen manifests lack required columns: {missing_columns}")
    if len(population) != 1969:
        raise ValueError(f"Expected frozen population of 1,969 rows; found {len(population):,}")
    if population["file_path"].duplicated().any():
        raise ValueError("Frozen manifests contain duplicate file paths")
    if "sha256" in population and population["sha256"].duplicated().any():
        raise ValueError("Frozen manifests contain duplicate audio hashes")
    if set(population["emotion"].astype(str)) != set(EMOTIONS):
        raise ValueError("Frozen manifest emotion labels do not match the required seven classes")

    missing_audio = [path for path in population["file_path"].astype(str) if not Path(path).is_file()]
    if missing_audio:
        raise FileNotFoundError(f"Audit source audio is missing, for example: {missing_audio[0]}")
    return population, manifest_paths


def create_audit_rows(population: pd.DataFrame, seed: int, samples_per_emotion: int) -> pd.DataFrame:
    if samples_per_emotion < 1:
        raise ValueError("samples_per_emotion must be positive")

    rng = np.random.default_rng(seed)
    selected_parts = []
    for emotion in EMOTIONS:
        emotion_rows = population[population["emotion"].astype(str).eq(emotion)]
        if len(emotion_rows) < samples_per_emotion:
            raise ValueError(
                f"Cannot select {samples_per_emotion} rows for {emotion}; only {len(emotion_rows)} available"
            )
        chosen_positions = rng.choice(len(emotion_rows), size=samples_per_emotion, replace=False)
        selected_parts.append(emotion_rows.iloc[chosen_positions].copy())

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)
    selected.insert(0, "audit_id", [f"AUDIT_{index:03d}" for index in range(1, len(selected) + 1)])

    audit = pd.DataFrame({
        "audit_id": selected["audit_id"],
        "file_path": selected["file_path"].astype(str),
        "filename": selected["filename"].astype(str),
        "original_emotion": selected["emotion"].astype(str),
        "source_media": selected["source_media"].fillna("").astype(str),
        "duration_sec": pd.to_numeric(selected["duration_sec"], errors="raise").round(6),
        "transcript": selected["transcript"].fillna("").astype(str),
        "reviewer_emotion": "",
        "reviewer_uncertain": "",
        "reviewer_notes": "",
        "agreement": "",
    })
    return audit


def write_csv_safely(frame: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing review file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_AUDIT_SEED)
    parser.add_argument("--samples-per-emotion", type=int, default=DEFAULT_SAMPLES_PER_EMOTION)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing blank audit files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_dir = Path(__file__).resolve().parent
    project_root = audit_dir.parent
    population, manifest_paths = load_frozen_population(project_root)
    audit = create_audit_rows(population, args.seed, args.samples_per_emotion)

    expected_total = len(EMOTIONS) * args.samples_per_emotion
    counts = audit["original_emotion"].value_counts().reindex(EMOTIONS, fill_value=0)
    if len(audit) != expected_total or not counts.eq(args.samples_per_emotion).all():
        raise AssertionError("Generated audit is not balanced at the requested per-emotion count")
    if audit["file_path"].duplicated().any():
        raise AssertionError("Generated audit contains duplicate audio paths")
    if audit[REVIEW_FIELDS + ["agreement"]].fillna("").ne("").any().any():
        raise AssertionError("Human-review fields must be blank in a newly generated audit")

    master_path = audit_dir / "audit_samples.csv"
    review_form_path = audit_dir / "audit_review_form.csv"
    config_path = audit_dir / "audit_config.json"
    if not args.overwrite:
        existing_targets = [str(path) for path in (master_path, review_form_path, config_path) if path.exists()]
        if existing_targets:
            raise FileExistsError(f"Refusing to overwrite existing audit artifact(s): {existing_targets}")

    write_csv_safely(audit, master_path, args.overwrite)

    # This optional form hides the original pseudo-label to reduce reviewer anchoring.
    # It also contains no model prediction or suggested alternative label.
    review_form = audit.drop(columns=["original_emotion", "agreement"])
    write_csv_safely(review_form, review_form_path, args.overwrite)

    config = {
        "audit_seed": args.seed,
        "samples_per_emotion": args.samples_per_emotion,
        "total_samples": len(audit),
        "emotion_order": EMOTIONS,
        "sampling_population_rows": len(population),
        "sampling_method": "without replacement within each emotion, followed by deterministic global shuffle",
        "source_manifests": [
            {"path": str(path), "sha256": sha256_file(path)} for path in manifest_paths
        ],
        "class_counts": {emotion: int(counts[emotion]) for emotion in EMOTIONS},
        "reviewer_form_is_blinded_to_original_emotion": True,
        "alternative_model_predictions_included": False,
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Created {master_path}")
    print(f"Created blinded reviewer form {review_form_path}")
    print(f"Selected {len(audit)} clips: {args.samples_per_emotion} per emotion, seed={args.seed}")


if __name__ == "__main__":
    main()
