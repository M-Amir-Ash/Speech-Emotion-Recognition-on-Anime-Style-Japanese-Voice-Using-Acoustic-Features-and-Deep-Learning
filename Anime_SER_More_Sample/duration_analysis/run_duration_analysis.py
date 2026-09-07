"""Improvement 6: read-only duration-policy analysis of the frozen dataset."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
FROZEN_OUTPUT = ROOT / "output"
ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_PARENT = ROOT.parent
UPSTREAM = PROJECT_PARENT / "Training Data" / "Auto Split More Sample"
PROPOSAL = Path(
    "D:/Telkom University Kuliah/Kuliah S1/Tugas S1/Semester 6/Proposal/"
    "Speech_Emotion_Recognition_on_Anime_Style_Japanese_Voice_Using_Acoustic_Features_and_Deep_Learning_"
    "Proposal_Thesis_FIF_Telkom_University__Tel_U____PHN.pdf"
)

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]
EXPECTED_SPLITS = {"train": 1313, "validation": 261, "test": 395}
EXPECTED_TOTAL = 1969
PLANNED_MIN_SECONDS = 1.0
PLANNED_MAX_SECONDS = 6.0
IMPLEMENTED_MIN_SECONDS = 0.65
PROPOSAL_DURATION_SENTENCE = "The planned duration of each audio segment is approximately 1 to 6 seconds."
PROPOSAL_ADJUSTMENT_SENTENCE = (
    "During the implementation stage, the final duration range may be adjusted "
    "on the basis of the quality of the collected segments."
)
RECOMMENDED_WORDING = (
    "Most selected utterances (92.5%) were within the planned 1–6 s duration interval. "
    "A small number of shorter utterances were retained when they satisfied the remaining "
    "Japanese-dialogue, audio-quality, and clean-selection criteria."
)


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


def load_frozen_manifests() -> pd.DataFrame:
    frames = []
    required = {
        "file_path", "filename", "emotion", "label_id", "sha256", "split",
        "duration_sec", "start_sec", "end_sec",
    }
    for split, expected_rows in EXPECTED_SPLITS.items():
        path = FROZEN_OUTPUT / f"{split}_manifest.csv"
        frame = pd.read_csv(path, low_memory=False)
        if len(frame) != expected_rows:
            raise RuntimeError(f"Frozen {split} count changed: {len(frame)} != {expected_rows}")
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(f"Frozen {split} manifest lacks {sorted(missing)}")
        if set(frame["split"].astype(str)) != {split}:
            raise RuntimeError(f"Unexpected split membership values in {path}")
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    if len(data) != EXPECTED_TOTAL or data["file_path"].duplicated().any():
        raise RuntimeError("Frozen manifests do not contain exactly 1,969 unique paths")
    if data["sha256"].duplicated().any():
        raise RuntimeError("Frozen manifests contain duplicate SHA-256 values")
    if set(data["emotion"]) != set(EMOTIONS):
        raise RuntimeError("Frozen emotion set changed")
    mapping = dict(zip(EMOTIONS, range(len(EMOTIONS))))
    expected_labels = data["emotion"].map(mapping)
    if not np.array_equal(expected_labels.to_numpy(int), data["label_id"].to_numpy(int)):
        raise RuntimeError("Frozen emotion labels changed")
    if any(not Path(path).is_file() for path in data["file_path"].astype(str)):
        raise RuntimeError("One or more frozen audio files are missing")

    data["duration_sec"] = pd.to_numeric(data["duration_sec"], errors="raise")
    data["start_sec"] = pd.to_numeric(data["start_sec"], errors="raise")
    data["end_sec"] = pd.to_numeric(data["end_sec"], errors="raise")
    if not np.isfinite(data["duration_sec"]).all() or (data["duration_sec"] <= 0).any():
        raise RuntimeError("Invalid duration values in frozen manifests")
    # The upstream manifest rounds these fields independently to milliseconds.
    timestamp_error = np.abs(data["duration_sec"] - (data["end_sec"] - data["start_sec"]))
    if float(timestamp_error.max()) > 0.0011:
        raise RuntimeError(f"Duration/timestamp discrepancy exceeds rounding tolerance: {timestamp_error.max()}")
    return data


def inspect_sources() -> dict:
    extraction_path = UPSTREAM / "extract_segment_multimodal_pipeline.py"
    runner_path = UPSTREAM / "emotion_workflow_runner.py"
    selector_path = UPSTREAM / "cluster_clean_target_selector.py"
    for path in (extraction_path, runner_path, selector_path, PROPOSAL):
        if not path.is_file():
            raise FileNotFoundError(path)

    extraction = extraction_path.read_text(encoding="utf-8")
    runner = runner_path.read_text(encoding="utf-8")
    selector = selector_path.read_text(encoding="utf-8")
    required_patterns = {
        "MIN_SEGMENT_SEC": (extraction, r"MIN_SEGMENT_SEC\s*=\s*0\.65"),
        "MIN_EMOTION_CLIP_SEC": (extraction, r"MIN_EMOTION_CLIP_SEC\s*=\s*0\.65"),
        "emotion_duration_gate": (extraction, r"duration_sec.*>=\s*min_emotion_clip_sec"),
        "runner_default": (runner, r"--min-emotion-clip-sec[\s\S]{0,100}default=0\.65"),
        "selector_quality_gate": (selector, r"audio_quality_status.*!=\s*[\"']pass[\"']"),
    }
    missing_patterns = [name for name, (text, pattern) in required_patterns.items() if not re.search(pattern, text)]
    if missing_patterns:
        raise RuntimeError(f"Expected upstream duration evidence changed: {missing_patterns}")

    pages = PdfReader(str(PROPOSAL)).pages
    proposal_page_number = None
    proposal_text = ""
    for index, page in enumerate(pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        if PROPOSAL_DURATION_SENTENCE in text:
            proposal_page_number = index
            proposal_text = text
            break
    if proposal_page_number is None or PROPOSAL_ADJUSTMENT_SENTENCE not in proposal_text:
        raise RuntimeError("Exact proposal duration statements were not found")

    return {
        "proposal_pdf": str(PROPOSAL),
        "proposal_pdf_sha256": sha256_file(PROPOSAL),
        "proposal_pdf_page_index_1_based": proposal_page_number,
        "proposal_printed_page": 22,
        "proposal_exact_duration_statement": PROPOSAL_DURATION_SENTENCE,
        "proposal_exact_adjustment_statement": PROPOSAL_ADJUSTMENT_SENTENCE,
        "upstream_extraction_script": str(extraction_path),
        "upstream_extraction_sha256": sha256_file(extraction_path),
        "upstream_workflow_runner": str(runner_path),
        "upstream_workflow_runner_sha256": sha256_file(runner_path),
        "upstream_clean_selector": str(selector_path),
        "upstream_clean_selector_sha256": sha256_file(selector_path),
        "implemented_segmentation_min_seconds": IMPLEMENTED_MIN_SECONDS,
        "implemented_emotion_clip_min_seconds": IMPLEMENTED_MIN_SECONDS,
        "implementation_interpretation": (
            "Clips below 1.00 second were intentionally permitted when duration was at least "
            "0.65 second and the Japanese-dialogue, quality, and later clean-selection gates passed."
        ),
        "final_clean_selector_duration_behavior": (
            "Carries duration_sec as metadata and requires audio_quality_status=pass, but does not "
            "apply a new 1.00-second exclusion threshold."
        ),
    }


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


def main() -> None:
    config_path = ANALYSIS_DIR / "duration_analysis_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            raise RuntimeError("Duration analysis is already complete; refusing to overwrite it")

    data = load_frozen_manifests()
    evidence = inspect_sources()
    durations = data["duration_sec"]
    overall = pd.DataFrame([{
        "count": len(durations),
        "minimum_seconds": durations.min(),
        "maximum_seconds": durations.max(),
        "mean_seconds": durations.mean(),
        "median_seconds": durations.median(),
        "standard_deviation_seconds": durations.std(ddof=0),
        "standard_deviation_definition": "population_ddof_0",
        "percentile_25_seconds": durations.quantile(0.25),
        "percentile_75_seconds": durations.quantile(0.75),
    }])

    data["duration_policy_category"] = np.select(
        [durations < PLANNED_MIN_SECONDS, durations <= PLANNED_MAX_SECONDS],
        ["below_1.00_second", "within_1.00_to_6.00_seconds"],
        default="above_6.00_seconds",
    )
    category_order = ["below_1.00_second", "within_1.00_to_6.00_seconds", "above_6.00_seconds"]
    counts = data["duration_policy_category"].value_counts().reindex(category_order, fill_value=0)
    policy = pd.DataFrame({
        "duration_policy_category": category_order,
        "sample_count": counts.to_numpy(int),
        "percentage_of_dataset": counts.to_numpy(int) / EXPECTED_TOTAL * 100.0,
    })

    rows = []
    overall_short_rate = float((durations < 1.0).mean())
    for emotion in EMOTIONS:
        group = data.loc[data["emotion"] == emotion, "duration_sec"]
        short_count = int((group < 1.0).sum())
        rows.append({
            "emotion": emotion,
            "sample_count": len(group),
            "minimum_seconds": group.min(),
            "maximum_seconds": group.max(),
            "mean_seconds": group.mean(),
            "median_seconds": group.median(),
            "below_1_second_count": short_count,
            "below_1_second_within_class_percentage": short_count / len(group) * 100.0,
            "share_of_all_below_1_second_clips_percentage": short_count / max(1, int((durations < 1.0).sum())) * 100.0,
            "short_clip_rate_ratio_vs_dataset": (short_count / len(group)) / overall_short_rate,
        })
    by_emotion = pd.DataFrame(rows)

    if int(policy.loc[policy["duration_policy_category"] == "below_1.00_second", "sample_count"].iloc[0]) != 148:
        raise RuntimeError("Expected 148 sub-one-second clips but frozen manifests disagree")
    if int(policy.loc[policy["duration_policy_category"] == "within_1.00_to_6.00_seconds", "sample_count"].iloc[0]) != 1821:
        raise RuntimeError("Expected 1,821 clips in the planned interval but manifests disagree")
    if int(policy["sample_count"].sum()) != EXPECTED_TOTAL or int(by_emotion["sample_count"].sum()) != EXPECTED_TOTAL:
        raise RuntimeError("Duration category or emotion totals do not reconcile")
    if int(by_emotion["below_1_second_count"].sum()) != 148:
        raise RuntimeError("Per-emotion short-clip totals do not reconcile")

    distribution = data[[
        "filename", "file_path", "sha256", "emotion", "label_id", "split",
        "source_media", "source_group", "start_sec", "end_sec", "duration_sec",
        "duration_policy_category",
    ]].sort_values(["emotion", "duration_sec", "filename"])
    overall.to_csv(ANALYSIS_DIR / "duration_summary.csv", index=False, encoding="utf-8-sig")
    by_emotion.to_csv(ANALYSIS_DIR / "duration_by_emotion.csv", index=False, encoding="utf-8-sig")
    policy.to_csv(ANALYSIS_DIR / "duration_policy_counts.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(ANALYSIS_DIR / "duration_distribution.csv", index=False, encoding="utf-8-sig")

    highest = by_emotion.sort_values("below_1_second_within_class_percentage", ascending=False).iloc[0]
    comparison = f"""# Proposal Duration Comparison

## Exact proposal wording

Proposal PDF page {evidence['proposal_pdf_page_index_1_based']} (printed page 22):

> {PROPOSAL_DURATION_SENTENCE}

The proposal also states:

> {PROPOSAL_ADJUSTMENT_SENTENCE}

## Actual implementation behavior

The upstream segmentation pipeline defines both `MIN_SEGMENT_SEC` and
`MIN_EMOTION_CLIP_SEC` as **0.65 seconds**. The full workflow exposes the same
0.65-second default. A clip can enter Japanese emotion classification when its
duration is at least that configured value and the Japanese-language, speech,
music/noise, and audio-quality gates pass. The final clean-cluster selector
requires passed audio quality and acoustic separability but does not add a
1.00-second exclusion.

Therefore, 1 second was a planning guideline rather than a strict technical
threshold. Clips from 0.65 through 0.99 seconds were intentionally allowed by
the implemented policy.

## Frozen dataset behavior

- 1,821/1,969 clips ({1821 / 1969 * 100:.3f}%) are within 1.00–6.00 seconds.
- 148/1,969 clips ({148 / 1969 * 100:.3f}%) are below 1.00 second.
- No selected clip exceeds 6.00 seconds.
- The actual range is {durations.min():.3f}–{durations.max():.3f} seconds.
- The highest within-class short-clip rate is {highest['emotion']} at
  {int(highest['below_1_second_count'])}/{int(highest['sample_count'])}
  ({highest['below_1_second_within_class_percentage']:.3f}%).

The short clips are not evenly distributed across emotions. Sad (40), Neutral
(34), and Disgust (33) together account for 107/148 (72.297%) of all clips below
one second. Relative to each class size, Sad (25.974%) and Disgust (22.148%) are
the clearest concentrations, compared with the dataset-wide short-clip rate of
7.517%.

## Classification

**Documented implementation refinement**, not an implementation bug.

This classification is supported because the proposal describes 1–6 seconds as
an approximate planned range and explicitly permits the final range to be adjusted
based on segment quality. The implemented 0.65-second minimum is explicit and
consistently passed through the workflow. The deviation should nevertheless be
reported transparently in the thesis rather than describing the realized dataset
as strictly 1–6 seconds.

## Recommended paper wording

> {RECOMMENDED_WORDING}
"""
    (ANALYSIS_DIR / "proposal_duration_comparison.md").write_text(comparison, encoding="utf-8")

    config = {
        "analysis": "Improvement 6 - Duration-Policy Analysis and Reporting",
        "status": "complete",
        "analysis_only": True,
        "models_trained": False,
        "dataset_rows": EXPECTED_TOTAL,
        "manifest_rows": EXPECTED_SPLITS,
        "duration_source": "duration_sec in frozen train/validation/test manifests",
        "duration_categories": {
            "below": "duration_sec < 1.00",
            "planned_interval": "1.00 <= duration_sec <= 6.00",
            "above": "duration_sec > 6.00",
        },
        "standard_deviation_definition": "population standard deviation (ddof=0)",
        "classification": "documented implementation refinement",
        "recommended_paper_wording": RECOMMENDED_WORDING,
        "canonical_manifest_sha256": {
            split: sha256_file(FROZEN_OUTPUT / f"{split}_manifest.csv")
            for split in EXPECTED_SPLITS
        },
        "source_evidence": evidence,
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
        "validation": {
            "exactly_1969_unique_paths": True,
            "exactly_1969_unique_sha256": True,
            "duration_rows_complete": True,
            "duration_timestamp_max_absolute_error_seconds": float(
                np.abs(data["duration_sec"] - (data["end_sec"] - data["start_sec"])).max()
            ),
            "emotion_counts_reconciled": True,
            "split_membership_read_only": True,
            "files_removed": 0,
            "labels_modified": 0,
            "models_trained": 0,
        },
    }
    atomic_json(config_path, config)

    readme = f"""# Duration-Policy Analysis

This directory contains Improvement 6, a read-only analysis of all 1,969 frozen
clips. No clips, labels, splits, preprocessing, or model artifacts were changed.

## Overall statistics

{markdown_table(overall)}

## Duration-policy categories

{markdown_table(policy)}

## Per-emotion statistics

{markdown_table(by_emotion)}

## Conclusion

Short clips are concentrated most strongly in Sad (40/154; 25.974%) and Disgust
(33/149; 22.148%). Sad, Neutral, and Disgust together contribute 107/148
(72.297%) of all sub-one-second clips.

The discrepancy is a **documented implementation refinement**, not a bug. The
proposal allowed its approximate planned range to be adjusted based on segment
quality, and the pipeline intentionally implemented a 0.65-second minimum.

Recommended wording:

> {RECOMMENDED_WORDING}

See `proposal_duration_comparison.md` for the source-level evidence and exact
proposal comparison. `duration_distribution.csv` preserves the row-level audit.
"""
    (ANALYSIS_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(overall.to_string(index=False))
    print(policy.to_string(index=False))
    print(by_emotion.to_string(index=False))
    print("Improvement 6 complete. No models were trained.")


if __name__ == "__main__":
    main()
