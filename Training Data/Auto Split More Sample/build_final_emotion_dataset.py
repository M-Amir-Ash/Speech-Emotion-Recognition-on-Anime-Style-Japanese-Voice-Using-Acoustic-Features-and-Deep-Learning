import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = BASE_DIR / "output" / "pipeline_workspace"
DEFAULT_OUTPUT = BASE_DIR / "output" / "final_character_emotion_dataset"
CSV_ENCODING = "utf-8-sig"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant_Surprise", "Sad"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def materialize_file(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_unique(src: Path, dst_dir: Path) -> Path:
    ensure_dir(dst_dir)
    candidate = dst_dir / src.name
    if not candidate.exists():
        materialize_file(src, candidate)
        return candidate
    idx = 2
    while True:
        candidate = dst_dir / f"{src.stem}__dup{idx}{src.suffix}"
        if not candidate.exists():
            materialize_file(src, candidate)
            return candidate
        idx += 1


def run_builder(args: argparse.Namespace) -> int:
    workspace = Path(args.output_root or DEFAULT_WORKSPACE).resolve()
    source_output_name = str(getattr(args, "source_output_name", "character_emotion_classified") or "character_emotion_classified")
    manifest_path = workspace / source_output_name / "_character_emotion_manifest.csv"
    final_root = Path(args.final_output or DEFAULT_OUTPUT).resolve()
    if not manifest_path.is_file():
        print(f"Character emotion manifest not found: {manifest_path}")
        return 1

    if final_root.exists():
        # Keep old reviewed/final data safe by moving it aside instead of deleting.
        idx = 1
        while True:
            backup = final_root.with_name(f"{final_root.name}_backup_{idx}")
            if not backup.exists():
                shutil.move(str(final_root), str(backup))
                print(f"Existing final dataset moved to: {backup}")
                break
            idx += 1
    ensure_dir(final_root)
    for label in EMOTIONS:
        ensure_dir(final_root / label)

    df = pd.read_csv(manifest_path)
    rows: List[dict] = []
    for _, row in df.iterrows():
        if str(row.get("confidence_status", "")) != "auto_accept":
            continue
        label = str(row.get("top_label", ""))
        if label not in EMOTIONS:
            continue
        source = Path(str(row.get("output_file", "")))
        if not source.is_file():
            source = Path(str(row.get("source_file", "")))
        if not source.is_file():
            continue
        dst = copy_unique(source, final_root / label)
        rows.append(
            {
                "clip_id": row.get("clip_id", source.stem),
                "file_path": str(dst),
                "character_name": row.get("character_name", ""),
                "speaker_cluster": row.get("speaker_cluster", ""),
                "emotion_label": label,
                "label_source": "auto_high_confidence",
                "confidence": row.get("top_score", ""),
                "source_media": row.get("source_media", ""),
                "start_sec": row.get("start_sec", ""),
                "end_sec": row.get("end_sec", ""),
                "transcript": row.get("transcript", ""),
            }
        )

    fields = [
        "clip_id",
        "file_path",
        "character_name",
        "speaker_cluster",
        "emotion_label",
        "label_source",
        "confidence",
        "source_media",
        "start_sec",
        "end_sec",
        "transcript",
    ]
    out_manifest = final_root / "final_dataset_manifest.csv"
    with out_manifest.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in rows])

    print(f"Final dataset saved: {final_root}")
    print(f"Final dataset manifest saved: {out_manifest}")
    print(f"Final clips: {len(rows)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build final emotion dataset from accepted labels.")
    parser.add_argument("--output-root", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--final-output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--source-output-name", default="character_emotion_classified")
    parser.add_argument("--auto-only", action="store_true", help="Kept for compatibility; final dataset uses auto_accept clips only.")
    parser.add_argument("--copy-files", action="store_true", default=True)
    return parser


def main() -> int:
    return run_builder(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
