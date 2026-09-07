import argparse
import csv
import json
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
CSV_ENCODING = "utf-8-sig"


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(value).strip()) or "Unknown"


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


def load_json(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8-sig") as f:
        return {str(k): str(v) for k, v in json.load(f).items()}


def load_cluster_status(report_path: Path) -> Dict[str, str]:
    if not report_path.is_file():
        return {}
    df = pd.read_csv(report_path)
    return {
        str(row.get("speaker_cluster", "")): str(row.get("recommended_status", "needs_manual_mapping"))
        for _, row in df.iterrows()
    }


def rejection_reason(status: str, character: str, cluster: str) -> str:
    if cluster == "SPEAKER_UNKNOWN" or character.strip().lower() == "unknown":
        return "unclear_speaker"
    if status == "bad_cluster":
        return "bad_cluster"
    return ""


def run_character_mapping(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root or DEFAULT_WORKSPACE).resolve()
    speaker_root = output_root / "speaker_clusters"
    manifest_path = speaker_root / "_speaker_cluster_manifest.csv"
    report_path = speaker_root / "_speaker_cluster_report.csv"
    map_path = Path(args.character_map).resolve() if args.character_map else speaker_root / "character_map_template.json"
    char_root = output_root / "character_dialogue"
    rejected_root = output_root / "character_dialogue_rejected"
    out_manifest = char_root / "_character_dialogue_manifest.csv"

    if not manifest_path.is_file():
        print(f"Speaker cluster manifest not found: {manifest_path}")
        return 1
    if not map_path.is_file():
        print(f"Character map not found: {map_path}")
        return 1

    if char_root.exists():
        shutil.rmtree(char_root)
    if rejected_root.exists():
        shutil.rmtree(rejected_root)
    ensure_dir(char_root)
    ensure_dir(rejected_root)

    mapping = load_json(map_path)
    statuses = load_cluster_status(report_path)
    df = pd.read_csv(manifest_path)
    rows: List[dict] = []

    for _, r in df.iterrows():
        cluster = str(r.get("speaker_cluster", "SPEAKER_UNKNOWN"))
        character = mapping.get(cluster, "Unknown").strip() or "Unknown"
        status = statuses.get(cluster, "needs_manual_mapping")
        reason = rejection_reason(status, character, cluster)
        src = Path(str(r.get("output_file") or r.get("file_path") or ""))
        if not src.is_file():
            src = speaker_root / cluster / str(r.get("source_file", ""))
        if not src.is_file():
            continue

        if reason:
            reject_folder = "Unknown" if reason == "unclear_speaker" else reason
            dst = copy_unique(src, rejected_root / reject_folder)
            mapped_status = "rejected"
        else:
            dst = copy_unique(src, char_root / safe_name(character))
            mapped_status = "accepted"

        rows.append(
            {
                "clip_id": r.get("clip_id", src.stem),
                "source_file": r.get("source_file", src.name),
                "output_file": str(dst),
                "character_name": character,
                "speaker_cluster": cluster,
                "mapping_status": mapped_status,
                "reject_reason": reason,
                "cluster_status": status,
                "transcript": r.get("transcript", ""),
                "duration_sec": r.get("duration_sec", ""),
                "source_media": r.get("source_media", ""),
                "start_sec": r.get("start_sec", ""),
                "end_sec": r.get("end_sec", ""),
            }
        )

    fields = [
        "clip_id",
        "source_file",
        "output_file",
        "character_name",
        "speaker_cluster",
        "mapping_status",
        "reject_reason",
        "cluster_status",
        "transcript",
        "duration_sec",
        "source_media",
        "start_sec",
        "end_sec",
    ]
    with out_manifest.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in rows])

    accepted = sum(1 for row in rows if row["mapping_status"] == "accepted")
    rejected = len(rows) - accepted
    print(f"Character dialogue saved: {char_root}")
    print(f"Rejected dialogue saved: {rejected_root}")
    print(f"Character dialogue manifest saved: {out_manifest}")
    print(f"Accepted clips: {accepted}; rejected clips: {rejected}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply speaker-cluster to character-name mapping.")
    parser.add_argument("--output-root", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--character-map")
    parser.add_argument("--copy-files", action="store_true", default=True)
    return parser


def main() -> int:
    return run_character_mapping(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
