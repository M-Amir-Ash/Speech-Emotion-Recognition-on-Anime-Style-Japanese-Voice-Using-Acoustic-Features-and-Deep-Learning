#!/usr/bin/env python3
"""Build reconstruction metadata from the three existing canonical manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUTPUT_FIELDS = [
    "clip_id", "filename", "split", "emotion", "source_media_id",
    "source_group", "start_sec", "end_sec", "duration_sec",
    "expected_sha256", "provenance_status", "missing_provenance",
]
MISSING_PROVENANCE = (
    "source_stem_sha256|source_separation_model|source_separation_settings|"
    "exact_ffmpeg_pydub_versions"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest_dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, str]] = []
    for split in ("train", "validation", "test"):
        path = args.manifest_dir / f"{split}_manifest.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for source in csv.DictReader(handle):
                rows.append({
                    "clip_id": source["clip_id"],
                    "filename": source["filename"],
                    "split": split,
                    "emotion": source["emotion"],
                    "source_media_id": source["source_media"],
                    "source_group": source["source_group"],
                    "start_sec": source["start_sec"],
                    "end_sec": source["end_sec"],
                    "duration_sec": source["duration_sec"],
                    "expected_sha256": source["sha256"].lower(),
                    "provenance_status": "conditional_exact_historical_stem_required",
                    "missing_provenance": MISSING_PROVENANCE,
                })

    if len(rows) != 1969 or len({r["clip_id"] for r in rows}) != 1969:
        raise ValueError("canonical manifests must contain 1,969 unique clip IDs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "reconstruction_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    sources: dict[str, str] = {}
    for row in rows:
        source_id = row["source_media_id"]
        group = row["source_group"]
        if source_id in sources and sources[source_id] != group:
            raise ValueError(f"conflicting source group for {source_id}")
        sources[source_id] = group
    mapping = args.output_dir / "source_media_map_template.csv"
    with mapping.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_media_id", "source_group", "local_path"])
        writer.writeheader()
        for source_id in sorted(sources, key=str.casefold):
            writer.writerow({"source_media_id": source_id, "source_group": sources[source_id], "local_path": ""})
    print(f"wrote {len(rows)} clips and {len(sources)} source identifiers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
