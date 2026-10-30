#!/usr/bin/env python3
"""Rebuild selected clips from explicitly mapped, user-supplied source media."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PACKAGE_DIR / "reconstruction_manifest.csv"
REQUIRED_MANIFEST_FIELDS = {
    "clip_id", "filename", "split", "emotion", "source_media_id",
    "start_sec", "end_sec", "expected_sha256",
}
ALLOWED_SPLITS = {"train", "validation", "test"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_component(value: str, field: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"unsafe {field}: {value!r}")
    return value


def load_source_map(path: Path) -> dict[str, Path]:
    rows = read_csv(path)
    if not rows or not {"source_media_id", "local_path"}.issubset(rows[0]):
        raise ValueError("source map requires source_media_id and local_path columns")
    result: dict[str, Path] = {}
    for row in rows:
        source_id = row["source_media_id"].strip()
        local_path = row["local_path"].strip()
        if source_id and local_path:
            if source_id in result:
                raise ValueError(f"duplicate source_media_id in source map: {source_id}")
            result[source_id] = Path(local_path).expanduser()
    return result


def validate_manifest(rows: list[dict[str, str]]) -> None:
    if not rows or not REQUIRED_MANIFEST_FIELDS.issubset(rows[0]):
        missing = REQUIRED_MANIFEST_FIELDS - (set(rows[0]) if rows else set())
        raise ValueError(f"manifest missing fields: {sorted(missing)}")
    seen: set[str] = set()
    for row in rows:
        clip_id = row["clip_id"].strip()
        if not clip_id or clip_id in seen:
            raise ValueError(f"missing or duplicate clip_id: {clip_id!r}")
        seen.add(clip_id)
        if row["split"] not in ALLOWED_SPLITS:
            raise ValueError(f"invalid split for {clip_id}: {row['split']!r}")
        start, end = float(row["start_sec"]), float(row["end_sec"])
        if start < 0 or end <= start:
            raise ValueError(f"invalid bounds for {clip_id}: {start}, {end}")
        digest = row["expected_sha256"].lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid expected SHA-256 for {clip_id}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--source-map", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--report", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    rows = read_csv(args.manifest)
    validate_manifest(rows)
    sources = load_source_map(args.source_map)
    results: list[dict[str, str]] = []
    cache: dict[str, object] = {}

    if not args.dry_run:
        try:
            from pydub import AudioSegment
        except ImportError:
            print("pydub is required: python -m pip install pydub", file=sys.stderr)
            return 2

    for row in rows:
        source_id = row["source_media_id"].strip()
        source = sources.get(source_id)
        status = ""
        actual = ""
        output = args.output_dir / safe_component(row["split"], "split") / safe_component(
            row["emotion"], "emotion"
        ) / safe_component(row["filename"], "filename")

        if source is None:
            status = "missing_source_mapping"
        elif not source.is_file():
            status = "source_file_not_found"
        elif args.dry_run:
            status = "ready"
        elif output.exists() and not args.overwrite:
            actual = sha256_file(output)
            status = "verified_existing" if actual == row["expected_sha256"].lower() else "existing_hash_mismatch"
        else:
            if source_id not in cache:
                cache[source_id] = (
                    AudioSegment.from_file(source)
                    .set_channels(1)
                    .set_frame_rate(16000)
                    .set_sample_width(2)
                )
            audio = cache[source_id]
            start_ms = int(float(row["start_sec"]) * 1000)
            end_ms = int(float(row["end_sec"]) * 1000)
            output.parent.mkdir(parents=True, exist_ok=True)
            audio[start_ms:end_ms].export(output, format="wav")
            actual = sha256_file(output)
            status = "verified" if actual == row["expected_sha256"].lower() else "hash_mismatch"

        results.append({
            "clip_id": row["clip_id"],
            "output_path": str(output),
            "status": status,
            "expected_sha256": row["expected_sha256"].lower(),
            "actual_sha256": actual,
        })

    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    for status in sorted(counts):
        print(f"{status}: {counts[status]}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)

    failed = sum(v for k, v in counts.items() if k not in {"ready", "verified", "verified_existing"})
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

