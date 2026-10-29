"""Secure JVNV extraction and manifest construction without label leakage into inference."""
from __future__ import annotations
import argparse, csv, hashlib, json, re, zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "jvnv_validation_config.json").read_text(encoding="utf-8"))

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def parse_member(name: str):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or re.match(r"^[A-Za-z]:", name):
        raise ValueError(f"Unsafe ZIP member: {name}")
    if p.suffix.lower() != ".wav" or len(p.parts) < 5:
        return None
    # jvnv_v1 / speaker / emotion / clip_type / file.wav
    speaker, raw_emotion, clip_type = p.parts[-4:-1]
    mapping = CFG["reference_mapping"]
    key = raw_emotion.lower()
    if key not in mapping:
        raise ValueError(f"Unrecognized JVNV emotion folder: {raw_emotion}")
    return speaker, raw_emotion, mapping[key], clip_type

def select_entries(zf: zipfile.ZipFile, limit: int | None):
    parsed = [(e, parse_member(e.filename)) for e in zf.infolist()]
    parsed = [(e, m) for e, m in parsed if m]
    if not limit:
        return parsed
    # Round-robin across reference labels; reference is used only for sampling/evaluation.
    buckets = {}
    for item in parsed:
        buckets.setdefault(item[1][1].lower(), []).append(item)
    chosen = []
    while len(chosen) < limit and any(buckets.values()):
        for label in sorted(buckets):
            if buckets[label] and len(chosen) < limit:
                chosen.append(buckets[label].pop(0))
    return chosen

def extract_and_manifest(zip_path: Path, destination: Path, manifest: Path, limit=None):
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for entry, meta in select_entries(zf, limit):
            target = (destination / PurePosixPath(entry.filename)).resolve()
            if destination.resolve() not in target.parents:
                raise ValueError(f"ZIP traversal rejected: {entry.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite extracted file: {target}")
            with zf.open(entry) as src, target.open("xb") as dst:
                while block := src.read(1024 * 1024): dst.write(block)
            speaker, raw, mapped, clip_type = meta
            rows.append({"clip_id": target.stem, "audio_path": str(target),
                         "speaker_id": f"JVNV_{speaker}", "reference_label_raw": raw,
                         "reference_emotion": mapped, "clip_type_reference": clip_type,
                         "size_bytes": target.stat().st_size, "sha256": sha256(target)})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["clip_id"]); w.writeheader(); w.writerows(rows)
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument("--jvnv-zip", required=True); p.add_argument("--destination", required=True); p.add_argument("--manifest", required=True); p.add_argument("--limit", type=int)
    a=p.parse_args(); rows=extract_and_manifest(Path(a.jvnv_zip),Path(a.destination),Path(a.manifest),a.limit); print(f"Manifested {len(rows)} JVNV recordings")
if __name__ == "__main__": main()
