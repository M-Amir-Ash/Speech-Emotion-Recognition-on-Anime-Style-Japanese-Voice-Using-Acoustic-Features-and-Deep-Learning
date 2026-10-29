from __future__ import annotations
import csv, hashlib, json
from datetime import datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE/"results/reports/original_integrity_before.csv"
PROTECTED=(HERE.parent/"Speech Emotion Recognition on Anime-Style Japanese Voice Using Acoustic Features and Deep Learning/Training Data/Auto Split More Sample").resolve()
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def verify():
    status=json.loads((HERE/"results/reports/original_integrity_status.json").read_text(encoding="utf-8"))
    if not status.get("complete"):
        raise RuntimeError("Baseline integrity manifest is incomplete; run build_integrity_manifest.py to completion first")
    expected={r["relative_path"]:(int(r["size_bytes"]),r["sha256"]) for r in csv.DictReader(BASE.open(encoding="utf-8-sig"))}
    current={str(p.relative_to(PROTECTED)):(p.stat().st_size,digest(p)) for p in PROTECTED.rglob("*") if p.is_file()}
    changed=sorted(k for k in expected.keys()&current.keys() if expected[k]!=current[k]); missing=sorted(expected.keys()-current.keys()); added=sorted(current.keys()-expected.keys())
    result={"passed":not(changed or missing or added),"expected_count":len(expected),"current_count":len(current),"changed":changed,"missing":missing,"added":added}
    out=HERE/"results/reports"/f"original_integrity_after_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"; out.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2)); return result
if __name__=="__main__": raise SystemExit(0 if verify()["passed"] else 1)
