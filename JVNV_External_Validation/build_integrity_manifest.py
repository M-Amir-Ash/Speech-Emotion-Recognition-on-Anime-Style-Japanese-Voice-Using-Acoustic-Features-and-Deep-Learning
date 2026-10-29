"""Resume the protected-tree SHA-256 inventory; reads only, appends new rows."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=(HERE.parent/"Speech Emotion Recognition on Anime-Style Japanese Voice Using Acoustic Features and Deep Learning/Training Data/Auto Split More Sample").resolve()
OUT=HERE/"results/reports/original_integrity_before.csv"
STATUS=HERE/"results/reports/original_integrity_status.json"
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def main():
    done=set()
    if OUT.exists():
        with OUT.open(encoding="utf-8-sig",newline="") as f: done={r["relative_path"] for r in csv.DictReader(f)}
    files=[p for p in ROOT.rglob("*") if p.is_file()]; new=not OUT.exists()
    with OUT.open("a",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        if new: w.writerow(["relative_path","size_bytes","sha256"])
        for i,p in enumerate(files,1):
            rel=str(p.relative_to(ROOT))
            if rel not in done: w.writerow([rel,p.stat().st_size,digest(p)]); f.flush(); done.add(rel)
            if i%1000==0: STATUS.write_text(json.dumps({"complete":False,"hashed":len(done),"discovered":len(files)},indent=2),encoding="utf-8")
    STATUS.write_text(json.dumps({"complete":len(done)==len(files),"hashed":len(done),"discovered":len(files)},indent=2),encoding="utf-8")
if __name__=="__main__": main()
