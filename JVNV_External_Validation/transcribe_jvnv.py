"""Isolated faster-whisper process to avoid PyTorch/NumPy OpenMP conflicts on Windows."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from faster_whisper import WhisperModel
import ctranslate2

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--model",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    # CPU is the safe fallback on this host: CTranslate2 detects the GPU but its
    # isolated runtime cannot resolve cublas64_12.dll.
    device="cpu"; compute="int8"
    model=WhisperModel(a.model,device=device,compute_type=compute); texts={}
    with open(a.manifest,encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            segments,_=model.transcribe(r["audio_path"],language="ja",beam_size=3)
            texts[r["clip_id"]]="".join(s.text for s in segments).strip()
    Path(a.output).write_text(json.dumps({"device":device,"compute_type":compute,"transcripts":texts},ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__": main()
