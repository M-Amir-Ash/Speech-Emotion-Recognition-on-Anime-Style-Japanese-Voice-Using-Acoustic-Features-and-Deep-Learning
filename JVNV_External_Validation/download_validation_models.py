"""Plan or explicitly download only pinned historical model dependencies."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

HERE=Path(__file__).resolve().parent; CACHE=HERE/"model_cache"/"huggingface"
REVISIONS=json.loads((HERE/"model_revisions.json").read_text(encoding="utf-8"))
REQUIRED=[
 ("Bagus/wav2vec2-xlsr-japanese-speech-emotion-recognition",["*.json","*.safetensors"]),
 ("Dpngtm/wav2vec2-emotion-recognition",["*.json","*.safetensors","*.txt"]),
 ("koshin2001/Japanese-to-emotions",["*.json","*.safetensors","*.txt","*.model"]),
 ("speechbrain/emotion-recognition-wav2vec2-IEMOCAP",None),
 ("Systran/faster-whisper-small",None),
]
OPTIONAL=[("superb/wav2vec2-base-superb-er",["*.json","*.bin","*.txt"])]
def set_isolation():
    os.environ.update({"HF_HOME":str(CACHE),"HF_HUB_CACHE":str(CACHE/"hub"),"HUGGINGFACE_HUB_CACHE":str(CACHE/"hub"),"TRANSFORMERS_CACHE":str(CACHE/"transformers"),"PYTHONDONTWRITEBYTECODE":"1","PYTHONPYCACHEPREFIX":str(HERE/"model_cache/pycache")})
def main():
    p=argparse.ArgumentParser(); p.add_argument("--download",action="store_true",help="Perform the reported pinned downloads"); p.add_argument("--include-optional-fallback",action="store_true"); a=p.parse_args(); set_isolation(); entries=REQUIRED+(OPTIONAL if a.include_optional_fallback else [])
    api=HfApi(); plan=[]
    for model,patterns in entries:
        info=api.model_info(model,revision=REVISIONS[model],files_metadata=True)
        files=[s for s in info.siblings if not patterns or any(__import__('fnmatch').fnmatch(s.rfilename,p) for p in patterns)]
        plan.append({"model":model,"revision":REVISIONS[model],"estimated_bytes":sum(s.size or 0 for s in files),"allow_patterns":patterns})
    total=sum(x["estimated_bytes"] for x in plan); print(json.dumps({"cache":str(CACHE),"models":plan,"total_bytes":total,"total_gib":round(total/2**30,3),"download_requested":a.download},indent=2))
    if not a.download: return
    CACHE.mkdir(parents=True,exist_ok=True)
    for model,patterns in entries:
        snapshot_download(repo_id=model,revision=REVISIONS[model],cache_dir=CACHE/"hub",allow_patterns=patterns)
if __name__=="__main__": main()
