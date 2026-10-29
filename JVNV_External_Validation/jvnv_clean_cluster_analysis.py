"""Reproduce the historical multi_feature centroid selector without reference-label access."""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

def load_original(path: Path):
    spec=importlib.util.spec_from_file_location("historical_cluster_selector",path)
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod

def analyze(predictions: Path, original_script: Path, output: Path, target=400):
    # Deliberately remove reference data before feature construction and fitting.
    source=pd.read_csv(predictions); safe=source.drop(columns=[c for c in source if c.startswith("reference")],errors="ignore")
    hist=load_original(original_script); rows=[]; feats=[]; cache={}
    for raw in safe.to_dict("records"):
        row=dict(raw); row["target_emotion"]=row.get("predicted_emotion",""); row["top_label"]=row["target_emotion"]
        row["top_score"]=row.get("confidence",0); row["top_gap"]=row.get("confidence_gap",0); row["evidence_count"]=row.get("model_agreement_count",0)
        audio=Path(str(row.get("audio_path","")))
        if row["target_emotion"] not in hist.EMOTIONS or not audio.is_file() or row.get("audio_quality_status")!="pass":
            row["separability_status"]="hard_ambiguous"; rows.append(row); continue
        debug={}
        for k in hist.SCORE_BRANCHES:
            value=row.get(k,{})
            if isinstance(value,str):
                try: value=json.loads(value)
                except (ValueError,TypeError): value={}
            debug[k]=value if isinstance(value,dict) else {}
        feature,has=hist.build_feature_row(row,debug,audio,cache,"multi_feature")
        row["has_features"]=has; row["audio_path"]=str(audio); rows.append(row); feats.append((row,feature))
    usable=[r for r,_ in feats]
    if usable:
        arr=np.asarray([x for _,x in feats],dtype=np.float32); med=np.nanmedian(np.where(np.isfinite(arr),arr,np.nan),axis=0); med=np.where(np.isfinite(med),med,0); arr=np.where(np.isfinite(arr),arr,med)
        hist.classify_separability(usable,StandardScaler().fit_transform(arr))
    # Historical balanced target: clean only by default; never duplicate/fabricate.
    selected=set()
    for label in hist.EMOTIONS:
        pool=[r for r in usable if r["target_emotion"]==label and r["separability_status"]=="clean_separable"]
        pool.sort(key=hist.selection_key)
        selected.update(str(r.get("clip_id")) for r in pool[:target])
    for r in rows: r["selected_clean_target"] = str(r.get("clip_id")) in selected
    output.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(output,index=False)
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument("--predictions",required=True); p.add_argument("--original-selector",required=True); p.add_argument("--output",required=True); p.add_argument("--target",type=int,default=400); a=p.parse_args(); print(f"Wrote {len(analyze(Path(a.predictions),Path(a.original_selector),Path(a.output),a.target))} rows")
if __name__=="__main__": main()
