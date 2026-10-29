"""Leakage-safe evaluation: reference columns are accessed only in this module."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

OUTPUT_LABELS=["Angry","Disgust","Fear","Happy","Neutral","Pleasant_Surprise","Sad"]
REFERENCE_LABELS=["Angry","Disgust","Fear","Happy","Pleasant_Surprise","Sad"]

def metrics_for(df, name="all"):
    total=len(df); ok=df[df.predicted_emotion.isin(OUTPUT_LABELS)].copy()
    y=ok.reference_emotion; pred=ok.predicted_emotion
    pr,rc,f1,sup=precision_recall_fscore_support(y,pred,labels=REFERENCE_LABELS,zero_division=0)
    macro=lambda x: float(np.mean(x)) if len(x) else None
    return {"subset":name,"sample_count":total,"successful_predictions":len(ok),
      "coverage":len(ok)/total if total else 0.0,"accuracy":float(accuracy_score(y,pred)) if len(ok) else None,
      "macro_precision":macro(pr),"macro_recall":macro(rc),"macro_f1":macro(f1),
      "false_neutral_count":int(((pred=="Neutral") & (y!="Neutral")).sum()),
      "per_class":{lab:{"precision":float(pr[i]),"recall":float(rc[i]),"f1":float(f1[i]),"support":int(sup[i])} for i,lab in enumerate(REFERENCE_LABELS)}}

def evaluate(predictions: Path, out_dir: Path, figures: Path):
    import matplotlib.pyplot as plt
    df=pd.read_csv(predictions); out_dir.mkdir(parents=True,exist_ok=True); figures.mkdir(parents=True,exist_ok=True)
    raw=metrics_for(df,"all_successful")
    status=df.confidence_status.fillna("")
    raw.update({"total_reference_recordings":len(df),"failed":int((~df.predicted_emotion.isin(OUTPUT_LABELS)).sum()),
      "auto_accepted":int((status=="auto_accept").sum()),"needs_review":int(status.str.contains("review").sum()),"rejected":int((status=="reject").sum()),
      "acceptance_rate":float((status=="auto_accept").mean()) if len(df) else 0.0,
      "operational_accuracy_rejected_as_incorrect":float(((df.predicted_emotion==df.reference_emotion)&(status!="reject")).sum()/len(df)) if len(df) else None,
      "conditional_accuracy_among_accepted":metrics_for(df[status=="auto_accept"],"accepted_only")["accuracy"]})
    (out_dir/"jvnv_raw_metrics.json").write_text(json.dumps(raw,indent=2,ensure_ascii=False),encoding="utf-8")
    pd.DataFrame([{k:v for k,v in raw.items() if k!="per_class"}]).to_csv(out_dir/"jvnv_raw_metrics.csv",index=False)
    pd.DataFrame([{"class":k,**v} for k,v in raw["per_class"].items()]).to_csv(out_dir/"jvnv_classwise_metrics.csv",index=False)
    speaker_rows=[]
    if "speaker_id" in df:
        for speaker,group in df.groupby("speaker_id"):
            sm=metrics_for(group,str(speaker)); speaker_rows.append({k:v for k,v in sm.items() if k!="per_class"})
    pd.DataFrame(speaker_rows).to_csv(out_dir/"jvnv_speaker_metrics.csv",index=False)
    success=df[df.predicted_emotion.isin(OUTPUT_LABELS)]
    cm=confusion_matrix(success.reference_emotion,success.predicted_emotion,labels=OUTPUT_LABELS)
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(cm,cmap="Blues"); ax.set_xticks(range(7),OUTPUT_LABELS,rotation=45,ha="right"); ax.set_yticks(range(7),OUTPUT_LABELS); ax.set_xlabel("Predicted"); ax.set_ylabel("Reference (Neutral row is structurally empty)")
    for i in range(7):
      for j in range(7): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(figures/"jvnv_confusion_matrix.png",dpi=180); plt.close(fig)
    cls=pd.read_csv(out_dir/"jvnv_classwise_metrics.csv"); fig,ax=plt.subplots(figsize=(9,5)); ax.bar(cls["class"],cls.f1); ax.set_ylim(0,1); ax.set_ylabel("F1"); ax.tick_params(axis="x",rotation=30); fig.tight_layout(); fig.savefig(figures/"jvnv_classwise_f1.png",dpi=180); plt.close(fig)
    return raw

def main():
    p=argparse.ArgumentParser(); p.add_argument("--predictions",required=True); p.add_argument("--output-dir",default="results/reports"); p.add_argument("--figures-dir",default="results/figures"); a=p.parse_args(); print(json.dumps(evaluate(Path(a.predictions),Path(a.output_dir),Path(a.figures_dir)),indent=2))
if __name__=="__main__":
    # Compatibility entry point: all new evaluations use the unfiltered v2 protocol.
    from evaluate_jvnv_results_v2 import main as primary_unfiltered_main
    primary_unfiltered_main()
