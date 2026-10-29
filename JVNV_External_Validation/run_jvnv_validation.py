"""Orchestrate isolated JVNV validation using the frozen historical labeler."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from validate_jvnv_labels import extract_and_manifest

HERE=Path(__file__).resolve().parent; PROJECT=HERE.parent
CFG=json.loads((HERE/"jvnv_validation_config.json").read_text(encoding="utf-8"))
ORIGINAL=(PROJECT/CFG["historical_pipeline_relative"]).resolve()
FROZEN=(HERE/"frozen_source").resolve()

def hardlink_or_copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    try: os.link(src,dst)
    except OSError: shutil.copy2(src,dst)

def prepare_inference_input(reference_manifest, workspace, transcripts=None):
    rows=pd.read_csv(reference_manifest); out=[]
    for r in rows.to_dict("records"):
        # Neutral technical speaker IDs and filenames contain no reference emotion.
        dst=workspace/"character_dialogue"/r["speaker_id"]/(r["clip_id"]+".wav")
        hardlink_or_copy(Path(r["audio_path"]),dst)
        out.append({"clip_id":r["clip_id"],"output_file":str(dst),"source_file":r["clip_id"]+".wav","character_name":r["speaker_id"],"speaker_cluster":r["speaker_id"],"transcript":(transcripts or {}).get(r["clip_id"],""),"duration_sec":""})
    pd.DataFrame(out).to_csv(workspace/"character_dialogue"/"_character_dialogue_manifest.csv",index=False)

def run_historical(workspace, limit):
    env=os.environ.copy(); cache=workspace/"caches"; env.update({"HF_HOME":str(cache/"huggingface"),"TORCH_HOME":str(cache/"torch"),"XDG_CACHE_HOME":str(cache),"PYTHONPYCACHEPREFIX":str(cache/"pycache")})
    shared=HERE/"model_cache"
    sb_rev=json.loads((HERE/"model_revisions.json").read_text(encoding="utf-8"))["speechbrain/emotion-recognition-wav2vec2-IEMOCAP"]
    sb_source=shared/"huggingface/hub/models--speechbrain--emotion-recognition-wav2vec2-IEMOCAP/snapshots"/sb_rev
    env.update({"HF_HOME":str(shared/"huggingface"),"HF_HUB_CACHE":str(shared/"huggingface/hub"),"HUGGINGFACE_HUB_CACHE":str(shared/"huggingface/hub"),"TRANSFORMERS_CACHE":str(shared/"huggingface/transformers"),"TORCH_HOME":str(shared/"torch"),"XDG_CACHE_HOME":str(shared),"PYTHONPYCACHEPREFIX":str(workspace/"caches/pycache"),"PYTHONDONTWRITEBYTECODE":"1","TMP":str(workspace/"tmp"),"TEMP":str(workspace/"tmp"),"HF_HUB_OFFLINE":"1","JVNV_SPEECHBRAIN_SOURCE":str(sb_source)}); (workspace/"tmp").mkdir(exist_ok=True)
    cmd=[sys.executable,str(FROZEN/"character_emotion_pseudolabel.py"),"--input-root",str(workspace/"character_dialogue"),"--output-root",str(workspace),"--classification-output-name","character_emotion_classified"]
    if limit: cmd += ["--classify-limit",str(limit)]
    log=workspace/"historical_labeler.log"
    with log.open("x",encoding="utf-8") as f: result=subprocess.run(cmd,cwd=str(FROZEN),env=env,stdout=f,stderr=subprocess.STDOUT,text=True)
    manifest=workspace/"character_emotion_classified"/"_character_emotion_manifest.csv"
    if result.returncode or not manifest.exists(): raise RuntimeError(f"Historical labeler failed ({result.returncode}); see {log}")
    return manifest

def transcribe(reference_manifest, workspace):
    revision=json.loads((HERE/"model_revisions.json").read_text(encoding="utf-8"))["Systran/faster-whisper-small"]
    snapshot=HERE/"model_cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots"/revision
    if not snapshot.is_dir(): raise RuntimeError(f"Pinned ASR snapshot missing: {snapshot}. Run download_validation_models.py --download first.")
    out=workspace/"asr_transcripts.json"; env=os.environ.copy(); env.update({"PYTHONDONTWRITEBYTECODE":"1","PYTHONPYCACHEPREFIX":str(workspace/"caches/pycache"),"TMP":str(workspace/"tmp"),"TEMP":str(workspace/"tmp")}); (workspace/"tmp").mkdir(exist_ok=True)
    asr_python=HERE/"asr_env/Scripts/python.exe"
    if not asr_python.is_file(): raise RuntimeError("Isolated ASR environment missing; create asr_env and install faster-whisper==1.2.1")
    subprocess.run([str(asr_python),str(HERE/"transcribe_jvnv.py"),"--manifest",str(reference_manifest),"--model",str(snapshot),"--output",str(out)],env=env,check=True)
    return json.loads(out.read_text(encoding="utf-8"))["transcripts"]

def normalize_predictions(hist_manifest, ref_manifest, target):
    h=pd.read_csv(hist_manifest); ref=pd.read_csv(ref_manifest)
    debug_path=hist_manifest.parent/"_debug_scores.jsonl"
    if debug_path.is_file():
        debug=pd.DataFrame([json.loads(line) for line in debug_path.read_text(encoding="utf-8").splitlines() if line.strip()])
        for c in [x for x in debug.columns if x.endswith("_scores")]: debug[c]=debug[c].map(lambda v:json.dumps(v,ensure_ascii=False,sort_keys=True))
        h=h.merge(debug[[c for c in debug.columns if c=="clip_id" or c.endswith("_scores")]],on="clip_id",how="left")
    h["clip_id"]=h.get("clip_id",h.get("source_file","")).astype(str).map(lambda x:Path(x).stem)
    merged=ref.merge(h,on="clip_id",how="left",suffixes=("","_historical"))
    def col(name,*fallback):
        for x in (name,)+fallback:
            if x in merged: return merged[x]
        return ""
    out=pd.DataFrame({"clip_id":merged.clip_id,"audio_path":merged.audio_path,"speaker_id":merged.speaker_id,"reference_emotion":merged.reference_emotion,
      "predicted_emotion":col("top_label","pred_folder"),"second_prediction":col("second_label"),"confidence":col("top_score"),"confidence_gap":col("top_gap"),
      "confidence_status":col("confidence_status","review_status"),"audio_quality_status":col("audio_quality_status"),"clip_type":col("clip_type"),"transcript":col("transcript"),
      "japanese_audio_prediction":col("japanese_audio_ser_top"),"general_audio_prediction":col("general_7class_top"),"audio_baseline_prediction":col("audio_baseline_top"),
      "text_prediction":col("text_model_top","text_top"),"prosody_prediction":col("prosody_top"),"model_agreement_count":col("model_agreement_count","evidence_count"),"error_message":col("error_message")})
    for c in merged.columns:
        if c.endswith("_scores") or c.startswith("final_score_"): out[c]=merged[c]
    target.parent.mkdir(parents=True,exist_ok=True); out.to_csv(target,index=False); return out

def metadata(path, zip_path):
    try:
        import torch; torch_info={"torch":torch.__version__,"cuda_available":torch.cuda.is_available()}
    except Exception as e: torch_info={"torch_error":str(e)}
    data={"run_utc":datetime.now(timezone.utc).isoformat(),"protocol_version":"primary_unfiltered_v2","primary_population":"all JVNV recordings without acoustic selection","clean_clustering_default":False,"python":sys.version,"os":platform.platform(),**torch_info,"jvnv_zip":str(zip_path),"config":CFG}
    path.write_text(json.dumps(data,indent=2),encoding="utf-8")

def finalize_selection_reports(pred_path, cluster_path, reports, figures):
    from evaluate_jvnv_results_v2 import metrics_for
    import matplotlib.pyplot as plt
    pred=pd.read_csv(pred_path); cluster=pd.read_csv(cluster_path)
    status=cluster[["clip_id","separability_status","selected_clean_target"]]
    df=pred.merge(status,on="clip_id",how="left")
    masks={"all":pd.Series(True,index=df.index),"auto_accepted":df.confidence_status.eq("auto_accept"),
           "clean_separable":df.separability_status.eq("clean_separable"),"usable_review":df.separability_status.eq("usable_review"),
           "overlap_review":df.separability_status.eq("overlap_review"),"hard_ambiguous":df.separability_status.eq("hard_ambiguous")}
    rows=[]
    for name,mask in masks.items():
        m=metrics_for(df[mask],name); rows.append({k:v for k,v in m.items() if k!="per_class"})
    comp=pd.DataFrame(rows); comp.to_csv(reports/"jvnv_selection_comparison.csv",index=False)
    fig,ax=plt.subplots(figsize=(9,5)); ax.bar(comp.subset,comp.accuracy.fillna(0)); ax.set_ylim(0,1); ax.set_ylabel("Accuracy"); ax.tick_params(axis="x",rotation=30); fig.tight_layout(); fig.savefig(figures/"jvnv_raw_vs_clean_selection.png",dpi=180); plt.close(fig)
    (reports/"jvnv_optional_clean_clustering_notes.md").write_text(
        "# Optional clean-clustering secondary experiment\n\n"
        "This output describes acoustic separability and training-sample selection only. "
        "It is disabled by default, does not alter the primary JVNV population or metrics, "
        "and must not be interpreted as evidence that acoustically overlapping samples have "
        "incorrect reference or predicted emotion labels. Conditional agreement values in "
        "`jvnv_selection_comparison.csv` are secondary descriptive statistics, not the "
        "external-validation result.\n",
        encoding="utf-8",
    )
    return
    raw=json.loads((reports/"jvnv_raw_metrics.json").read_text(encoding="utf-8"))
    pct=lambda x:"N/A" if x is None else f"{100*x:.2f}%"
    summary=f"""# JVNV validation summary\n\n## What was tested\n\nThe frozen/reconstructed seven-output pseudo-labeler was evaluated on {raw['total_reference_recordings']} independently labeled JVNV recordings. JVNV labels were read from the archive's speaker/emotion/type hierarchy and never passed to inference or clustering.\n\n## Measured results\n\n- Successful predictions: {raw['successful_predictions']} ({pct(raw['coverage'])} coverage)\n- Accuracy on successful predictions: {pct(raw['accuracy'])}\n- Six-reference-class macro-F1: {pct(raw['macro_f1'])}\n- Auto-accepted: {raw['auto_accepted']} ({pct(raw['acceptance_rate'])}); conditional accepted accuracy: {pct(raw['conditional_accuracy_among_accepted'])}\n- Needs review: {raw['needs_review']}; rejected: {raw['rejected']}; failed: {raw['failed']}\n- False Neutral predictions: {raw['false_neutral_count']}\n\nSpeaker and classwise details are in the CSV/JSON reports. Neutral has no JVNV reference row but remains an error-producing prediction column. Surprise→Pleasant_Surprise is an approximate, not identity, mapping.\n\n## Interpretation\n\nRaw results measure label-generator agreement with a Japanese acted-speech benchmark. Selection-category results are retrospective and conditioned on pseudo-label geometry; they are not held-out proof that clustering improves recognition. JVNV is not anime dialogue, so this does not directly validate emotion labels for anime characters. Missing branches and model failures are documented in the run log.\n\n## Publication-ready Results\n\nOn {raw['total_reference_recordings']} JVNV utterances, the reconstructed seven-class pseudo-labeling pipeline produced {raw['successful_predictions']} successful predictions (coverage {pct(raw['coverage'])}), with accuracy {pct(raw['accuracy'])} and six-reference-class macro-F1 {pct(raw['macro_f1'])}. It predicted Neutral incorrectly {raw['false_neutral_count']} times. The auto-accepted subset covered {pct(raw['acceptance_rate'])} of recordings and had conditional accuracy {pct(raw['conditional_accuracy_among_accepted'])}.\n\n## Limitations\n\nThe evaluation reconstructs, rather than proves bit-identical reproduction of, the historical pipeline; upstream pretrained-model exposure to JVNV cannot be fully excluded. JVNV Surprise is only approximately mapped to Pleasant_Surprise, JVNV has no Neutral ground truth, and benchmark speech differs from anime-style dialogue. Clean-subset statistics are descriptive because the same evaluation audio determines pseudo-label-specific centroids.\n"""
    (reports/"jvnv_validation_summary.md").write_text(summary,encoding="utf-8")

def write_primary_summary(reports):
    raw=json.loads((reports/"jvnv_primary_unfiltered_metrics.json").read_text(encoding="utf-8"))
    pct=lambda x:"N/A" if x is None else f"{100*x:.2f}%"
    summary=f"""# Primary unfiltered JVNV validation\n\n## Research question\n\nHow well does the reconstructed automatic Japanese emotion-labeling pipeline agree with independently labeled Japanese emotional speech, without acoustic sample selection?\n\n## Primary population\n\nAll {raw['total_reference_recordings']} JVNV reference recordings are retained. No acoustic-overlap or clean-clustering status affects inclusion. Acoustic overlap is a separability property, not evidence that either the reference or prediction is incorrect.\n\n## Primary operational results\n\n- Full-population accuracy: {pct(raw['primary_operational_accuracy_full_population'])}\n- Full-population macro-F1: {pct(raw['primary_operational_macro_f1_full_population'])}\n- Prediction coverage: {pct(raw['prediction_coverage'])}\n- Operational coverage after abstention: {pct(raw['operational_coverage'])}\n- Successful seven-class outputs: {raw['successful_predictions']}\n- Abstentions: {raw['abstentions']}\n- Failed inferences: {raw['failed_inferences']}\n- False Neutral predictions: {raw['false_neutral_count']}\n\n## Conditional diagnostic results\n\n- Accuracy among successfully returned labels: {pct(raw['accuracy_among_successfully_returned_labels'])}\n- Macro-F1 among successfully returned labels: {pct(raw['macro_f1_among_successfully_returned_labels'])}\n\nThese conditional values do not replace the primary full-population metrics. Neutral remains a valid model output and is an error because JVNV has no Neutral reference class. JVNV Surprise is prospectively mapped to Pleasant_Surprise.\n\n## Scope\n\nThis is external Japanese acted-speech benchmark validation. It is not direct verification of anime-domain pseudo-label accuracy. Models, fusion weights, thresholds, and labeling rules were frozen; JVNV labels were used only for evaluation.\n"""
    (reports/"jvnv_primary_unfiltered_summary.md").write_text(summary,encoding="utf-8")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--jvnv-zip",required=True); mode=p.add_mutually_exclusive_group(required=True); mode.add_argument("--smoke-test",action="store_true"); mode.add_argument("--full",action="store_true"); p.add_argument("--run-id"); p.add_argument("--reuse-transcripts-json",help="Reuse transcripts from a prior identical clip manifest; reference labels are not read"); p.add_argument("--run-clean-clustering",action="store_true",help="Optional secondary acoustic-separability experiment. Disabled by default and never changes primary metrics."); a=p.parse_args()
    if HERE.resolve().is_relative_to(ORIGINAL): raise RuntimeError("Validation workspace is inside protected tree")
    if a.full:
        readiness=json.loads((HERE/"results/reports/full_run_readiness.json").read_text(encoding="utf-8"))
        if not readiness.get("ready"): raise RuntimeError("Full run is scientifically blocked: "+"; ".join(readiness.get("blockers",[])))
        smoke_certificate=HERE/"results/reports/corrected_smoke_passed.json"
        if not smoke_certificate.is_file() or not json.loads(smoke_certificate.read_text(encoding="utf-8")).get("passed"):
            raise RuntimeError("Full run is blocked until a corrected smoke test completes and writes corrected_smoke_passed.json")
    requested_run_id=a.run_id or datetime.now().strftime("%Y%m%d_%H%M%S"); run_id=requested_run_id; run=HERE/"results"/"runs"/run_id
    version=2
    while run.exists():
        run_id=f"{requested_run_id}_v{version}"; run=HERE/"results"/"runs"/run_id; version+=1
    if run_id != requested_run_id: print(f"Run ID already exists; preserving it and using: {run_id}",flush=True)
    for d in ("data/jvnv_extracted","workspace","raw_predictions","clean_cluster","figures","reports","logs"): (run/d).mkdir(parents=True,exist_ok=False)
    ref=run/"data/jvnv_manifest.csv"; limit=CFG["smoke_clips"] if a.smoke_test else None
    extract_and_manifest(Path(a.jvnv_zip).resolve(),run/"data/jvnv_extracted",ref,limit)
    transcripts=json.loads(Path(a.reuse_transcripts_json).read_text(encoding="utf-8"))["transcripts"] if a.reuse_transcripts_json else transcribe(ref,run/"workspace")
    prepare_inference_input(ref,run/"workspace",transcripts)
    metadata(run/"reports/reproducibility_metadata.json",Path(a.jvnv_zip).resolve())
    hist=run_historical(run/"workspace",limit); pred=run/"raw_predictions/jvnv_predictions.csv"; normalize_predictions(hist,ref,pred)
    subprocess.run([sys.executable,str(HERE/"evaluate_jvnv_results_v2.py"),"--predictions",str(pred),"--output-dir",str(run/"reports"),"--figures-dir",str(run/"figures")],check=True)
    write_primary_summary(run/"reports")
    if a.run_clean_clustering:
        subprocess.run([sys.executable,str(HERE/"jvnv_clean_cluster_analysis.py"),"--predictions",str(pred),"--original-selector",str(FROZEN/"cluster_clean_target_selector.py"),"--output",str(run/"clean_cluster/jvnv_cluster_manifest.csv")],check=True)
        finalize_selection_reports(pred,run/"clean_cluster/jvnv_cluster_manifest.csv",run/"reports",run/"figures")
    if a.smoke_test:
        smoke_metrics=json.loads((run/"reports/jvnv_primary_unfiltered_metrics.json").read_text(encoding="utf-8"))
        passed=smoke_metrics["total_reference_recordings"]==CFG["smoke_clips"] and smoke_metrics["failed_inferences"]==0 and smoke_metrics["successful_predictions"]==CFG["smoke_clips"]
        certificate={"passed":passed,"run_id":run_id,"completed_utc":datetime.now(timezone.utc).isoformat(),"recordings":smoke_metrics["total_reference_recordings"],"failed_inferences":smoke_metrics["failed_inferences"],"successful_predictions":smoke_metrics["successful_predictions"],"protocol_version":"primary_unfiltered_v2"}
        (HERE/"results/reports/corrected_smoke_passed.json").write_text(json.dumps(certificate,indent=2),encoding="utf-8")
        if not passed: raise RuntimeError(f"Corrected smoke did not pass; see {run/'reports/jvnv_primary_unfiltered_metrics.json'}")
    print(f"Completed isolated run: {run}")
if __name__=="__main__": main()
