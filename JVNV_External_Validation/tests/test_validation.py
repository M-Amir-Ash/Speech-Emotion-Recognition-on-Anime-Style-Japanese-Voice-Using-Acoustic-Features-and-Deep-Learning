import csv, sys, zipfile
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validate_jvnv_labels import parse_member, extract_and_manifest
from evaluate_jvnv_results_v2 import metrics_for, full_population_metrics

def test_label_mapping_and_neutral_absence():
    assert parse_member("jvnv_v1/F1/surprise/free/F1_surprise_free_01.wav")[2]=="Pleasant_Surprise"
    assert parse_member("jvnv_v1/M1/anger/regular/x.wav")[2]=="Angry"

def test_zip_traversal_rejected():
    try: parse_member("../evil.wav")
    except ValueError: pass
    else: raise AssertionError("traversal accepted")

def test_metrics_missing_classes_and_false_neutral():
    df=pd.DataFrame({"reference_emotion":["Angry","Happy"],"predicted_emotion":["Neutral","Happy"]})
    m=metrics_for(df); assert m["accuracy"]==0.5 and m["false_neutral_count"]==1
    assert set(m["per_class"])=={"Angry","Disgust","Fear","Happy","Pleasant_Surprise","Sad"}

def test_failed_prediction_reduces_coverage():
    df=pd.DataFrame({"reference_emotion":["Angry","Happy"],"predicted_emotion":["Angry",""]})
    assert metrics_for(df)["coverage"]==0.5

def test_abstention_has_no_conditional_score():
    df=pd.DataFrame({"reference_emotion":[],"predicted_emotion":[]})
    m=metrics_for(df,"accepted_only"); assert m["sample_count"]==0 and m["accuracy"] is None

def test_primary_metrics_retain_failures_abstentions_and_neutral():
    df=pd.DataFrame({
        "clip_id":["a","b","c","d"],
        "reference_emotion":["Angry","Happy","Sad","Fear"],
        "predicted_emotion":["Angry","Neutral","Sad",""],
        "confidence_status":["auto_accept","needs_review","reject",""],
        "error_message":["","","","MemoryError"],
    })
    m,ledger=full_population_metrics(df)
    assert len(ledger)==4
    assert m["prediction_coverage"]==0.75
    assert m["raw_predicted_label_accuracy_full_population"]==0.5
    assert m["operational_coverage"]==0.25
    assert m["auto_accepted_predictions"]==1
    assert m["auto_accepted_accuracy"]==1.0
    assert m["primary_operational_accuracy_full_population"]==0.25
    assert m["accuracy_among_successfully_returned_labels"]==2/3
    assert m["abstentions"]==2 and m["failed_inferences"]==1
    assert m["false_neutral_count"]==1
    assert set(ledger.inference_outcome)=={"prediction","abstention","failed_inference"}
    assert ledger.loc[ledger.clip_id.eq("b"), "operational_prediction"].item()=="__ABSTAIN__"

def test_clean_clustering_is_opt_in_in_runner():
    source=(Path(__file__).resolve().parents[1]/"run_jvnv_validation.py").read_text(encoding="utf-8")
    assert 'add_argument("--run-clean-clustering",action="store_true"' in source
    assert "if a.run_clean_clustering:" in source

def test_historical_clean_cluster_statuses_load_without_reference_labels():
    import importlib.util
    root=Path(__file__).resolve().parents[2]
    script=root/"Speech Emotion Recognition on Anime-Style Japanese Voice Using Acoustic Features and Deep Learning/Training Data/Auto Split More Sample/cluster_clean_target_selector.py"
    spec=importlib.util.spec_from_file_location("selector_test",script); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    rows=[]; pts=[]
    for label,offset in (("Angry",0.0),("Happy",10.0)):
        for i in range(8):
            rows.append({"target_emotion":label,"audio_quality_status":"pass","has_features":True}); pts.append([offset+i*.01,offset])
    mod.classify_separability(rows,np.asarray(pts,dtype=np.float32))
    assert {r["separability_status"] for r in rows} <= {"clean_separable","usable_review","overlap_review","hard_ambiguous"}
    assert all("reference_emotion" not in r for r in rows)

def test_extraction_does_not_overwrite(tmp_path):
    z=tmp_path/"x.zip"; member="jvnv_v1/F1/happy/free/F1_happy_free_01.wav"
    with zipfile.ZipFile(z,"w") as f: f.writestr(member,b"RIFF")
    out=tmp_path/"out"; man=tmp_path/"m.csv"; extract_and_manifest(z,out,man)
    try: extract_and_manifest(z,out,tmp_path/"m2.csv")
    except FileExistsError: pass
    else: raise AssertionError("overwrite allowed")

def test_protected_tree_is_outside_validation():
    here=Path(__file__).resolve().parents[1]
    protected=(here.parent/"Speech Emotion Recognition on Anime-Style Japanese Voice Using Acoustic Features and Deep Learning/Training Data/Auto Split More Sample").resolve()
    assert not here.resolve().is_relative_to(protected)
