"""Primary leakage-safe evaluation of every JVNV recording, without sample selection."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

OUTPUT_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant_Surprise", "Sad"]
REFERENCE_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Pleasant_Surprise", "Sad"]
ABSTENTION_LABEL = "__ABSTAIN__"
FAILURE_LABEL = "__FAILED__"

def _status(df):
    return df.get("confidence_status", pd.Series("", index=df.index)).fillna("").astype(str)

def _outcomes(df):
    predicted = df.predicted_emotion.fillna("").astype(str)
    outcome = pd.Series("prediction", index=df.index, dtype="object")
    outcome.loc[~predicted.isin(OUTPUT_LABELS)] = "failed_inference"
    outcome.loc[predicted.isin(OUTPUT_LABELS) & ~_status(df).eq("auto_accept")] = "abstention"
    return outcome

def metrics_for(df, name="all"):
    """Compatibility helper: conditional metrics among returned seven-class labels."""
    total = len(df); ok = df[df.predicted_emotion.isin(OUTPUT_LABELS)].copy()
    y, pred = ok.reference_emotion, ok.predicted_emotion
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=REFERENCE_LABELS, zero_division=0)
    return {"subset": name, "sample_count": total, "successful_predictions": len(ok),
        "coverage": len(ok) / total if total else 0.0,
        "accuracy": float(accuracy_score(y, pred)) if len(ok) else None,
        "macro_precision": float(np.mean(pr)) if len(ok) else None,
        "macro_recall": float(np.mean(rc)) if len(ok) else None,
        "macro_f1": float(np.mean(f1)) if len(ok) else None,
        "false_neutral_count": int(((pred == "Neutral") & (y != "Neutral")).sum()),
        "per_class": {lab: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])} for i, lab in enumerate(REFERENCE_LABELS)}}

def full_population_metrics(df):
    """Primary metrics: every JVNV reference row contributes to the denominator."""
    ledger = df.copy(); ledger["inference_outcome"] = _outcomes(ledger)
    predicted = ledger.predicted_emotion.fillna("").astype(str)
    evaluated = predicted.copy()
    evaluated.loc[ledger.inference_outcome.eq("abstention")] = ABSTENTION_LABEL
    evaluated.loc[ledger.inference_outcome.eq("failed_inference")] = FAILURE_LABEL
    ledger["operational_prediction"] = evaluated
    ledger["operationally_correct"] = evaluated.eq(ledger.reference_emotion)
    pr, rc, f1, sup = precision_recall_fscore_support(ledger.reference_emotion, evaluated, labels=REFERENCE_LABELS, zero_division=0)
    total = len(ledger); returned = predicted.isin(OUTPUT_LABELS); operational = ledger.inference_outcome.eq("prediction")
    auto_accepted = returned & _status(ledger).eq("auto_accept")
    auto_accepted_correct = auto_accepted & predicted.eq(ledger.reference_emotion)
    conditional = metrics_for(ledger[returned], "successfully_returned_labels_only")
    metrics = {
        "validation_question": "How well does the reconstructed automatic Japanese emotion-labeling pipeline agree with independently labeled Japanese emotional speech, without acoustic sample selection?",
        "benchmark_scope": "external Japanese acted-speech benchmark; not direct anime-domain pseudo-label verification",
        "primary_dataset": "all JVNV reference recordings; no clean-clustering or acoustic-overlap filtering",
        "total_reference_recordings": total,
        "successful_predictions": int(returned.sum()),
        "prediction_coverage": float(returned.mean()) if total else 0.0,
        "raw_predicted_label_accuracy_full_population": (
            float((returned & predicted.eq(ledger.reference_emotion)).sum() / total)
            if total
            else None
        ),
        "operational_predictions": int(operational.sum()),
        "operational_coverage": float(operational.mean()) if total else 0.0,
        "auto_accepted_predictions": int(auto_accepted.sum()),
        "auto_accepted_accuracy": (
            float(auto_accepted_correct.sum() / auto_accepted.sum())
            if auto_accepted.any()
            else None
        ),
        "abstentions": int(ledger.inference_outcome.eq("abstention").sum()),
        "failed_inferences": int(ledger.inference_outcome.eq("failed_inference").sum()),
        "primary_operational_accuracy_full_population": (
            float(auto_accepted_correct.sum() / total) if total else None
        ),
        "primary_operational_macro_precision_full_population": float(np.mean(pr)) if total else None,
        "primary_operational_macro_recall_full_population": float(np.mean(rc)) if total else None,
        "primary_operational_macro_f1_full_population": float(np.mean(f1)) if total else None,
        "accuracy_among_successfully_returned_labels": conditional["accuracy"],
        "macro_f1_among_successfully_returned_labels": conditional["macro_f1"],
        "false_neutral_count": int(((predicted == "Neutral") & (ledger.reference_emotion != "Neutral")).sum()),
        "reference_labels": REFERENCE_LABELS, "model_output_labels": OUTPUT_LABELS,
        "reference_mapping_note": "JVNV Surprise is prospectively mapped to Pleasant_Surprise; JVNV has no Neutral reference class.",
        "acoustic_overlap_interpretation": "Acoustic overlap is not label error and never excludes a row from primary metrics.",
        "per_class_full_population": {lab: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])} for i, lab in enumerate(REFERENCE_LABELS)},
        "conditional_successful_predictions": conditional}
    return metrics, ledger

def evaluate(predictions, out_dir, figures):
    import matplotlib.pyplot as plt
    df = pd.read_csv(predictions); out_dir.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    raw, ledger = full_population_metrics(df); status = _status(df)
    raw.update({"auto_accepted": int(status.eq("auto_accept").sum()), "needs_review": int(status.str.contains("review").sum()),
        "rejected": int(status.eq("reject").sum()), "acceptance_rate": float(status.eq("auto_accept").mean()) if len(df) else 0.0})
    (out_dir / "jvnv_primary_unfiltered_metrics.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in raw.items() if not isinstance(v, (dict, list))}]).to_csv(out_dir / "jvnv_primary_unfiltered_metrics.csv", index=False)
    pd.DataFrame([{"class": k, **v} for k, v in raw["per_class_full_population"].items()]).to_csv(out_dir / "jvnv_primary_classwise_metrics.csv", index=False)
    ledger.to_csv(out_dir / "jvnv_primary_all_samples_ledger.csv", index=False)
    speaker_rows = []
    if "speaker_id" in ledger:
        for speaker, group in ledger.groupby("speaker_id"):
            sm, _ = full_population_metrics(group)
            speaker_rows.append({"speaker_id": speaker, **{k: v for k, v in sm.items() if not isinstance(v, (dict, list))}})
    pd.DataFrame(speaker_rows).to_csv(out_dir / "jvnv_primary_speaker_metrics.csv", index=False)
    columns = OUTPUT_LABELS + [ABSTENTION_LABEL, FAILURE_LABEL]
    cm = np.zeros((len(REFERENCE_LABELS), len(columns)), dtype=int)
    for i, ref in enumerate(REFERENCE_LABELS):
        for j, pred in enumerate(columns): cm[i, j] = int(((ledger.reference_emotion == ref) & (ledger.operational_prediction == pred)).sum())
    cm_df = pd.DataFrame(cm, index=REFERENCE_LABELS, columns=columns); cm_df.index.name = "reference_emotion"
    cm_df.to_csv(out_dir / "jvnv_primary_confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(12, 7)); im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right"); ax.set_yticks(range(len(REFERENCE_LABELS)), REFERENCE_LABELS)
    ax.set_xlabel("Operational prediction"); ax.set_ylabel("JVNV reference")
    for i in range(len(REFERENCE_LABELS)):
        for j in range(len(columns)): ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(figures / "jvnv_primary_unfiltered_confusion_matrix.png", dpi=180); plt.close(fig)
    return raw

def main():
    p = argparse.ArgumentParser(); p.add_argument("--predictions", required=True); p.add_argument("--output-dir", default="results/reports"); p.add_argument("--figures-dir", default="results/figures"); a = p.parse_args()
    print(json.dumps(evaluate(Path(a.predictions), Path(a.output_dir), Path(a.figures_dir)), indent=2))

if __name__ == "__main__": main()
