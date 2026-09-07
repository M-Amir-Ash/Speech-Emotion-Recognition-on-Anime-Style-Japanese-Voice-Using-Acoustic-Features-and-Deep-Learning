import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "output" / "final_character_emotion_dataset"
MODELS_DIR = BASE_DIR / "models"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant_Surprise", "Sad"]
TARGET_SR = 16000


def extract_embedding(wav_path: Path, backend: str, device: str) -> Optional[np.ndarray]:
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
    except Exception as exc:
        print(f"Embedding dependencies unavailable: {type(exc).__name__}: {exc}")
        return None

    model_names = {
        "emotion2vec": "facebook/wav2vec2-base",
        "wav2vec2": "facebook/wav2vec2-base",
        "hubert": "facebook/hubert-base-ls960",
    }
    model_name = model_names.get(backend, model_names["wav2vec2"])
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    try:
        y, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        inputs = extractor(y, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            return None
        return hidden.mean(dim=1).detach().cpu().numpy()[0].astype(np.float32)
    except Exception as exc:
        print(f"Skipping {wav_path}: {type(exc).__name__}: {exc}")
        return None


def collect_dataset(root: Path, backend: str, device: str) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for emotion in EMOTIONS:
        wavs = sorted((root / emotion).glob("*.wav"))
        if len(wavs) < 10:
            print(f"Class {emotion} has low sample count. Model may be unstable.")
        for wav in wavs:
            emb = extract_embedding(wav, backend, device)
            if emb is None:
                continue
            xs.append(emb)
            ys.append(emotion)
    if not xs:
        raise RuntimeError(f"No embeddings extracted from {root}")
    return np.stack(xs), np.asarray(ys, dtype=object)


def evaluate(model, x_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, float, float, float, np.ndarray]:
    pred = model.predict(x_val)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, pred, labels=EMOTIONS, average="macro", zero_division=0
    )
    return accuracy_score(y_val, pred), precision, recall, f1, confusion_matrix(y_val, pred, labels=EMOTIONS)


def write_metrics(path: Path, name: str, metrics: Tuple[float, float, float, float, np.ndarray]) -> None:
    acc, precision, recall, f1, cm = metrics
    rows = [
        {"model": name, "metric": "accuracy", "value": acc},
        {"model": name, "metric": "macro_precision", "value": precision},
        {"model": name, "metric": "macro_recall", "value": recall},
        {"model": name, "metric": "macro_f1", "value": f1},
        {"model": name, "metric": "confusion_matrix", "value": json.dumps(cm.tolist())},
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    with path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": name,
                "accuracy": float(acc),
                "macro_precision": float(precision),
                "macro_recall": float(recall),
                "macro_f1": float(f1),
                "confusion_matrix": cm.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT))
    parser.add_argument("--backend", choices=["emotion2vec", "wav2vec2", "hubert"], default="emotion2vec")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    x, y = collect_dataset(Path(args.input_root), args.backend, args.device)
    counts = {label: int(np.sum(y == label)) for label in EMOTIONS}
    present = [label for label, count in counts.items() if count > 0]
    if len(present) < 2:
        print("Not enough class diversity for self-training. Need at least 2 emotion classes with enough clips.")
        return 1
    stratify = y if min(counts.values()) >= 2 else None
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=stratify)
    candidates = {
        "svm": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        candidates["xgboost"] = XGBClassifier(eval_metric="mlogloss", random_state=42)
    except Exception:
        pass

    best_name = ""
    best_model = None
    best_metrics = None
    for name, model in candidates.items():
        try:
            model.fit(x_train, y_train)
            metrics = evaluate(model, x_val, y_val)
        except Exception as exc:
            print(f"Skipping {name}: {type(exc).__name__}: {exc}")
            continue
        print(f"{name}: accuracy={metrics[0]:.4f} macro_f1={metrics[3]:.4f}")
        if best_metrics is None or metrics[3] > best_metrics[3]:
            best_name, best_model, best_metrics = name, model, metrics

    if best_model is None or best_metrics is None:
        raise RuntimeError("No embedding classifier trained successfully.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(best_model, MODELS_DIR / "emotion_embedding_classifier.joblib")
    with (MODELS_DIR / "emotion_embedding_label_map.json").open("w", encoding="utf-8") as f:
        json.dump({"labels": EMOTIONS, "backend": args.backend, "best_model": best_name}, f, ensure_ascii=False, indent=2)
    write_metrics(MODELS_DIR / "emotion_embedding_metrics.csv", best_name, best_metrics)
    print(f"Saved {MODELS_DIR / 'emotion_embedding_classifier.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
