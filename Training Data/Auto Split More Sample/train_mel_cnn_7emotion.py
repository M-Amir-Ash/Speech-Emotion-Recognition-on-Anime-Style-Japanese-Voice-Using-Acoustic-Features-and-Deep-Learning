import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "output" / "final_character_emotion_dataset"
MODELS_DIR = BASE_DIR / "models"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant_Surprise", "Sad"]
TARGET_SR = 16000
N_MELS = 128
MAX_DURATION_SEC = 4.0


class SimpleMelCNN(nn.Module):
    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.head(self.net(x).flatten(1))


def log_mel(wav_path: Path) -> np.ndarray:
    y, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    target_len = int(TARGET_SR * MAX_DURATION_SEC)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    mel = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MELS, power=2.0)
    arr = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    return (arr - float(arr.mean())) / (float(arr.std()) + 1e-8)


def collect_dataset(root: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    xs, ys, files = [], [], []
    for idx, emotion in enumerate(EMOTIONS):
        wavs = sorted((root / emotion).glob("*.wav"))
        if len(wavs) < 10:
            print(f"Class {emotion} has low sample count. Model may be unstable.")
        for wav in wavs:
            try:
                xs.append(log_mel(wav))
                ys.append(idx)
                files.append(str(wav))
            except Exception as exc:
                print(f"Skipping {wav}: {type(exc).__name__}: {exc}")
    if not xs:
        raise RuntimeError(f"No wav files found in {root}")
    return np.stack(xs), np.asarray(ys, dtype=np.int64), files


def write_metrics(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(EMOTIONS))), average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTIONS))))
    rows = [
        {"metric": "accuracy", "value": accuracy_score(y_true, y_pred)},
        {"metric": "macro_precision", "value": precision},
        {"metric": "macro_recall", "value": recall},
        {"metric": "macro_f1", "value": f1},
        {"metric": "confusion_matrix", "value": json.dumps(cm.tolist())},
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    json_path = path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": float(accuracy_score(y_true, y_pred)),
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = Path(args.input_root)
    x, y, _files = collect_dataset(root)
    present_classes = [emotion for idx, emotion in enumerate(EMOTIONS) if int(np.sum(y == idx)) > 0]
    if len(present_classes) < 2:
        print("Not enough class diversity for self-training. Need at least 2 emotion classes with enough clips.")
        return 1
    stratify = y if min(np.bincount(y, minlength=len(EMOTIONS))) >= 2 else None
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=stratify)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    model = SimpleMelCNN(len(EMOTIONS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.from_numpy(x_train).unsqueeze(1), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
        print(f"epoch={epoch} train_loss={total / max(1, len(train_ds)):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_val).unsqueeze(1).to(device))
        pred = logits.argmax(dim=1).cpu().numpy()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "label_map": EMOTIONS}, MODELS_DIR / "mel_cnn_7emotion.pt")
    with (MODELS_DIR / "mel_cnn_label_map.json").open("w", encoding="utf-8") as f:
        json.dump(EMOTIONS, f, ensure_ascii=False, indent=2)
    write_metrics(MODELS_DIR / "mel_cnn_metrics.csv", y_val, pred)
    print(f"Saved {MODELS_DIR / 'mel_cnn_7emotion.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
