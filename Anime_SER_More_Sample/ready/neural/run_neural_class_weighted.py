"""Improvement 4: isolated class-weighted neural robustness experiment.

The validation-only ``tune`` phase deliberately does not load the test manifest.
The guarded ``evaluate`` phase consumes the frozen selection and evaluates each
weighted model once on the unchanged final test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
FROZEN_OUTPUT = ROOT / "output"
EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_CACHE = EXPERIMENT_DIR / "logmel_cache"

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]
EXPECTED_ROWS = {"train": 1313, "validation": 261, "test": 395}
RANDOM_STATE = 42
TARGET_SR = 16_000
DURATION_SECONDS = 4
N_SAMPLES = TARGET_SR * DURATION_SECONDS
N_MELS = 128
BATCH_SIZE = 32
TUNING_EPOCHS = 10
FINAL_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 5
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEARCH_SPACES = {
    "cnn": [
        {"dropout": 0.30, "learning_rate": 1e-3},
        {"dropout": 0.40, "learning_rate": 5e-4},
        {"dropout": 0.50, "learning_rate": 3e-4},
    ],
    "cnn_bilstm": [
        {"dropout": 0.30, "hidden_size": 64, "learning_rate": 1e-3},
        {"dropout": 0.40, "hidden_size": 96, "learning_rate": 5e-4},
        {"dropout": 0.40, "hidden_size": 128, "learning_rate": 3e-4},
    ],
}

DISPLAY_NAMES = {
    "cnn": "Log-Mel CNN + weighted cross-entropy",
    "cnn_bilstm": "Log-Mel CNN-BiLSTM + weighted cross-entropy",
}
BASELINE_NAMES = {"cnn": "Log-mel CNN", "cnn_bilstm": "Log-mel CNN-BiLSTM"}


def reset_random_state() -> None:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_manifest(split: str) -> pd.DataFrame:
    path = FROZEN_OUTPUT / f"{split}_manifest.csv"
    frame = pd.read_csv(path)
    if len(frame) != EXPECTED_ROWS[split]:
        raise RuntimeError(f"Frozen {split} split changed: expected {EXPECTED_ROWS[split]}, found {len(frame)}")
    required = {"file_path", "filename", "emotion", "label_id", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Frozen {split} manifest lacks columns: {sorted(missing)}")
    if set(frame["split"].astype(str).str.lower()) != {split}:
        raise RuntimeError(f"Unexpected split values inside {path}")
    expected_ids = frame["emotion"].map({name: i for i, name in enumerate(EMOTIONS)})
    if expected_ids.isna().any() or not np.array_equal(expected_ids.to_numpy(int), frame["label_id"].to_numpy(int)):
        raise RuntimeError(f"Emotion/label_id mapping changed in {path}")
    if frame["file_path"].duplicated().any():
        raise RuntimeError(f"Duplicate file paths inside {path}")
    return frame.reset_index(drop=True)


def validate_disjoint(*frames: pd.DataFrame) -> None:
    combined = pd.concat([frame[["file_path"]] for frame in frames], ignore_index=True)
    if combined["file_path"].duplicated().any():
        raise RuntimeError("Frozen train/validation/test manifests are not file-disjoint")
    missing = [path for path in combined["file_path"].astype(str) if not Path(path).is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} frozen audio path(s) do not exist; first: {missing[0]}")


def compute_class_weights(train_frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    counts = train_frame["label_id"].value_counts().reindex(range(len(EMOTIONS)), fill_value=0).to_numpy(int)
    if np.any(counts == 0):
        raise RuntimeError("Every emotion must be represented in training before class weights can be computed")
    weights = len(train_frame) / (len(EMOTIONS) * counts.astype(float))
    table = pd.DataFrame(
        {
            "emotion": EMOTIONS,
            "label_id": range(len(EMOTIONS)),
            "train_count": counts,
            "class_weight": weights,
            "formula": [f"1313 / (7 * {count})" for count in counts],
            "source_split": "train_only",
        }
    )
    return weights.astype(np.float32), table


def load_common_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    peak = np.max(np.abs(audio)) if len(audio) else 0.0
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)


def log_mel(path: str) -> np.ndarray:
    audio = librosa.util.fix_length(load_common_audio(path), size=N_SAMPLES).astype(np.float32)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=TARGET_SR, n_fft=1024, hop_length=256, n_mels=N_MELS
    )
    db = librosa.power_to_db(mel, ref=np.max)
    return ((db - db.mean()) / (db.std() + 1e-8)).astype(np.float32)


def cache_name_for(path: str) -> str:
    stat = Path(path).stat()
    identity = (
        f"v4|{path}|bytes={stat.st_size}|mtime={stat.st_mtime_ns}|sr={TARGET_SR}|"
        f"seconds={DURATION_SECONDS}|mels={N_MELS}|fft=1024|hop=256|enhance=False"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".npy"


def cached_log_mel(path: str) -> torch.Tensor:
    name = cache_name_for(path)
    frozen_cache_path = FROZEN_OUTPUT / "logmel_cache" / name
    experiment_cache_path = EXPERIMENT_CACHE / name
    feature = None
    for candidate in (frozen_cache_path, experiment_cache_path):
        if candidate.exists():
            try:
                loaded = np.load(candidate, allow_pickle=False)
                if loaded.shape == (N_MELS, 251) and np.isfinite(loaded).all():
                    feature = loaded.astype(np.float32, copy=False)
                    break
            except (OSError, ValueError):
                if candidate == experiment_cache_path:
                    candidate.unlink(missing_ok=True)
    if feature is None:
        EXPERIMENT_CACHE.mkdir(parents=True, exist_ok=True)
        feature = log_mel(path)
        np.save(experiment_cache_path, feature, allow_pickle=False)
    return torch.from_numpy(feature).unsqueeze(0)


def prepare_feature_cache(frames: list[pd.DataFrame]) -> dict[str, torch.Tensor]:
    paths = pd.concat([frame["file_path"] for frame in frames], ignore_index=True).astype(str).drop_duplicates().tolist()
    print(f"Preparing read-only-compatible log-Mel features for {len(paths)} clips ...", flush=True)
    cache: dict[str, torch.Tensor] = {}
    start = time.time()
    for index, path in enumerate(paths, start=1):
        cache[path] = cached_log_mel(path)
        if index == 1 or index % 100 == 0 or index == len(paths):
            elapsed = time.time() - start
            print(f"  features {index}/{len(paths)} ({elapsed:.1f}s elapsed)", flush=True)
    return cache


class AnimeMelDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, feature_cache: dict[str, torch.Tensor]):
        self.frame = frame.reset_index(drop=True)
        self.feature_cache = feature_cache

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        return self.feature_cache[str(row["file_path"])], int(row["label_id"])


def make_loader(frame: pd.DataFrame, cache: dict[str, torch.Tensor], shuffle: bool) -> DataLoader:
    return DataLoader(
        AnimeMelDataset(frame, cache),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


class MelCNN(nn.Module):
    def __init__(self, n_classes: int = 7, dropout: float = 0.35, hidden_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CNNBiLSTM(nn.Module):
    def __init__(self, n_classes: int = 7, dropout: float = 0.35, hidden_size: int = 96):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 1)),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size * 2, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.mean(dim=2).permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.classifier(x.mean(dim=1))


def build_model(family: str, configuration: dict) -> nn.Module:
    parameters = {key: value for key, value in configuration.items() if key != "learning_rate"}
    return MelCNN(**parameters) if family == "cnn" else CNNBiLSTM(**parameters)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    truth: list[int] = []
    predictions: list[int] = []
    for features, labels in loader:
        logits = model(features.to(DEVICE))
        truth.extend(labels.numpy().tolist())
        predictions.extend(logits.argmax(1).cpu().numpy().tolist())
    return np.asarray(truth), np.asarray(predictions)


def train_model(
    model: nn.Module,
    family: str,
    configuration: dict,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    class_weights: np.ndarray,
    max_epochs: int,
    run_label: str,
) -> tuple[nn.Module, float, list[dict]]:
    model = model.to(DEVICE)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(configuration["learning_rate"]), weight_decay=WEIGHT_DECAY
    )
    best_state = None
    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    started = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
            sample_count += len(labels)

        validation_true, validation_pred = predict(model, validation_loader)
        validation_macro_f1 = f1_score(
            validation_true, validation_pred, average="macro", zero_division=0
        )
        improved = validation_macro_f1 > best_f1
        history.append(
            {
                "model_family": family,
                "run": run_label,
                "epoch": epoch,
                "train_loss": running_loss / sample_count,
                "validation_macro_f1": validation_macro_f1,
                "is_best_epoch": improved,
            }
        )
        print(
            f"[{run_label}] epoch={epoch:02d}/{max_epochs} "
            f"loss={history[-1]['train_loss']:.4f} val_macro_f1={validation_macro_f1:.4f}",
            flush=True,
        )
        if improved:
            best_f1 = float(validation_macro_f1)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= EARLY_STOPPING_PATIENCE:
                print(f"[{run_label}] early stopping after epoch {epoch}; best epoch={best_epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError(f"No model state recorded for {run_label}")
    model.load_state_dict(best_state)
    print(f"[{run_label}] selected epoch={best_epoch}, val_macro_f1={best_f1:.6f}, elapsed={time.time()-started:.1f}s")
    return model, best_f1, history


def base_config() -> dict:
    return {
        "experiment": "Improvement 4 - Class-Weighted Neural Experiment",
        "status": "initialized",
        "seed": RANDOM_STATE,
        "device": str(DEVICE),
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "librosa": librosa.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "frozen_split_rows": EXPECTED_ROWS,
        "treatment": "nn.CrossEntropyLoss(weight=class_weights)",
        "class_weight_formula": "n_train / (n_classes * train_class_count)",
        "class_weight_source": "training data only",
        "unchanged_training": {
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "tuning_epoch_limit": TUNING_EPOCHS,
            "final_epoch_limit": FINAL_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_metric": "validation_macro_f1",
            "oversampling": False,
        },
        "unchanged_log_mel": {
            "sample_rate_hz": TARGET_SR,
            "duration_seconds": DURATION_SECONDS,
            "n_samples": N_SAMPLES,
            "mono": True,
            "peak_normalization": True,
            "experimental_enhancement": False,
            "n_fft": 1024,
            "hop_length": 256,
            "n_mels": N_MELS,
            "power_to_db_ref": "np.max",
            "standardization": "per_clip_global_mean_and_std",
        },
        "architectures": {
            "cnn": "Conv(1,16)-pool2; Conv(16,32)-pool2; Conv(32,64)-adaptive_pool4x4; FC1024-128-dropout-FC7",
            "cnn_bilstm": "three Conv-BatchNorm-ReLU blocks with pools (2,2),(2,2),(2,1); mel mean; bidirectional LSTM; mean-time; dropout-FC128-ReLU-dropout-FC7",
        },
        "search_spaces": SEARCH_SPACES,
        "test_selection_policy": "test manifest is not loaded in tune phase; one evaluation per frozen weighted model in evaluate phase",
        "fusion_modified": False,
        "ser_training_scope": ["weighted_cnn", "weighted_cnn_bilstm"],
    }


def run_smoke() -> None:
    reset_random_state()
    train = read_manifest("train")
    validation = read_manifest("validation")
    validate_disjoint(train, validation)
    weights, table = compute_class_weights(train)
    train_small = train.groupby("label_id", group_keys=False).head(1).reset_index(drop=True)
    validation_small = validation.groupby("label_id", group_keys=False).head(1).reset_index(drop=True)
    cache = prepare_feature_cache([train_small, validation_small])
    train_loader = make_loader(train_small, cache, shuffle=False)
    features, labels = next(iter(train_loader))
    for family, configuration in (("cnn", SEARCH_SPACES["cnn"][0]), ("cnn_bilstm", SEARCH_SPACES["cnn_bilstm"][0])):
        model = build_model(family, configuration).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))
        optimizer = torch.optim.AdamW(model.parameters(), lr=configuration["learning_rate"], weight_decay=WEIGHT_DECAY)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features.to(DEVICE))
        loss = criterion(logits, labels.to(DEVICE))
        loss.backward()
        optimizer.step()
        if logits.shape != (len(train_small), len(EMOTIONS)) or not torch.isfinite(loss):
            raise RuntimeError(f"Smoke test failed for {family}")
        print(f"Smoke {family}: output={tuple(logits.shape)}, weighted_loss={loss.item():.6f}")
    print(table.to_string(index=False))
    print("SMOKE PASS: train/validation only; no test manifest loaded; no full training launched.")


def run_tune() -> None:
    if (EXPERIMENT_DIR / "final_results.csv").exists():
        raise RuntimeError("Final results already exist; refusing to retune after test evaluation")
    if (EXPERIMENT_DIR / "test_evaluation_started.json").exists():
        raise RuntimeError("The one-time test gate has already been opened; refusing to retune")
    reset_random_state()
    train = read_manifest("train")
    validation = read_manifest("validation")
    validate_disjoint(train, validation)
    weights, weight_table = compute_class_weights(train)
    weight_table.to_csv(EXPERIMENT_DIR / "class_weights.csv", index=False, encoding="utf-8-sig")
    cache = prepare_feature_cache([train, validation])
    train_loader = make_loader(train, cache, shuffle=True)
    validation_loader = make_loader(validation, cache, shuffle=False)

    tuning_rows: list[dict] = []
    tuning_histories: list[dict] = []
    selected: dict[str, dict] = {}
    final_histories: dict[str, list[dict]] = {}
    for family, configurations in SEARCH_SPACES.items():
        best: tuple[float, dict, int] | None = None
        for candidate_index, configuration in enumerate(configurations, start=1):
            run_label = f"tune_{family}_{candidate_index}"
            print(f"\nTuning {DISPLAY_NAMES[family]}: {configuration}", flush=True)
            model, validation_f1, history = train_model(
                build_model(family, configuration), family, configuration,
                train_loader, validation_loader, weights, TUNING_EPOCHS, run_label,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            best_epoch = next(row["epoch"] for row in history if row["is_best_epoch"] and row["validation_macro_f1"] == validation_f1)
            tuning_rows.append(
                {
                    "model_family": family,
                    "method": DISPLAY_NAMES[family],
                    "candidate_index": candidate_index,
                    "dropout": configuration["dropout"],
                    "hidden_size": configuration.get("hidden_size", ""),
                    "learning_rate": configuration["learning_rate"],
                    "best_epoch": best_epoch,
                    "validation_macro_f1": validation_f1,
                }
            )
            tuning_histories.extend(history)
            if best is None or validation_f1 > best[0]:
                best = (validation_f1, deepcopy(configuration), best_epoch)
        assert best is not None
        selected[family] = {
            **best[1], "validation_macro_f1": best[0], "tuning_best_epoch": best[2]
        }

        # Preserve the frozen notebook's execution order: tune one architecture,
        # freeze its candidate, and immediately perform its final train/validation
        # run before beginning the next architecture. No test data is loaded here.
        final_configuration = best[1]
        print(
            f"\nFinal train/validation run for {DISPLAY_NAMES[family]} "
            f"with frozen configuration: {final_configuration}",
            flush=True,
        )
        final_model, final_validation_f1, final_history = train_model(
            build_model(family, final_configuration), family, final_configuration,
            train_loader, validation_loader, weights, FINAL_EPOCHS, f"final_{family}",
        )
        selected[family]["final_training_validation_macro_f1"] = final_validation_f1
        selected[family]["final_training_best_epoch"] = next(
            row["epoch"] for row in final_history
            if row["is_best_epoch"] and row["validation_macro_f1"] == final_validation_f1
        )
        final_histories[family] = final_history
        torch.save(final_model.state_dict(), EXPERIMENT_DIR / f"weighted_{family}.pt")
        del final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tuning_df = pd.DataFrame(tuning_rows).sort_values(
        ["model_family", "validation_macro_f1"], ascending=[True, False]
    )
    tuning_df.to_csv(EXPERIMENT_DIR / "tuning_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tuning_histories).to_csv(
        EXPERIMENT_DIR / "tuning_training_history.csv", index=False, encoding="utf-8-sig"
    )
    selected_rows = []
    for family, selection in selected.items():
        selected_rows.append(
            {
                "model_family": family,
                "method": DISPLAY_NAMES[family],
                "dropout": selection["dropout"],
                "hidden_size": selection.get("hidden_size", ""),
                "learning_rate": selection["learning_rate"],
                "selection_validation_macro_f1": selection["validation_macro_f1"],
                "tuning_best_epoch": selection["tuning_best_epoch"],
                "final_training_validation_macro_f1": selection["final_training_validation_macro_f1"],
                "final_training_best_epoch": selection["final_training_best_epoch"],
                "selection_data": "train_and_validation_only",
            }
        )
    pd.DataFrame(selected_rows).to_csv(
        EXPERIMENT_DIR / "selected_hyperparameters.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(final_histories["cnn"]).to_csv(
        EXPERIMENT_DIR / "training_history_cnn.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(final_histories["cnn_bilstm"]).to_csv(
        EXPERIMENT_DIR / "training_history_cnn_bilstm.csv", index=False, encoding="utf-8-sig"
    )
    config = base_config()
    config.update(
        {
            "status": "weighted_models_frozen_test_not_loaded",
            "manifest_sha256": {
                "train": sha256_file(FROZEN_OUTPUT / "train_manifest.csv"),
                "validation": sha256_file(FROZEN_OUTPUT / "validation_manifest.csv"),
                "test": None,
            },
            "class_weights": {emotion: float(weight) for emotion, weight in zip(EMOTIONS, weights)},
            "selected_configurations": selected,
            "test_manifest_loaded": False,
            "test_evaluations_per_model": 0,
        }
    )
    atomic_json(EXPERIMENT_DIR / "experiment_config.json", config)
    print("\nVALIDATION SELECTION AND FINAL TRAIN/VALIDATION COMPLETE. The test manifest was not loaded.")
    for family, selection in selected.items():
        print(f"  {DISPLAY_NAMES[family]}: {selection}")
    print("Both checkpoints are frozen. Run --phase evaluate for the one-time held-out test evaluation.")


def metric_row(family: str, truth: np.ndarray, prediction: np.ndarray, validation_f1: float) -> dict:
    return {
        "model_family": family,
        "method": DISPLAY_NAMES[family],
        "validation_macro_f1": validation_f1,
        "test_accuracy": accuracy_score(truth, prediction),
        "test_macro_precision": precision_score(truth, prediction, average="macro", zero_division=0),
        "test_macro_recall": recall_score(truth, prediction, average="macro", zero_division=0),
        "test_macro_f1": f1_score(truth, prediction, average="macro", zero_division=0),
        "test_size": len(truth),
        "test_evaluation_number": 1,
    }


def write_readme(final_df: pd.DataFrame, comparison: pd.DataFrame, config: dict) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        rendered = frame.copy()
        for column in rendered.columns:
            if pd.api.types.is_float_dtype(rendered[column]):
                rendered[column] = rendered[column].map(lambda value: f"{value:.6f}")
        rendered = rendered.fillna("").astype(str).map(
            lambda value: value.replace("|", "\\|").replace("\n", " ")
        )
        header = "| " + " | ".join(rendered.columns) + " |"
        divider = "| " + " | ".join(["---"] * len(rendered.columns)) + " |"
        rows = ["| " + " | ".join(row) + " |" for row in rendered.to_numpy().tolist()]
        return "\n".join([header, divider, *rows])

    lines = [
        "# Improvement 4: Class-Weighted Neural Experiment",
        "",
        "This isolated robustness experiment changes only the neural cross-entropy loss by applying inverse-frequency class weights computed from the 1,313 training samples. The frozen dataset, split, preprocessing, architectures, unweighted outputs, text baseline, speaker analysis, pseudo-labeling, and fusion were not changed.",
        "",
        "## Selection and test protocol",
        "",
        "The three existing documented candidates per architecture were compared using validation Macro-F1 only. The test manifest was not loaded during selection. After configurations were frozen, each final weighted model was evaluated once on the unchanged 395-sample test set.",
        "",
        "## Selected configurations",
        "",
    ]
    for family, selection in config["selected_configurations"].items():
        lines.append(f"- {DISPLAY_NAMES[family]}: `{json.dumps(selection, sort_keys=True)}`")
    lines.extend(["", "## Final results", "", markdown_table(final_df), "", "## Comparison with frozen unweighted models", "", markdown_table(comparison), "", "See `classwise_results.csv` and the two confusion-matrix CSV files for class-level behavior.", ""])
    (EXPERIMENT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_report() -> None:
    config_path = EXPERIMENT_DIR / "experiment_config.json"
    final_path = EXPERIMENT_DIR / "final_results.csv"
    comparison_path = EXPERIMENT_DIR / "comparison_with_unweighted.csv"
    for path in (config_path, final_path, comparison_path):
        if not path.exists():
            raise RuntimeError(f"Cannot rebuild report; missing {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "complete" or config.get("test_evaluations_per_model") != 1:
        raise RuntimeError("Report-only phase requires a completed one-time evaluation")
    final_df = pd.read_csv(final_path)
    comparison_df = pd.read_csv(comparison_path)
    write_readme(final_df, comparison_df, config)
    gate_path = EXPERIMENT_DIR / "test_evaluation_started.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate.update({"status": "one_time_test_evaluation_complete", "final_results_sha256": sha256_file(final_path)})
    atomic_json(gate_path, gate)
    print("REPORT REBUILT from saved results; no model training or test evaluation performed.")


def run_evaluate() -> None:
    final_path = EXPERIMENT_DIR / "final_results.csv"
    if final_path.exists():
        raise RuntimeError("Final test results already exist; refusing a second test evaluation")
    test_gate = EXPERIMENT_DIR / "test_evaluation_started.json"
    if test_gate.exists():
        raise RuntimeError(
            "The one-time test gate was already opened. Refusing to risk a repeated test evaluation."
        )
    config_path = EXPERIMENT_DIR / "experiment_config.json"
    if not config_path.exists():
        raise RuntimeError("Run --phase tune first")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "weighted_models_frozen_test_not_loaded":
        raise RuntimeError(f"Unexpected experiment status: {config.get('status')}")

    train = read_manifest("train")
    validation = read_manifest("validation")
    weights, recomputed_table = compute_class_weights(train)
    stored_table = pd.read_csv(EXPERIMENT_DIR / "class_weights.csv")
    if not np.allclose(stored_table["class_weight"].to_numpy(float), weights, rtol=0, atol=1e-7):
        raise RuntimeError("Stored training-only class weights do not match recomputation")
    if sha256_file(FROZEN_OUTPUT / "train_manifest.csv") != config["manifest_sha256"]["train"]:
        raise RuntimeError("Frozen train manifest changed after selection")
    if sha256_file(FROZEN_OUTPUT / "validation_manifest.csv") != config["manifest_sha256"]["validation"]:
        raise RuntimeError("Frozen validation manifest changed after selection")

    checkpoints = {
        family: EXPERIMENT_DIR / f"weighted_{family}.pt" for family in ("cnn", "cnn_bilstm")
    }
    histories = {
        family: EXPERIMENT_DIR / f"training_history_{family}.csv"
        for family in ("cnn", "cnn_bilstm")
    }
    missing = [str(path) for path in [*checkpoints.values(), *histories.values()] if not path.exists()]
    if missing:
        raise RuntimeError(f"Frozen train/validation artifacts are incomplete: {missing}")

    # The durable gate is written before parsing the held-out data. If the process
    # is interrupted after this point, an automatic rerun is deliberately refused.
    atomic_json(
        test_gate,
        {
            "status": "one_time_test_evaluation_started",
            "selected_configurations_sha256": sha256_file(
                EXPERIMENT_DIR / "selected_hyperparameters.csv"
            ),
            "models": {family: sha256_file(path) for family, path in checkpoints.items()},
        },
    )

    # This is the first point at which this experiment parses the held-out test manifest.
    test = read_manifest("test")
    validate_disjoint(train, validation, test)
    print("TEST GATE OPENED: configurations and model checkpoints are frozen; evaluating once per model.")
    cache = prepare_feature_cache([test])
    test_loader = make_loader(test, cache, shuffle=False)

    result_rows: list[dict] = []
    classwise_rows: list[dict] = []
    for family in ("cnn", "cnn_bilstm"):
        selection = config["selected_configurations"][family]
        configuration = {
            key: selection[key] for key in ("dropout", "hidden_size", "learning_rate") if key in selection
        }
        print(f"\nOne-time test evaluation for {DISPLAY_NAMES[family]}: {configuration}")
        model = build_model(family, configuration).to(DEVICE)
        model.load_state_dict(torch.load(checkpoints[family], map_location=DEVICE, weights_only=True))
        test_true, test_pred = predict(model, test_loader)
        if not np.array_equal(test_true, test["label_id"].to_numpy(int)):
            raise RuntimeError("Test loader order or membership changed")
        result_rows.append(
            metric_row(
                family,
                test_true,
                test_pred,
                float(selection["final_training_validation_macro_f1"]),
            )
        )
        precision, recall, f1_values, support = precision_recall_fscore_support(
            test_true, test_pred, labels=range(len(EMOTIONS)), zero_division=0
        )
        for label_id, emotion in enumerate(EMOTIONS):
            classwise_rows.append(
                {
                    "model_family": family,
                    "method": DISPLAY_NAMES[family],
                    "emotion": emotion,
                    "label_id": label_id,
                    "precision": precision[label_id],
                    "recall": recall[label_id],
                    "f1": f1_values[label_id],
                    "support": int(support[label_id]),
                }
            )
        matrix = pd.DataFrame(
            confusion_matrix(test_true, test_pred, labels=range(len(EMOTIONS))),
            index=EMOTIONS,
            columns=EMOTIONS,
        )
        matrix.index.name = "true_emotion"
        matrix.columns.name = "predicted_emotion"
        matrix.to_csv(EXPERIMENT_DIR / f"confusion_matrix_{family}.csv", encoding="utf-8-sig")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_df = pd.DataFrame(result_rows)
    classwise_df = pd.DataFrame(classwise_rows)
    baseline = pd.read_csv(FROZEN_OUTPUT / "anime_only_final_results.csv")
    baseline_classwise = pd.read_csv(FROZEN_OUTPUT / "classwise_results.csv")
    baseline_selected = pd.read_csv(FROZEN_OUTPUT / "selected_hyperparameters.csv")
    unweighted_f1_lookup = {
        (family, row["emotion"]): float(row["f1"])
        for family in ("cnn", "cnn_bilstm")
        for _, row in baseline_classwise.loc[
            baseline_classwise["method"] == BASELINE_NAMES[family]
        ].iterrows()
    }
    classwise_df["unweighted_f1"] = [
        unweighted_f1_lookup[(row.model_family, row.emotion)]
        for row in classwise_df.itertuples(index=False)
    ]
    classwise_df["f1_difference_from_unweighted"] = (
        classwise_df["f1"] - classwise_df["unweighted_f1"]
    )
    classwise_df["previously_zero_f1"] = classwise_df["unweighted_f1"].eq(0)
    comparison_rows = []
    for result in result_rows:
        family = result["model_family"]
        old = baseline.loc[baseline["method"] == BASELINE_NAMES[family]].iloc[0]
        old_selection = baseline_selected.loc[
            baseline_selected["method"] == BASELINE_NAMES[family]
        ].iloc[0]
        old_classes = baseline_classwise.loc[baseline_classwise["method"] == BASELINE_NAMES[family]]
        new_classes = classwise_df.loc[classwise_df["model_family"] == family]
        old_zero = sorted(old_classes.loc[old_classes["f1"] == 0, "emotion"].tolist())
        improved_zero = sorted(
            new_classes.loc[new_classes["emotion"].isin(old_zero) & (new_classes["f1"] > 0), "emotion"].tolist()
        )
        comparison_rows.append(
            {
                "model_family": family,
                "unweighted_method": BASELINE_NAMES[family],
                "weighted_method": DISPLAY_NAMES[family],
                "unweighted_selection_validation_macro_f1": old_selection["validation_macro_f1"],
                "weighted_selection_validation_macro_f1": config["selected_configurations"][family]["validation_macro_f1"],
                "weighted_final_training_validation_macro_f1": result["validation_macro_f1"],
                "unweighted_test_accuracy": old["accuracy"],
                "weighted_test_accuracy": result["test_accuracy"],
                "accuracy_difference": result["test_accuracy"] - old["accuracy"],
                "unweighted_test_macro_f1": old["macro_f1"],
                "weighted_test_macro_f1": result["test_macro_f1"],
                "macro_f1_difference": result["test_macro_f1"] - old["macro_f1"],
                "unweighted_zero_f1_classes": ";".join(old_zero),
                "previously_zero_classes_improved": ";".join(improved_zero),
                "weighted_is_better_by_macro_f1": result["test_macro_f1"] > old["macro_f1"],
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)

    # Write final tables only after both models have completed their one permitted test evaluation.
    final_df.to_csv(final_path, index=False, encoding="utf-8-sig")
    classwise_df.to_csv(EXPERIMENT_DIR / "classwise_results.csv", index=False, encoding="utf-8-sig")
    comparison_df.to_csv(
        EXPERIMENT_DIR / "comparison_with_unweighted.csv", index=False, encoding="utf-8-sig"
    )
    config.update(
        {
            "status": "complete",
            "manifest_sha256": {
                **config["manifest_sha256"],
                "test": sha256_file(FROZEN_OUTPUT / "test_manifest.csv"),
            },
            "test_manifest_loaded": True,
            "test_evaluations_per_model": 1,
            "test_size": len(test),
            "final_results": result_rows,
        }
    )
    atomic_json(config_path, config)
    write_readme(final_df, comparison_df, config)
    print("\nFINAL WEIGHTED RESULTS")
    print(final_df.to_string(index=False))
    print("\nCOMPARISON WITH FROZEN UNWEIGHTED RESULTS")
    print(comparison_df.to_string(index=False))
    print("Improvement 4 complete. No fusion or other experiment was run.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "tune", "evaluate", "report"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Improvement 4 phase={args.phase}; device={DEVICE}; root={ROOT}", flush=True)
    if args.phase == "smoke":
        run_smoke()
    elif args.phase == "tune":
        run_tune()
    elif args.phase == "evaluate":
        run_evaluate()
    else:
        run_report()


if __name__ == "__main__":
    main()
