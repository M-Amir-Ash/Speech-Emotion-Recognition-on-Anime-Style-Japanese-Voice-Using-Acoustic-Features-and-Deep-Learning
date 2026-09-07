import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import multimodal_emotion_classifier as mec

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE = BASE_DIR / "output" / "pipeline_workspace"
SEGMENT_ROOT = WORKSPACE / "segmented"
INPUT_DIRS = [
    SEGMENT_ROOT / "japanese_dialogue",
    SEGMENT_ROOT / "character_vocal_effect",
]
SENTENCE_CLIP_DIR = SEGMENT_ROOT / "whisperx_sentence_clips"
REJECTED_SEGMENT_DIR = SEGMENT_ROOT / "rejected_low_quality"
QUALITY_MANIFEST = SEGMENT_ROOT / "_quality_manifest.csv"
OUTPUT_ROOT = WORKSPACE / "emotion_text_guided_classified"
MODELS_DIR = BASE_DIR / "models"

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Pleasant_Surprise",
    "Sad",
]
OUTPUT_FOLDERS = EMOTIONS + ["Needs_Review", "Rejected"]

VOCAL_REACTION_CUES = {
    "あ", "あっ", "え", "えっ", "はっ", "うわ", "うわっ", "きゃ", "きゃあ",
    "おお", "へえ", "ah", "oh", "eh", "gasp", "scream",
}

KEYWORD_CUES = {
    "Angry": ["ふざけ", "うるさい", "バカ", "くそ", "許せない", "怒", "なんだよ"],
    "Disgust": ["気持ち悪", "最悪", "嫌", "うざ", "汚", "吐き気"],
    "Fear": ["怖", "やばい", "助け", "逃げ", "無理", "こわい", "いや", "やめて"],
    "Happy": ["嬉しい", "楽しい", "最高", "よかった", "好き", "やった"],
    "Pleasant_Surprise": ["すごい", "やった", "ほんとう", "まさか", "びっくり", "驚", "えっ", "うそ", "おお"],
    "Sad": ["悲しい", "つらい", "寂しい", "泣", "ごめん", "しんどい"],
}

COMPATIBLE = {
    "Angry": {"Disgust", "Fear"},
    "Disgust": {"Angry"},
    "Fear": {"Sad", "Angry", "Pleasant_Surprise"},
    "Happy": {"Pleasant_Surprise", "Neutral"},
    "Neutral": {"Happy", "Sad"},
    "Pleasant_Surprise": {"Happy", "Fear"},
    "Sad": {"Fear", "Neutral"},
}

MEL_CNN_MODEL = None
MEL_CNN_WARNING_PRINTED = False
EMBEDDING_CLASSIFIER = None
EMBEDDING_WARNING_PRINTED = False


if nn is not None:
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
            x = self.net(x).flatten(1)
            return self.head(x)
else:
    SimpleMelCNN = None


def ensure_dirs() -> None:
    for path in (OUTPUT_ROOT, SENTENCE_CLIP_DIR, REJECTED_SEGMENT_DIR):
        if path.exists():
            shutil.rmtree(path)
    SENTENCE_CLIP_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for folder in OUTPUT_FOLDERS:
        (OUTPUT_ROOT / folder).mkdir(parents=True, exist_ok=True)


def compact(text: object) -> str:
    return mec.compact_text_for_csv("" if text is None else str(text))


def safe_copy_path(dst_dir: Path, name: str) -> Path:
    stem = Path(name).stem
    suffix = Path(name).suffix or ".wav"
    candidate = dst_dir / f"{mec.safe_name(stem)}{suffix}"
    idx = 1
    while candidate.exists():
        candidate = dst_dir / f"{mec.safe_name(stem)}_copy_{idx:02d}{suffix}"
        idx += 1
    return candidate


def config_section(*keys: str) -> Dict[str, object]:
    value: object = mec.CFG
    for key in keys:
        if not isinstance(value, dict):
            return {}
        value = value.get(key, {})
    return value if isinstance(value, dict) else {}


def resolve_config_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


def empty_scores() -> Dict[str, float]:
    return {label: 0.0 for label in EMOTIONS}


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    active = {label: max(0.0, float(scores.get(label, 0.0))) for label in EMOTIONS}
    total = sum(active.values())
    if total <= 1e-12:
        return {}
    return {label: value / total for label, value in active.items()}


def read_manifest_rows(folder: Path) -> Dict[str, Dict[str, str]]:
    manifest = folder / "_segments_manifest.csv"
    if not manifest.is_file():
        return {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        return {row.get("filename", ""): row for row in csv.DictReader(f)}


def collect_items() -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for folder in INPUT_DIRS:
        if not folder.is_dir():
            continue
        manifest_rows = read_manifest_rows(folder)
        for wav in sorted(folder.glob("*.wav")):
            row = manifest_rows.get(wav.name, {})
            items.append(
                {
                    "file_path": wav,
                    "file_name": wav.name,
                    "source_group": folder.name,
                    "manifest": row,
                }
            )
    return items


def float_or_none(value: object) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def japanese_char_count(text: str) -> int:
    return sum(1 for c in text if "\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff")


def normalized_reaction_text(text: str) -> str:
    return compact(text).lower().strip("。、！？!?…ー~～ ")


def is_short_vocal_reaction(transcript: str, duration: float, source_group: str) -> bool:
    norm = normalized_reaction_text(transcript)
    if source_group == "character_vocal_effect" and duration <= 3.0:
        return True
    if norm in VOCAL_REACTION_CUES:
        return True
    if len(norm) <= 3 and any(cue in norm for cue in VOCAL_REACTION_CUES):
        return True
    return duration <= 2.2 and len(norm) <= 5 and japanese_char_count(norm) <= 4


def has_meaningful_transcript(transcript: str, rel: float) -> bool:
    text = compact(transcript)
    return bool(text) and japanese_char_count(text) >= 2 and len(text) >= 4 and rel >= 0.35


def detect_clip_type(transcript: str, duration: float, rel: float, source_group: str = "") -> str:
    if duration <= 0 or duration > 8.0:
        return "bad_audio"
    if 0.5 <= duration <= 3.0 and is_short_vocal_reaction(transcript, duration, source_group):
        return "short_vocal_reaction"
    if has_meaningful_transcript(transcript, rel):
        return "normal_dialogue"
    return "bad_audio"


def quality_gate(feats: Dict[str, float], no_speech_prob: Optional[float], clip_type: str) -> Tuple[str, str]:
    duration = float(feats.get("duration_sec", 0.0))
    voiced = float(feats.get("voiced_ratio", 0.0))
    harm = float(feats.get("harmonic_ratio", 0.0))
    flat = float(feats.get("flatness_mean", 0.0))
    zcr = float(feats.get("zcr_mean", 0.0))
    rms = float(feats.get("rms_mean", 0.0))
    reasons: List[str] = []
    if duration < 0.5:
        reasons.append("duration_lt_0.5")
    if duration > 8.0:
        reasons.append("duration_gt_8.0")
    if voiced < 0.18:
        reasons.append("very_low_voiced_ratio")
    if harm < 0.05 and clip_type != "short_vocal_reaction":
        reasons.append("very_low_harmonic_ratio")
    if flat > 0.22 and harm < 0.12:
        reasons.append("flat_noise_like")
    if zcr > 0.38 and flat > 0.12:
        reasons.append("very_high_zcr_noise")
    if rms < 0.0025:
        reasons.append("too_low_rms")
    if no_speech_prob is not None and no_speech_prob > 0.85:
        reasons.append("high_no_speech_prob")
    if flat > 0.35 and zcr > 0.30:
        reasons.append("possible_uvr_artifact")
    return ("reject", "|".join(reasons)) if reasons else ("pass", "")


def keyword_scores(transcript: str) -> Dict[str, float]:
    scores = {label: 0.0 for label in EMOTIONS}
    for label, cues in KEYWORD_CUES.items():
        hits = sum(1 for cue in cues if cue in transcript)
        if hits:
            scores[label] += min(0.65, 0.22 * hits)
    return mec.normalize_scores(scores) if sum(scores.values()) > 0 else {}


def text_candidate_scores(transcript: str, clip_type: str, rel: float) -> Dict[str, float]:
    if clip_type == "short_vocal_reaction":
        return keyword_scores(transcript)
    if not has_meaningful_transcript(transcript, rel):
        return keyword_scores(transcript)
    model_scores = mec.text_emotion_scores(transcript)
    cues = keyword_scores(transcript)
    if not model_scores:
        return cues
    merged = {label: float(model_scores.get(label, 0.0)) for label in EMOTIONS}
    for label, value in cues.items():
        merged[label] = merged.get(label, 0.0) + 0.35 * float(value)
    return mec.normalize_scores(merged)


def load_wav_fixed(wav_path: str, duration_sec: float, target_sr: int) -> np.ndarray:
    y, _sr = librosa.load(wav_path, sr=target_sr, mono=True)
    target_len = int(round(duration_sec * target_sr))
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]
    return y.astype(np.float32, copy=False)


def log_mel_tensor(wav_path: str, n_mels: int, duration_sec: float, target_sr: int):
    if torch is None:
        return None
    y = load_wav_fixed(wav_path, duration_sec, target_sr)
    mel = librosa.feature.melspectrogram(y=y, sr=target_sr, n_mels=n_mels, power=2.0)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - float(np.mean(log_mel))) / (float(np.std(log_mel)) + 1e-8)
    return torch.from_numpy(log_mel.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def mel_cnn_emotion_scores(wav_path: str) -> Dict[str, float]:
    global MEL_CNN_MODEL, MEL_CNN_WARNING_PRINTED
    cfg = config_section("models", "mel_cnn")
    if not cfg.get("enabled", True) or torch is None or SimpleMelCNN is None:
        return {}

    checkpoint_path = resolve_config_path(str(cfg.get("checkpoint_path", "models/mel_cnn_7emotion.pt")))
    if not checkpoint_path.is_file():
        if not MEL_CNN_WARNING_PRINTED:
            print("Mel-CNN checkpoint not found. Skipping Mel-CNN branch.", flush=True)
            MEL_CNN_WARNING_PRINTED = True
        return {}

    device = str(cfg.get("device", "cuda"))
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    try:
        if MEL_CNN_MODEL is None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            label_map = checkpoint.get("label_map", EMOTIONS) if isinstance(checkpoint, dict) else EMOTIONS
            state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model = SimpleMelCNN(num_classes=len(label_map)).to(device)
            model.load_state_dict(state)
            model.eval()
            MEL_CNN_MODEL = {"model": model, "device": device, "label_map": list(label_map)}
        tensor = log_mel_tensor(
            wav_path,
            int(cfg.get("n_mels", 128)),
            float(cfg.get("max_duration_sec", 4.0)),
            int(config_section("audio").get("target_sr", 16000)),
        )
        if tensor is None:
            return {}
        with torch.no_grad():
            logits = MEL_CNN_MODEL["model"](tensor.to(MEL_CNN_MODEL["device"]))[0]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        raw = {}
        for label, prob in zip(MEL_CNN_MODEL["label_map"], probs):
            if label in EMOTIONS:
                raw[label] = float(prob)
        return normalize_scores(raw)
    except Exception as exc:
        if not MEL_CNN_WARNING_PRINTED:
            print(f"Mel-CNN branch failed. Skipping Mel-CNN branch. Reason: {type(exc).__name__}: {exc}", flush=True)
            MEL_CNN_WARNING_PRINTED = True
        return {}


def extract_embedding(wav_path: str) -> Optional[np.ndarray]:
    if torch is None:
        return None
    try:
        from transformers import AutoFeatureExtractor, AutoModel

        cfg = config_section("models", "embedding_emotion")
        backend = str(cfg.get("backend", "emotion2vec")).lower()
        candidates = [backend] + [str(x).lower() for x in cfg.get("fallback_backends", ["wav2vec2", "hubert"])]
        model_names = {
            "emotion2vec": "facebook/wav2vec2-base",
            "wav2vec2": "facebook/wav2vec2-base",
            "hubert": "facebook/hubert-base-ls960",
        }
        device = str(cfg.get("device", "cuda"))
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        y, _ = librosa.load(wav_path, sr=int(config_section("audio").get("target_sr", 16000)), mono=True)
        for item in candidates:
            name = str(cfg.get("checkpoint_path") or "").strip() or model_names.get(item)
            if not name:
                continue
            try:
                extractor = AutoFeatureExtractor.from_pretrained(name)
                model = AutoModel.from_pretrained(name).to(device).eval()
                inputs = extractor(y, sampling_rate=int(config_section("audio").get("target_sr", 16000)), return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model(**inputs)
                hidden = getattr(out, "last_hidden_state", None)
                if hidden is None:
                    continue
                return hidden.mean(dim=1).detach().cpu().numpy()[0].astype(np.float32)
            except Exception:
                continue
    except Exception:
        return None
    return None


def embedding_emotion_scores(wav_path: str) -> Dict[str, float]:
    global EMBEDDING_CLASSIFIER, EMBEDDING_WARNING_PRINTED
    cfg = config_section("models", "embedding_emotion")
    if not cfg.get("enabled", True):
        return {}
    classifier_path = resolve_config_path(str(cfg.get("classifier_path", "models/emotion_embedding_classifier.joblib")))
    if not classifier_path.is_file():
        if not EMBEDDING_WARNING_PRINTED:
            print("Embedding emotion classifier not found. Skipping embedding emotion branch.", flush=True)
            EMBEDDING_WARNING_PRINTED = True
        return {}
    try:
        if EMBEDDING_CLASSIFIER is None:
            import joblib
            EMBEDDING_CLASSIFIER = joblib.load(classifier_path)
        emb = extract_embedding(wav_path)
        if emb is None:
            return {}
        classifier = EMBEDDING_CLASSIFIER
        probs = classifier.predict_proba([emb])[0]
        labels = getattr(classifier, "classes_", EMOTIONS)
        raw = {str(label): float(prob) for label, prob in zip(labels, probs) if str(label) in EMOTIONS}
        return normalize_scores(raw)
    except Exception as exc:
        if not EMBEDDING_WARNING_PRINTED:
            print(f"Embedding emotion branch failed. Skipping embedding emotion branch. Reason: {type(exc).__name__}: {exc}", flush=True)
            EMBEDDING_WARNING_PRINTED = True
        return {}


def audio_candidate_scores(file_path: Path) -> Dict[str, float]:
    # Uses only labels supported by the loaded model's configured mapping.
    if not config_section("models", "audio_emotion_baseline").get("enabled", True):
        return {}
    return mec.audio_emotion_scores(str(file_path))


def prosody_scores(feats: Dict[str, float], scaler: Dict[str, Dict[str, float]], transcript: str, clip_type: str) -> Dict[str, float]:
    if not config_section("models", "prosody").get("enabled", True):
        return {}
    scores = {label: 0.0 for label in EMOTIONS}
    z_f0_mean = mec.zscore_feature(feats["f0_mean"], scaler["f0_mean"]["mean"], scaler["f0_mean"]["std"])
    z_f0_std = mec.zscore_feature(feats["f0_std"], scaler["f0_std"]["mean"], scaler["f0_std"]["std"])
    z_rms = mec.zscore_feature(feats["rms_mean"], scaler["rms_mean"]["mean"], scaler["rms_mean"]["std"])
    z_rms_std = mec.zscore_feature(feats["rms_std"], scaler["rms_std"]["mean"], scaler["rms_std"]["std"])
    z_burst = mec.zscore_feature(feats["burst_ratio"], scaler["burst_ratio"]["mean"], scaler["burst_ratio"]["std"])
    z_zcr = mec.zscore_feature(feats["zcr_mean"], scaler["zcr_mean"]["mean"], scaler["zcr_mean"]["std"])
    z_harm = mec.zscore_feature(feats["harmonic_ratio"], scaler["harmonic_ratio"]["mean"], scaler["harmonic_ratio"]["std"])
    duration = float(feats.get("duration_sec", 0.0))
    voiced = float(feats.get("voiced_ratio", 0.0))
    reaction = clip_type == "short_vocal_reaction"
    cues = keyword_scores(transcript)

    if z_rms > 0.8 and z_burst > 0.7 and z_f0_std > 0.45 and z_harm > -1.2:
        if float(cues.get("Angry", 0.0)) > 0:
            scores["Angry"] += 0.45
        elif not reaction and z_f0_mean < 0.75 and z_f0_std < 1.35:
            scores["Angry"] += 0.30
            scores["Fear"] += 0.10
        else:
            scores["Fear"] += 0.25
            scores["Pleasant_Surprise"] += 0.20
    if z_f0_mean > 0.7 and z_f0_std > 0.7 and (z_rms_std > 0.35 or reaction):
        scores["Fear"] += 0.45
    if reaction and z_zcr > 0.2 and z_harm > -1.5 and z_f0_std > 0.25:
        scores["Fear"] += 0.35
    if reaction and voiced > 0.45 and z_f0_mean > 0.35 and z_f0_std > 0.45 and z_rms_std > 0.25:
        scores["Fear"] += 0.45
    if reaction and duration <= 2.2 and z_burst > 0.75 and z_f0_std > 0.65:
        scores["Fear"] += 0.30
    if reaction and duration <= 3.0 and z_f0_mean > 0.45 and z_burst > 0.25:
        scores["Pleasant_Surprise"] += 0.45
    if reaction and duration <= 2.5 and z_f0_mean > 0.55 and z_f0_std > 0.35 and z_rms > -0.4:
        scores["Pleasant_Surprise"] += 0.35
    if reaction and float(cues.get("Pleasant_Surprise", 0.0)) > 0:
        scores["Pleasant_Surprise"] += 0.50
    if float(cues.get("Pleasant_Surprise", 0.0)) > 0 and z_f0_std > 0.15:
        scores["Pleasant_Surprise"] += 0.35
    if z_rms < -0.45 and z_f0_mean < -0.35 and duration >= 1.2:
        scores["Sad"] += 0.55
    if abs(z_rms) < 0.65 and abs(z_f0_std) < 0.65 and abs(z_burst) < 0.9 and not reaction:
        scores["Neutral"] += 0.55
    if float(cues.get("Disgust", 0.0)) > 0 and z_harm < 0.5:
        scores["Disgust"] += 0.45
    if float(cues.get("Happy", 0.0)) > 0 and z_rms > -0.6:
        scores["Happy"] += 0.35

    return mec.normalize_scores(scores) if sum(scores.values()) > 0 else {}


def weights_for(clip_type: str, rel: float) -> Dict[str, float]:
    fusion_cfg = config_section("fusion")
    if clip_type == "short_vocal_reaction":
        return dict(fusion_cfg.get("short_vocal_reaction", {})) or {
            "mel_cnn": 0.35, "embedding": 0.30, "audio_baseline": 0.10, "prosody": 0.25, "text": 0.0
        }
    if rel < 0.45:
        return dict(fusion_cfg.get("low_transcript_reliability", {})) or {
            "mel_cnn": 0.35, "embedding": 0.30, "audio_baseline": 0.10, "prosody": 0.25, "text": 0.0
        }
    return dict(fusion_cfg.get("normal_dialogue", {})) or {
        "mel_cnn": 0.30, "embedding": 0.30, "audio_baseline": 0.15, "prosody": 0.20, "text": 0.05
    }


def branch_top(scores: Dict[str, float]) -> str:
    return mec.top2(scores)[0][0] if scores else "unknown"


def adaptive_fusion(
    mel_cnn_scores: Dict[str, float],
    embedding_scores: Dict[str, float],
    audio_baseline_scores: Dict[str, float],
    text_scores: Dict[str, float],
    prosody: Dict[str, float],
    clip_type: str,
    rel: float,
) -> Dict[str, float]:
    weights = weights_for(clip_type, rel)
    branches = {
        "mel_cnn": mel_cnn_scores,
        "embedding": embedding_scores,
        "audio_baseline": audio_baseline_scores,
        "prosody": prosody,
        "text": text_scores,
    }
    if not mel_cnn_scores and not embedding_scores:
        weights = {
            "mel_cnn": 0.0,
            "embedding": 0.0,
            "audio_baseline": 0.30,
            "prosody": 0.60,
            "text": 0.10 if clip_type == "normal_dialogue" and rel >= 0.45 else 0.0,
        }
    else:
        if not mel_cnn_scores:
            missing = float(weights.get("mel_cnn", 0.0))
            weights["mel_cnn"] = 0.0
            weights["embedding"] = float(weights.get("embedding", 0.0)) + (missing * 0.60)
            weights["prosody"] = float(weights.get("prosody", 0.0)) + (missing * 0.40)
        if not embedding_scores:
            missing = float(weights.get("embedding", 0.0))
            weights["embedding"] = 0.0
            weights["mel_cnn"] = float(weights.get("mel_cnn", 0.0)) + (missing * 0.60)
            weights["prosody"] = float(weights.get("prosody", 0.0)) + (missing * 0.40)

    available_weight = sum(float(weights.get(name, 0.0)) for name, scores in branches.items() if scores)
    if available_weight <= 1e-12:
        return {}

    fused = {label: 0.0 for label in EMOTIONS}
    for branch_name, scores in branches.items():
        if not scores:
            continue
        weight = float(weights.get(branch_name, 0.0)) / available_weight
        for label in EMOTIONS:
            fused[label] += weight * float(scores.get(label, 0.0))
    text_top = branch_top(text_scores)
    strong_audio_top = branch_top(mel_cnn_scores) if mel_cnn_scores else branch_top(embedding_scores)
    if text_top in EMOTIONS and strong_audio_top in EMOTIONS:
        if text_top == strong_audio_top:
            fused[text_top] *= 1.05
        elif strong_audio_top in COMPATIBLE.get(text_top, set()):
            fused[strong_audio_top] *= 1.03
        else:
            fused[text_top] *= 0.70
    # Disgust should not be prosody-only.
    if (
        float(text_scores.get("Disgust", 0.0)) <= 0.05
        and float(mel_cnn_scores.get("Disgust", 0.0)) <= 0.05
        and float(embedding_scores.get("Disgust", 0.0)) <= 0.05
    ):
        fused["Disgust"] *= 0.25
    return mec.normalize_scores(fused)


def decide(
    final_top: str,
    top_score: float,
    top_gap: float,
    agreement: int,
    quality_status: str,
    clip_type: str,
    duration: float,
    audio_top: str,
    prosody_top: str,
) -> Tuple[str, str, str]:
    if quality_status != "pass" or clip_type == "bad_audio":
        return "Rejected", "reject", "rejected_audio_quality"
    if final_top not in EMOTIONS:
        return "Rejected", "reject", "rejected_unknown_label"
    decision_cfg = config_section("decision")
    needs_review_min = float(config_section("decision", "needs_review").get("top_score", 0.55))
    default_auto = config_section("decision", "auto_accept")
    default_top = float(default_auto.get("top_score", 0.80))
    default_gap = float(default_auto.get("top_gap", 0.25))
    default_agree = int(default_auto.get("model_agreement_count", 2))
    strict = config_section("decision", "strict_classes")
    if top_score < needs_review_min:
        return "Rejected", "reject", "rejected_low_confidence"

    if final_top == "Disgust":
        disgust_cfg = strict.get("Disgust", {}) if isinstance(strict, dict) else {}
        if (
            top_score >= float(disgust_cfg.get("top_score", 0.88))
            and top_gap >= float(disgust_cfg.get("top_gap", 0.35))
            and agreement >= int(disgust_cfg.get("model_agreement_count", 2))
        ):
            return final_top, "auto_accept", "auto_accept_disgust_strict"
        return "Needs_Review", "needs_review", "needs_review_disgust_strict"

    if final_top == "Pleasant_Surprise":
        surprise_cfg = strict.get("Pleasant_Surprise", {}) if isinstance(strict, dict) else {}
        normal_ok = (
            top_score >= float(surprise_cfg.get("top_score", 0.86))
            and top_gap >= float(surprise_cfg.get("top_gap", 0.30))
            and agreement >= int(surprise_cfg.get("model_agreement_count", 2))
        )
        reaction_ok = (
            clip_type == "short_vocal_reaction"
            and prosody_top in {"Pleasant_Surprise", "Fear"}
            and top_score >= 0.82
            and top_gap >= 0.25
            and 0.5 <= duration <= 3.0
        )
        if normal_ok:
            return final_top, "auto_accept", "auto_accept_surprise_dialogue"
        if reaction_ok:
            return final_top, "auto_accept", "auto_accept_surprise_short_reaction"
        return "Needs_Review", "needs_review", "needs_review_surprise_strict"

    if final_top == "Fear":
        normal_ok = top_score >= default_top and top_gap >= default_gap and agreement >= default_agree
        reaction_ok = (
            clip_type == "short_vocal_reaction"
            and prosody_top == "Fear"
            and audio_top in {"Fear", "Sad", "Angry", "unknown"}
            and top_score >= 0.78
        )
        if normal_ok or reaction_ok:
            return final_top, "auto_accept", "auto_accept_fear"
        return "Needs_Review", "needs_review", "needs_review_fear_uncertain"

    if top_score >= default_top and top_gap >= default_gap and agreement >= default_agree:
        return final_top, "auto_accept", "auto_accept_default"
    return "Needs_Review", "needs_review", "needs_review_default"


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def main() -> int:
    ensure_dirs()
    os.environ["EMOTION_SET"] = "7"
    mec.CFG.setdefault("classification", {})["emotion_set"] = "7"

    items = collect_items()
    if not items:
        print(f"No segmented wav clips found under {SEGMENT_ROOT}. Run the pipeline segmentation first.")
        return 1

    print(f"Found {len(items)} utterance/reaction clips")
    print("Fitting dataset-relative prosody scaler...")
    scaler = mec.fit_dataset_scaler([{"file_path": str(item["file_path"])} for item in items])

    rows: List[Dict[str, object]] = []
    sentence_rows: List[Dict[str, object]] = []
    quality_rows: List[Dict[str, object]] = []
    debug_rows: List[Dict[str, object]] = []
    debug_json_rows: List[Dict[str, object]] = []
    counts = {folder: 0 for folder in OUTPUT_FOLDERS}
    branch_totals = {
        branch: {label: 0.0 for label in EMOTIONS}
        for branch in ("mel_cnn", "embedding", "audio_baseline", "prosody", "text", "final")
    }

    for index, item in enumerate(items, start=1):
        src = Path(item["file_path"])
        manifest = item.get("manifest") or {}
        transcript = compact(manifest.get("transcript", ""))
        avg_logprob = float_or_none(manifest.get("avg_logprob"))
        no_speech_prob = float_or_none(manifest.get("no_speech_prob"))
        rel = mec.transcript_reliability(avg_logprob, no_speech_prob)
        feats = mec.extract_features(str(src))
        duration = float(feats.get("duration_sec", 0.0))
        clip_type = detect_clip_type(transcript, duration, rel, str(item.get("source_group", "")))
        quality_status, reject_reason = quality_gate(feats, no_speech_prob, clip_type)

        clip_id = f"clip_{index:05d}"
        sentence_path = safe_copy_path(SENTENCE_CLIP_DIR, f"{clip_id}_{src.name}")
        if not sentence_path.exists():
            shutil.copy2(src, sentence_path)
        sentence_rows.append(
            {
                "clip_id": clip_id,
                "source_file": str(src),
                "cut_audio_file": str(sentence_path),
                "start_sec": manifest.get("start_sec", ""),
                "end_sec": manifest.get("end_sec", ""),
                "duration_sec": duration,
                "transcript": transcript,
                "language": manifest.get("lang", ""),
                "avg_logprob": "" if avg_logprob is None else avg_logprob,
                "no_speech_prob": "" if no_speech_prob is None else no_speech_prob,
                "transcript_reliability": rel,
            }
        )

        if quality_status != "pass":
            rejected_copy = safe_copy_path(REJECTED_SEGMENT_DIR, f"{clip_id}_{src.name}")
            if not rejected_copy.exists():
                shutil.copy2(src, rejected_copy)

        text_scores = {} if quality_status != "pass" else text_candidate_scores(transcript, clip_type, rel)
        if clip_type == "short_vocal_reaction" or rel < 0.45:
            text_scores = {}
        mel_scores = {} if quality_status != "pass" else mel_cnn_emotion_scores(str(sentence_path))
        embedding_scores = {} if quality_status != "pass" else embedding_emotion_scores(str(sentence_path))
        audio_scores = {} if quality_status != "pass" else audio_candidate_scores(sentence_path)
        prosody = {} if quality_status != "pass" else prosody_scores(feats, scaler, transcript, clip_type)
        final_scores = (
            {}
            if quality_status != "pass"
            else adaptive_fusion(
                mel_cnn_scores=mel_scores,
                embedding_scores=embedding_scores,
                audio_baseline_scores=audio_scores,
                text_scores=text_scores,
                prosody=prosody,
                clip_type=clip_type,
                rel=rel,
            )
        )
        (final_top, top_score), (second_label, second_score) = mec.top2(final_scores)
        top_gap = float(top_score - second_score)
        text_top = branch_top(text_scores)
        mel_top = branch_top(mel_scores)
        embedding_top = branch_top(embedding_scores)
        audio_top = branch_top(audio_scores)
        prosody_top = branch_top(prosody)
        agreement = sum(1 for label in (mel_top, embedding_top, audio_top, prosody_top, text_top) if label == final_top)
        output_folder, confidence_status, label_source = decide(
            final_top=final_top,
            top_score=float(top_score),
            top_gap=top_gap,
            agreement=agreement,
            quality_status=quality_status,
            clip_type=clip_type,
            duration=duration,
            audio_top=audio_top,
            prosody_top=prosody_top,
        )

        counts[output_folder] += 1
        output_file = safe_copy_path(OUTPUT_ROOT / output_folder, f"{clip_id}_{src.name}")
        shutil.copy2(sentence_path, output_file)

        start_sec = manifest.get("start_sec", "")
        end_sec = manifest.get("end_sec", "")
        row = {
            "clip_id": clip_id,
            "source_file": str(src),
            "cut_audio_file": str(sentence_path),
            "output_file": str(output_file),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": duration,
            "transcript": transcript,
            "language": manifest.get("lang", ""),
            "avg_logprob": "" if avg_logprob is None else avg_logprob,
            "no_speech_prob": "" if no_speech_prob is None else no_speech_prob,
            "transcript_reliability": rel,
            "clip_type": clip_type,
            "audio_quality_status": quality_status,
            "reject_reason": reject_reason,
            "text_top": text_top,
            "mel_cnn_top": mel_top,
            "embedding_top": embedding_top,
            "audio_top": audio_top,
            "prosody_top": prosody_top,
            "final_top": final_top,
            "top_score": top_score,
            "second_label": second_label,
            "second_score": second_score,
            "top_gap": top_gap,
            "model_agreement_count": agreement,
            "confidence_status": confidence_status,
            "label_source": label_source,
        }
        for label in EMOTIONS:
            row[f"final_score_{label.lower()}"] = float(final_scores.get(label, 0.0)) if final_scores else 0.0
        rows.append(row)

        quality_rows.append(
            {
                "clip_id": clip_id,
                "source_file": str(src),
                "cut_audio_file": str(sentence_path),
                "duration_sec": duration,
                "transcript": transcript,
                "clip_type": clip_type,
                "voiced_ratio": feats.get("voiced_ratio", ""),
                "harmonic_ratio": feats.get("harmonic_ratio", ""),
                "flatness_mean": feats.get("flatness_mean", ""),
                "zcr_mean": feats.get("zcr_mean", ""),
                "rms_mean": feats.get("rms_mean", ""),
                "burst_ratio": feats.get("burst_ratio", ""),
                "no_speech_prob": "" if no_speech_prob is None else no_speech_prob,
                "transcript_reliability": rel,
                "quality_status": quality_status,
                "reject_reason": reject_reason,
            }
        )

        debug_rows.append(
            {
                "clip_id": clip_id,
                "transcript": transcript,
                "clip_type": clip_type,
                "text_top": text_top,
                "mel_cnn_top": mel_top,
                "embedding_top": embedding_top,
                "audio_top": audio_top,
                "prosody_top": prosody_top,
                "final_top": final_top,
                "top_score": top_score,
                "top_gap": top_gap,
                "model_agreement_count": agreement,
                "transcript_reliability": rel,
                "confidence_status": confidence_status,
            }
        )
        for branch_name, scores in (
            ("mel_cnn", mel_scores),
            ("embedding", embedding_scores),
            ("audio_baseline", audio_scores),
            ("prosody", prosody),
            ("text", text_scores),
            ("final", final_scores),
        ):
            for label in EMOTIONS:
                branch_totals[branch_name][label] += float(scores.get(label, 0.0)) if scores else 0.0
        debug_json_rows.append(
            {
                "clip_id": clip_id,
                "transcript": transcript,
                "clip_type": clip_type,
                "mel_cnn_scores": mel_scores,
                "embedding_scores": embedding_scores,
                "audio_baseline_scores": audio_scores,
                "prosody_scores": prosody,
                "text_scores": text_scores,
                "final_scores": final_scores,
                "top_label": final_top,
                "confidence_status": confidence_status,
            }
        )

        print(f"{clip_id} {src.name} -> {output_folder} ({label_source})")

    manifest_fields = [
        "clip_id", "source_file", "cut_audio_file", "output_file", "start_sec", "end_sec",
        "duration_sec", "transcript", "transcript_reliability", "clip_type",
        "audio_quality_status", "text_top", "audio_top", "prosody_top", "final_top",
        "top_score", "second_label", "second_score", "top_gap", "model_agreement_count",
        "confidence_status", "label_source",
        "final_score_angry", "final_score_disgust", "final_score_fear", "final_score_happy",
        "final_score_neutral", "final_score_pleasant_surprise", "final_score_sad",
    ]
    quality_fields = [
        "clip_id", "source_file", "cut_audio_file", "duration_sec", "transcript", "clip_type",
        "voiced_ratio", "harmonic_ratio", "flatness_mean", "zcr_mean", "rms_mean",
        "burst_ratio", "no_speech_prob", "transcript_reliability", "quality_status",
        "reject_reason",
    ]
    debug_fields = [
        "clip_id", "transcript", "clip_type", "mel_cnn_top", "embedding_top",
        "audio_top", "prosody_top", "text_top", "final_top", "top_score", "top_gap", "model_agreement_count",
        "transcript_reliability", "confidence_status",
    ]
    sentence_fields = [
        "clip_id", "source_file", "cut_audio_file", "start_sec", "end_sec",
        "duration_sec", "transcript", "language", "avg_logprob", "no_speech_prob",
        "transcript_reliability",
    ]
    write_csv(OUTPUT_ROOT / "_text_guided_emotion_manifest.csv", manifest_fields, rows)
    write_csv(SENTENCE_CLIP_DIR / "_sentence_clip_manifest.csv", sentence_fields, sentence_rows)
    write_csv(QUALITY_MANIFEST, quality_fields, quality_rows)
    write_csv(OUTPUT_ROOT / "_debug_text_audio_agreement.csv", debug_fields, debug_rows)
    branch_rows = []
    for branch_name in ("mel_cnn", "embedding", "audio_baseline", "prosody", "text", "final"):
        row = {"branch": branch_name}
        row.update(branch_totals[branch_name])
        branch_rows.append(row)
    write_csv(OUTPUT_ROOT / "_branch_prediction_report.csv", ["branch"] + EMOTIONS, branch_rows)
    with (OUTPUT_ROOT / "_debug_scores.jsonl").open("w", encoding="utf-8") as f:
        for row in debug_json_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    distribution_rows = []
    for emotion in OUTPUT_FOLDERS:
        if emotion in EMOTIONS:
            emotion_rows = [row for row in rows if row["final_top"] == emotion]
        elif emotion == "Needs_Review":
            emotion_rows = [row for row in rows if row["confidence_status"] == "needs_review"]
        else:
            emotion_rows = [row for row in rows if row["confidence_status"] == "reject"]
        distribution_rows.append(
            {
                "emotion": emotion,
                "auto_accept_count": sum(1 for row in emotion_rows if row["confidence_status"] == "auto_accept"),
                "needs_review_count": sum(1 for row in emotion_rows if row["confidence_status"] == "needs_review"),
                "rejected_count": sum(1 for row in emotion_rows if row["confidence_status"] == "reject"),
                "total_count": len(emotion_rows),
            }
        )
    write_csv(
        OUTPUT_ROOT / "_emotion_distribution_report.csv",
        ["emotion", "auto_accept_count", "needs_review_count", "rejected_count", "total_count"],
        distribution_rows,
    )
    auto_rows = [row for row in rows if row["confidence_status"] == "auto_accept"]
    if auto_rows:
        auto_counts: Dict[str, int] = {}
        for row in auto_rows:
            label = str(row.get("final_top") or "unknown")
            auto_counts[label] = auto_counts.get(label, 0) + 1
        label, count = max(auto_counts.items(), key=lambda item: item[1])
        if count / max(1, len(auto_rows)) > 0.70:
            print(f"Warning: label collapse detected. Most clips are assigned to {label}.", flush=True)

    print("\nDone.")
    for folder in OUTPUT_FOLDERS:
        print(f"{folder}: {counts[folder]}")
    print(f"Manifest: {OUTPUT_ROOT / '_text_guided_emotion_manifest.csv'}")
    print(f"Sentence clip manifest: {SENTENCE_CLIP_DIR / '_sentence_clip_manifest.csv'}")
    print(f"Quality manifest: {QUALITY_MANIFEST}")
    print(f"Debug agreement: {OUTPUT_ROOT / '_debug_text_audio_agreement.csv'}")
    print(f"Branch report: {OUTPUT_ROOT / '_branch_prediction_report.csv'}")
    print(f"Debug scores: {OUTPUT_ROOT / '_debug_scores.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
