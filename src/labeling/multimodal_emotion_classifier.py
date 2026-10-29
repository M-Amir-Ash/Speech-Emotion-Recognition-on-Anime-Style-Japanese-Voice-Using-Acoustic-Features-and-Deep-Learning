import os
import csv
import json
import math
import subprocess
import shutil
import sys
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def configure_windows_cuda_dlls() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    nvidia_base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for relative in (
        ("cublas", "bin"),
        ("cuda_runtime", "bin"),
        ("cuda_nvrtc", "bin"),
        ("cudnn", "bin"),
    ):
        dll_dir = nvidia_base.joinpath(*relative)
        if dll_dir.is_dir():
            os.add_dll_directory(str(dll_dir))
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")


configure_windows_cuda_dlls()

import numpy as np
import pandas as pd
import librosa


def patch_torchaudio_legacy_backend_api() -> None:
    """SpeechBrain versions before the dispatcher rewrite expect old torchaudio APIs."""
    try:
        import torchaudio
    except Exception:
        return

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend=None: None  # type: ignore[attr-defined]

TEXT_MODEL_LABEL_FALLBACK = [
    "Joy",
    "Sadness",
    "Anticipation",
    "Surprise",
    "Anger",
    "Fear",
    "Disgust",
    "Trust",
]

JAPANESE_ANGER_CUES = (
    "じゃねぇ",
    "じゃねー",
    "ふざけ",
    "ふざんな",
    "ざけんな",
    "許せ",
    "嫌い",
    "うるせ",
    "うるさい",
    "なんだよ",
    "やめろ",
    "違う",
    "間違",
    "バカ",
    "くそ",
    "ムカつ",
    "腹立",
    "怒",
)
JAPANESE_DISGUST_CUES = ("気持ち悪", "最悪", "嫌い", "うざ", "汚", "吐き気")
JAPANESE_FEAR_CUES = ("怖", "恐", "不安", "やば", "死にそう", "逃げ", "助け", "無理", "震")
JAPANESE_SAD_CUES = ("悲", "寂", "つら", "辛", "泣", "苦し", "しんど", "失っ", "ひとり", "死にたい")
JAPANESE_HAPPY_CUES = ("嬉", "楽しい", "よかった", "最高", "好き", "幸せ")
JAPANESE_SURPRISE_CUES = ("えっ", "え?", "え？", "まさか", "すごい", "びっくり", "驚", "やった")


# =========================================================
# CONFIG LOADER
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "emotion_classifier_config.json")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


CFG = load_json(CONFIG_PATH)

PATHS_CFG = CFG["paths"]
RAW_INPUT_BASES = PATHS_CFG.get("input_bases")
if RAW_INPUT_BASES:
    INPUT_BASES = [os.path.join(BASE_DIR, p) for p in RAW_INPUT_BASES]
else:
    INPUT_BASES = [os.path.join(BASE_DIR, PATHS_CFG["input_base"])]

OUTPUT_BASE = os.path.join(BASE_DIR, CFG["paths"]["output_base"])
TARGET_SR = int(CFG["audio"]["target_sr"])
COPY_FILES = bool(CFG["output"]["copy_files"])
PRESERVE_ORIGINAL_FILENAME = bool(CFG["output"].get("preserve_original_filename", True))

CLASSIFY_SOURCE_FOLDERS = set(PATHS_CFG.get("classify_folders", ["japanese_dialogue", "dialogue"]))
REJECT_SOURCE_FOLDERS = set(PATHS_CFG.get("reject_folders", ["character_vocal_effect", "other_or_effect"]))

PRIMARY_5_LABELS = [
    "Angry",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
]

EXTENDED_LABELS = [
    "Disgust",
    "Pleasant_Surprise",
]

CANONICAL_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Pleasant_Surprise",
    "Sad",
    "Needs_Review",
]

EXTRA_FOLDERS = [
    "Rejected_LowQuality",
]

ALL_FOLDERS = CANONICAL_LABELS + EXTRA_FOLDERS
REVIEW_AUTO_LABEL = "high_confidence"
REVIEW_MANUAL_LABEL = "manual_review"
REVIEW_REJECTED_LABEL = "rejected"

# =========================================================
# GLOBAL MODELS
# =========================================================
WHISPER_MODEL = None
WHISPER_MODEL_DISABLED = False
WHISPER_MODEL_ERROR = None
WHISPER_RUNTIME = {}
TEXT_MODEL = None
TEXT_MODEL_DISABLED = False
TEXT_MODEL_ERROR = None
AUDIO_MODEL = None
AUDIO_MODEL_DISABLED = False
AUDIO_MODEL_ERROR = None
TEXT_RUNTIME = {}
AUDIO_RUNTIME = {}
OPENAI_JUDGE_CALLS = 0
OPENAI_JUDGE_ERRORS = 0
OPENAI_JUDGE_DISABLED = False


# =========================================================
# HELPERS
# =========================================================
def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in name)


def ensure_dirs():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    for folder in active_output_folders():
        os.makedirs(os.path.join(OUTPUT_BASE, folder), exist_ok=True)


def emotion_set_name() -> str:
    env_value = os.environ.get("EMOTION_SET", "").strip()
    if env_value:
        return "7" if env_value == "7" else "5"
    cfg_value = str(CFG.get("classification", {}).get("emotion_set", "5")).strip()
    return "7" if cfg_value == "7" else "5"


def auto_label_policy() -> str:
    env_value = os.environ.get("EMOTION_AUTO_LABEL_POLICY", "").strip().lower()
    cfg_value = str(CFG.get("classification", {}).get("auto_label_policy", "strict")).strip().lower()
    policy = env_value or cfg_value
    if policy not in {"strict", "balanced", "volume"}:
        return "strict"
    return policy


def active_emotion_labels() -> List[str]:
    if emotion_set_name() == "7":
        return [label for label in CANONICAL_LABELS if label != "Needs_Review"]
    return list(PRIMARY_5_LABELS)


def active_output_folders() -> List[str]:
    return active_emotion_labels() + ["Needs_Review"] + EXTRA_FOLDERS


def canonical_to_active_label(label: str) -> Optional[str]:
    label = str(label or "").strip()
    active = set(active_emotion_labels())
    if label in active:
        return label
    if emotion_set_name() == "5":
        if label == "Disgust":
            return "Angry"
        if label == "Pleasant_Surprise":
            return "Happy"
    return None


def empty_active_scores() -> Dict[str, float]:
    return {lab: 0.0 for lab in active_emotion_labels()}


def normalize_to_active_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    mapped = empty_active_scores()
    for label, value in (scores or {}).items():
        dst = canonical_to_active_label(label)
        if dst in mapped:
            mapped[dst] += float(value)
    if sum(max(float(v), 0.0) for v in mapped.values()) <= 1e-12:
        return {}
    return normalize_scores(mapped)


def add_score(scores: Dict[str, float], label: str, value: float) -> None:
    dst = canonical_to_active_label(label)
    if dst in scores:
        scores[dst] += float(value)


def compact_text_for_csv(value: Optional[str]) -> str:
    if not value:
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").replace("\t", " ").split())


def build_feature_summary(feats: Dict[str, float]) -> Dict[str, float]:
    keys = (
        "duration_sec",
        "voiced_ratio",
        "harmonic_ratio",
        "flatness_mean",
        "zcr_mean",
        "rms_mean",
        "burst_ratio",
    )
    return {key: float(feats.get(key, 0.0)) for key in keys}


def build_score_columns(scores: Dict[str, float], prefix: str) -> Dict[str, float]:
    out = {}
    for label in CANONICAL_LABELS:
        if label == "Needs_Review":
            continue
        out[f"{prefix}_{label.lower()}"] = float(scores.get(label, 0.0))
    return out


def resolve_label_names(id2label: Dict[int, str], fallback_order: List[str]) -> Dict[int, str]:
    resolved = {}
    generic = True
    for idx, label in sorted(id2label.items()):
        label_str = str(label)
        if not label_str.upper().startswith("LABEL_"):
            generic = False
        resolved[int(idx)] = label_str

    if generic and len(resolved) == len(fallback_order):
        return {idx: fallback_order[idx] for idx in sorted(resolved.keys())}

    return resolved


def count_cues(text: str, cues: Tuple[str, ...]) -> int:
    text = str(text or "")
    return sum(text.count(cue) for cue in cues)


def text_emotion_cues(text: str) -> Dict[str, int]:
    return {
        "Angry": count_cues(text, JAPANESE_ANGER_CUES),
        "Disgust": count_cues(text, JAPANESE_DISGUST_CUES),
        "Fear": count_cues(text, JAPANESE_FEAR_CUES),
        "Happy": count_cues(text, JAPANESE_HAPPY_CUES),
        "Pleasant_Surprise": count_cues(text, JAPANESE_SURPRISE_CUES),
        "Sad": count_cues(text, JAPANESE_SAD_CUES),
        "exclaim": text.count("!") + text.count("！"),
    }


def calibrate_text_scores(text: str, mapped: Dict[str, float]) -> Dict[str, float]:
    if not mapped:
        return {}

    cues = text_emotion_cues(text)
    calibrated = dict(mapped)

    angry_signal = float(cues["Angry"] + cues["Disgust"] + min(cues["exclaim"], 2))
    sad_signal = float(cues["Sad"])
    fear_signal = float(cues["Fear"])
    happy_signal = float(cues["Happy"])
    surprise_signal = float(cues["Pleasant_Surprise"])

    if angry_signal > 0.0:
        add_score(calibrated, "Angry", 0.16 * min(3.0, angry_signal))
        add_score(calibrated, "Disgust", 0.08 * min(2.0, float(cues["Disgust"])))
        surprise_label = canonical_to_active_label("Pleasant_Surprise")
        if surprise_label in calibrated:
            calibrated[surprise_label] *= 0.45
        if "Sad" in calibrated:
            calibrated["Sad"] *= 0.72

    if sad_signal > 0.0:
        add_score(calibrated, "Sad", 0.15 * min(3.0, sad_signal))
        surprise_label = canonical_to_active_label("Pleasant_Surprise")
        if surprise_label in calibrated:
            calibrated[surprise_label] *= 0.70

    if fear_signal > 0.0:
        add_score(calibrated, "Fear", 0.14 * min(3.0, fear_signal))

    if happy_signal > 0.0:
        add_score(calibrated, "Happy", 0.12 * min(3.0, happy_signal))

    if surprise_signal > 0.0 and angry_signal == 0.0 and sad_signal == 0.0:
        add_score(calibrated, "Pleasant_Surprise", 0.10 * min(3.0, surprise_signal))

    return normalize_scores(calibrated)


def review_decision(
    pred_folder: str,
    top_score: float,
    gap: float,
    transcript_rel: Optional[float],
) -> Tuple[str, str]:
    if pred_folder == "Rejected_LowQuality":
        return REVIEW_REJECTED_LABEL, "rejected_by_quality_gate"
    if pred_folder == "Needs_Review":
        return REVIEW_MANUAL_LABEL, "uncertain_fusion_threshold"

    review_cfg = CFG.get("review", {})
    min_conf = float(review_cfg.get("high_confidence_min_confidence", 0.80))
    min_gap = float(review_cfg.get("high_confidence_min_gap", 0.25))
    min_rel = float(review_cfg.get("min_transcript_reliability", 0.45))
    tr_rel = 0.5 if transcript_rel is None else float(transcript_rel)

    if top_score >= min_conf and gap >= min_gap and tr_rel >= min_rel:
        return REVIEW_AUTO_LABEL, "high_confidence_auto_accept"

    reasons = []
    if top_score < min_conf:
        reasons.append("low_top_score")
    if gap < min_gap:
        reasons.append("low_top_gap")
    if tr_rel < min_rel:
        reasons.append("low_transcript_reliability")
    return REVIEW_MANUAL_LABEL, "|".join(reasons) if reasons else "manual_review"


def review_priority(row: Dict[str, object]) -> float:
    top_score = float(row.get("top_score", 0.0) or 0.0)
    top_gap = float(row.get("top_gap", 0.0) or 0.0)
    tr_rel = float(row.get("transcript_reliability", 0.5) or 0.5)
    return float((1.0 - top_score) + (0.6 * (1.0 - top_gap)) + (0.4 * (1.0 - tr_rel)))


def env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def extract_response_text(response: Dict[str, object]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: List[str] = []
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
    return "\n".join(parts).strip()


def forced_label_scores(label: str, confidence: float, old_scores: Dict[str, float]) -> Dict[str, float]:
    label = canonical_to_active_label(label) or label
    labels = active_emotion_labels()
    top_score = min(0.92, max(0.80, float(confidence)))
    remaining = max(0.0, 1.0 - top_score)
    old_rest_total = sum(float(old_scores.get(lab, 0.0)) for lab in labels if lab != label)
    out: Dict[str, float] = {}
    for lab in labels:
        if lab == label:
            out[lab] = top_score
        elif old_rest_total > 0:
            out[lab] = remaining * float(old_scores.get(lab, 0.0)) / old_rest_total
        else:
            out[lab] = remaining / max(1, len(labels) - 1)
    return out


def openai_uncertain_emotion_judge(
    transcript: str,
    final_scores: Dict[str, float],
    audio_scores: Dict[str, float],
    text_scores: Dict[str, float],
    feats: Dict[str, float],
    reason: str,
) -> Dict[str, object]:
    global OPENAI_JUDGE_CALLS, OPENAI_JUDGE_ERRORS, OPENAI_JUDGE_DISABLED

    if OPENAI_JUDGE_DISABLED or not env_truthy("OPENAI_EMOTION_JUDGE"):
        return {}
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        OPENAI_JUDGE_DISABLED = True
        print("OpenAI emotion judge disabled: OPENAI_API_KEY is missing.", flush=True)
        return {}

    limit = int(os.environ.get("OPENAI_EMOTION_JUDGE_LIMIT", "0") or "0")
    if limit <= 0 or OPENAI_JUDGE_CALLS >= limit:
        return {}

    model = os.environ.get("OPENAI_EMOTION_MODEL", "gpt-4.1").strip() or "gpt-4.1"
    timeout_sec = int(os.environ.get("OPENAI_EMOTION_TIMEOUT_SEC", "120") or "120")
    threshold = float(os.environ.get("OPENAI_EMOTION_MIN_CONF", "0.70") or "0.70")
    top_ranked = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)[:4]
    feature_summary = build_feature_summary(feats)
    prompt = {
        "task": f"Pick the best {emotion_set_name()}-emotion label for a Japanese anime dialogue clip.",
        "allowed_labels": active_emotion_labels() + ["Unknown"],
        "rule": (
            "Use transcript, audio model scores, text model scores, fused scores, and prosody. "
            "Only choose an emotion if the evidence is clear. "
            "If it is ambiguous, mixed, singing, sound effect, or too weak, choose Unknown. "
            "Return JSON only."
        ),
        "uncertainty_reason": reason,
        "transcript": compact_text_for_csv(transcript)[:1200],
        "fused_top_scores": {k: round(float(v), 6) for k, v in top_ranked},
        "audio_scores": {k: round(float(v), 6) for k, v in audio_scores.items()},
        "text_scores": {k: round(float(v), 6) for k, v in text_scores.items()},
        "prosody_features": feature_summary,
        "output_schema": {"emotion": "label or Unknown", "confidence": "0.0-1.0", "reason": "short"},
    }
    body = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": "You are a strict speech-emotion dataset judge. Output compact JSON only.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "text": {"format": {"type": "json_object"}},
        "max_output_tokens": 160,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        OPENAI_JUDGE_CALLS += 1
        with urllib.request.urlopen(req, timeout=max(30, timeout_sec)) as resp:
            response = json.loads(resp.read().decode("utf-8"))
        parsed = json.loads(extract_response_text(response))
        emotion = str(parsed.get("emotion", "")).strip()
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        emotion = canonical_to_active_label(emotion) or emotion
        if emotion not in active_emotion_labels():
            emotion = "Unknown"
        OPENAI_JUDGE_ERRORS = 0
        return {
            "emotion": emotion,
            "confidence": confidence,
            "reason": str(parsed.get("reason", "")).strip()[:240],
            "model": model,
            "accepted": emotion != "Unknown" and confidence >= threshold,
        }
    except (TimeoutError, socket.timeout):
        OPENAI_JUDGE_ERRORS += 1
        print(f"OpenAI emotion judge timed out ({OPENAI_JUDGE_ERRORS}/3); using local result.", flush=True)
    except urllib.error.HTTPError as e:
        OPENAI_JUDGE_ERRORS += 1
        detail = e.read().decode("utf-8", errors="replace")
        print(f"OpenAI emotion judge error ({OPENAI_JUDGE_ERRORS}/3): HTTP {e.code}: {detail[:300]}", flush=True)
    except Exception as e:
        OPENAI_JUDGE_ERRORS += 1
        print(f"OpenAI emotion judge error ({OPENAI_JUDGE_ERRORS}/3): {type(e).__name__}: {e}", flush=True)

    if OPENAI_JUDGE_ERRORS >= 3:
        OPENAI_JUDGE_DISABLED = True
        print("OpenAI emotion judge disabled after repeated errors.", flush=True)
    return {}


def write_review_exports(output_base: str, rows: List[Dict[str, object]]) -> Dict[str, object]:
    high_rows = [row for row in rows if row.get("review_status") == REVIEW_AUTO_LABEL]
    manual_rows = [row for row in rows if row.get("review_status") == REVIEW_MANUAL_LABEL]
    needs_review_rows = [row for row in rows if row.get("pred_folder") == "Needs_Review"]
    rejected_rows = [row for row in rows if row.get("pred_folder") == "Rejected_LowQuality"]
    emotion_rows = [
        row
        for row in rows
        if row.get("pred_folder") not in {"Needs_Review", "Rejected_LowQuality"}
    ]
    manual_emotion_rows = [
        row
        for row in manual_rows
        if row.get("pred_folder") not in {"Needs_Review", "Rejected_LowQuality"}
    ]

    high_rows = sorted(high_rows, key=lambda row: (-float(row.get("top_score", 0.0) or 0.0), row["source_file"]))
    manual_rows = sorted(manual_rows, key=review_priority, reverse=True)

    review_fieldnames = [
        "source_file",
        "source_group",
        "speaker",
        "character",
        "character_score",
        "character_margin",
        "pred_folder",
        "review_status",
        "review_reason",
        "gpt_judge_label",
        "gpt_judge_confidence",
        "gpt_judge_reason",
        "gpt_judge_model",
        "top_label",
        "top_score",
        "second_label",
        "second_score",
        "top_gap",
        "transcript_reliability",
        "transcript",
        "avg_logprob",
        "no_speech_prob",
        "output_file",
    ]

    high_path = os.path.join(output_base, "_high_confidence_manifest.csv")
    manual_path = os.path.join(output_base, "_manual_review_queue.csv")

    with open(high_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=review_fieldnames)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in review_fieldnames} for row in high_rows])

    with open(manual_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=review_fieldnames)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in review_fieldnames} for row in manual_rows])

    return {
        "high_confidence_csv": high_path,
        "manual_review_csv": manual_path,
        "high_confidence_count": len(high_rows),
        "manual_review_count": len(manual_rows),
        "auto_labeled_emotion_count": len(emotion_rows),
        "needs_review_folder_count": len(needs_review_rows),
        "low_confidence_emotion_review_count": len(manual_emotion_rows),
        "rejected_low_quality_count": len(rejected_rows),
    }


def should_classify_source(source_group: str) -> bool:
    if source_group in REJECT_SOURCE_FOLDERS:
        return False
    if CLASSIFY_SOURCE_FOLDERS:
        return source_group in CLASSIFY_SOURCE_FOLDERS
    return True


def seconds_to_samples(sec: float, sr: int) -> int:
    return int(sec * sr)


def is_cuda_runtime_error(error: Exception) -> bool:
    msg = f"{type(error).__name__}: {error}".lower()
    cuda_terms = (
        "cuda",
        "cublas",
        "cudnn",
        "out of memory",
        "device-side",
        "cannot be loaded",
        "ct2",
    )
    return any(term in msg for term in cuda_terms)


def softmax_dict(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    keys = list(d.keys())
    vals = np.array([float(d[k]) for k in keys], dtype=np.float64)
    vals = vals - np.max(vals)
    ex = np.exp(vals)
    probs = ex / (np.sum(ex) + 1e-12)
    return {k: float(v) for k, v in zip(keys, probs)}


def normalize_scores(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    s = float(sum(max(v, 0.0) for v in d.values()))
    if s <= 1e-12:
        n = len(d)
        return {k: 1.0 / n for k in d.keys()}
    return {k: float(max(v, 0.0) / s) for k, v in d.items()}


def blend_with_uniform(scores: Dict[str, float], ratio: float) -> Dict[str, float]:
    if not scores:
        return {}
    ratio = max(0.0, min(1.0, float(ratio)))
    uniform = 1.0 / float(len(scores))
    return normalize_scores({k: ((1.0 - ratio) * float(v)) + (ratio * uniform) for k, v in scores.items()})


def soften_distribution(scores: Dict[str, float], cap: float = 0.62, floor: float = 0.01) -> Dict[str, float]:
    if not scores:
        return {}
    active = empty_active_scores()
    for label in active:
        active[label] = min(float(scores.get(label, 0.0)), cap) + floor
    return normalize_scores(active)


def normalize_evidence_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores or sum(max(float(v), 0.0) for v in scores.values()) <= 1e-12:
        return {}
    return normalize_to_active_scores(scores)


def top2(scores: Dict[str, float]) -> Tuple[Tuple[str, float], Tuple[str, float]]:
    if not scores:
        return ("none", 0.0), ("none", 0.0)
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    first = items[0]
    second = items[1] if len(items) > 1 else ("none", 0.0)
    return first, second


def read_manifest_transcript(manifest_path: str, file_name: str) -> Dict[str, Optional[float]]:
    if not os.path.isfile(manifest_path):
        return {"transcript": "", "avg_logprob": None, "no_speech_prob": None, "speaker": "", "character": "", "character_score": "", "character_margin": ""}

    try:
        df = pd.read_csv(manifest_path)
    except Exception:
        return {"transcript": "", "avg_logprob": None, "no_speech_prob": None, "speaker": "", "character": "", "character_score": "", "character_margin": ""}

    if "filename" not in df.columns:
        return {"transcript": "", "avg_logprob": None, "no_speech_prob": None, "speaker": "", "character": "", "character_score": "", "character_margin": ""}

    row = df[df["filename"] == file_name]
    if row.empty:
        return {"transcript": "", "avg_logprob": None, "no_speech_prob": None, "speaker": "", "character": "", "character_score": "", "character_margin": ""}

    r = row.iloc[0]
    return {
        "transcript": "" if pd.isna(r.get("transcript")) else str(r.get("transcript")).strip(),
        "avg_logprob": None if pd.isna(r.get("avg_logprob")) else float(r.get("avg_logprob")),
        "no_speech_prob": None if pd.isna(r.get("no_speech_prob")) else float(r.get("no_speech_prob")),
        "speaker": "" if pd.isna(r.get("speaker")) else str(r.get("speaker")).strip(),
        "character": "" if pd.isna(r.get("character")) else str(r.get("character")).strip(),
        "character_score": "" if pd.isna(r.get("character_score")) else str(r.get("character_score")).strip(),
        "character_margin": "" if pd.isna(r.get("character_margin")) else str(r.get("character_margin")).strip(),
    }


def collect_dialogue_segments() -> List[dict]:
    items = []

    for input_base in INPUT_BASES:
        if not os.path.isdir(input_base):
            print(f"Input folder not found: {input_base}")
            continue

        base_group = os.path.basename(os.path.normpath(input_base))
        for root, _, files in os.walk(input_base):
            manifest_path = os.path.join(root, "_segments_manifest.csv")
            rel_root = os.path.relpath(root, input_base)
            if rel_root == ".":
                source_group = base_group
            else:
                source_group = rel_root.split(os.sep, 1)[0]
            for file_name in sorted(files):
                if not file_name.lower().endswith(".wav"):
                    continue
                items.append({
                    "file_path": os.path.join(root, file_name),
                    "file_name": file_name,
                    "segment_dir": root,
                    "manifest_path": manifest_path,
                    "source_group": source_group,
                })
    return items


# =========================================================
# MODEL LOADERS
# =========================================================
def load_whisper(force_cpu: bool = False):
    global WHISPER_MODEL, WHISPER_MODEL_DISABLED, WHISPER_MODEL_ERROR, WHISPER_RUNTIME
    if WHISPER_MODEL_DISABLED:
        return None

    if WHISPER_MODEL is None:
        from faster_whisper import WhisperModel

        asr_cfg = CFG["models"]["asr"]
        requested_device = str(asr_cfg.get("device", "cpu")).strip().lower()
        requested_compute = str(asr_cfg.get("compute_type", "int8")).strip().lower()

        attempts = []
        if force_cpu:
            attempts.extend([("cpu", "int8"), ("cpu", "float32")])
        else:
            attempts.append((requested_device, requested_compute))
            if requested_device == "cuda":
                attempts.extend([("cpu", "int8"), ("cpu", "float32")])
            elif requested_device == "cpu" and requested_compute not in {"int8", "float32"}:
                attempts.append(("cpu", "int8"))

        last_error = None
        for device, compute_type in attempts:
            try:
                WHISPER_MODEL = WhisperModel(
                    asr_cfg["name"],
                    device=device,
                    compute_type=compute_type,
                )
                WHISPER_RUNTIME = {
                    "device": device,
                    "compute_type": compute_type,
                }
                if device != requested_device or compute_type != requested_compute:
                    print(
                        "Warning: ASR model is running with a fallback configuration "
                        f"({device}/{compute_type}) instead of {requested_device}/{requested_compute}."
                    )
                break
            except Exception as e:
                last_error = e
                WHISPER_MODEL = None
                WHISPER_RUNTIME = {}

        if WHISPER_MODEL is None:
            WHISPER_MODEL_DISABLED = True
            WHISPER_MODEL_ERROR = f"{type(last_error).__name__}: {last_error}"
            print(
                "Warning: ASR model unavailable; continuing without transcript generation. "
                f"Reason: {WHISPER_MODEL_ERROR}"
            )
            return None

    return WHISPER_MODEL


def load_text_model():
    global TEXT_MODEL, TEXT_MODEL_DISABLED, TEXT_MODEL_ERROR, TEXT_RUNTIME
    if TEXT_MODEL_DISABLED:
        return None

    if TEXT_MODEL is None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            text_cfg = CFG["models"]["text_emotion"]
            requested_device = str(text_cfg.get("device", "cuda")).strip().lower()
            use_cuda = requested_device == "cuda" and torch.cuda.is_available()
            device = "cuda" if use_cuda else "cpu"

            tokenizer = AutoTokenizer.from_pretrained(text_cfg["name"])
            model = AutoModelForSequenceClassification.from_pretrained(text_cfg["name"])
            model.to(device)
            model.eval()
            id2label = resolve_label_names(
                {int(k): str(v) for k, v in model.config.id2label.items()},
                TEXT_MODEL_LABEL_FALLBACK,
            )
            TEXT_MODEL = {
                "model_name": text_cfg["name"],
                "tokenizer": tokenizer,
                "model": model,
                "device": device,
                "torch": torch,
                "id2label": id2label,
            }
            TEXT_RUNTIME = {"device": device}
            if requested_device == "cuda" and device != "cuda":
                print("Warning: text emotion model is running on CPU fallback instead of cuda.")
        except Exception as e:
            TEXT_MODEL_DISABLED = True
            TEXT_MODEL_ERROR = f"{type(e).__name__}: {e}"
            print(
                "Warning: text emotion model unavailable; continuing without text branch. "
                f"Reason: {TEXT_MODEL_ERROR}"
            )
            return None

    return TEXT_MODEL


def load_audio_model():
    global AUDIO_MODEL, AUDIO_MODEL_DISABLED, AUDIO_MODEL_ERROR, AUDIO_RUNTIME
    if AUDIO_MODEL_DISABLED:
        return None

    if AUDIO_MODEL is None:
        audio_cfg = CFG["models"]["audio_emotion"]
        requested_device = str(audio_cfg.get("device", "cpu")).strip().lower()
        last_error = None
        torch = None
        AutoFeatureExtractor = None
        AutoModelForAudioClassification = None

        try:
            import torch as _torch
            from transformers import AutoFeatureExtractor as _AutoFeatureExtractor
            from transformers import AutoModelForAudioClassification as _AutoModelForAudioClassification

            torch = _torch
            AutoFeatureExtractor = _AutoFeatureExtractor
            AutoModelForAudioClassification = _AutoModelForAudioClassification
        except Exception:
            pass

        try:
            patch_torchaudio_legacy_backend_api()
            from speechbrain.inference.classifiers import AudioClassifier
            attempts = [requested_device]
            if requested_device == "cuda":
                attempts.append("cpu")
            for device in attempts:
                try:
                    # Windows-safe / SpeechBrain-safe loading
                    kwargs = {
                        "source": audio_cfg["name"],
                        "savedir": os.path.join(BASE_DIR, ".cache", "speechbrain_emotion"),
                        "run_opts": {"device": device},
                    }

                    # Try local_strategy if SpeechBrain version supports it
                    try:
                        from speechbrain.utils.fetching import LocalStrategy
                        kwargs["local_strategy"] = LocalStrategy.COPY_SKIP_CACHE
                    except Exception:
                        pass

                    AUDIO_MODEL = AudioClassifier.from_hparams(**kwargs)
                    AUDIO_RUNTIME = {
                        "backend": "speechbrain",
                        "device": device,
                        "model_name": audio_cfg["name"],
                        "is_fallback": False,
                    }
                    if device != requested_device:
                        print(
                            "Warning: audio emotion model is running on CPU fallback "
                            f"instead of {requested_device}."
                        )
                    break
                except Exception as e:
                    last_error = e
                    AUDIO_MODEL = None
                    if device == requested_device and not is_cuda_runtime_error(e):
                        break

            if AUDIO_MODEL is None and last_error is not None:
                raise last_error
        except Exception as e:
            last_error = e
            AUDIO_MODEL = None

        if AUDIO_MODEL is None:
            try:
                fallback_name = audio_cfg.get("fallback_name", "superb/wav2vec2-base-superb-er")
                if torch is None or AutoFeatureExtractor is None or AutoModelForAudioClassification is None:
                    raise RuntimeError("transformers audio fallback classes were not available")
                use_cuda = requested_device == "cuda" and torch.cuda.is_available()
                device = "cuda" if use_cuda else "cpu"

                feature_extractor = AutoFeatureExtractor.from_pretrained(fallback_name)
                model = AutoModelForAudioClassification.from_pretrained(fallback_name)
                model.to(device)
                model.eval()
                AUDIO_MODEL = {
                    "backend": "transformers_audio",
                    "model_name": fallback_name,
                    "feature_extractor": feature_extractor,
                    "model": model,
                    "device": device,
                    "torch": torch,
                    "id2label": {int(k): str(v) for k, v in model.config.id2label.items()},
                }
                AUDIO_RUNTIME = {
                    "backend": "transformers_audio",
                    "device": device,
                    "model_name": fallback_name,
                    "is_fallback": True,
                }
                print(
                    "Warning: SpeechBrain audio emotion model failed; using transformers fallback "
                    f"({fallback_name}) on {device}. Reason: {type(last_error).__name__}: {last_error}"
                )
            except Exception as fallback_error:
                AUDIO_MODEL_DISABLED = True
                AUDIO_MODEL_ERROR = (
                    f"speechbrain={type(last_error).__name__}: {last_error} | "
                    f"fallback={type(fallback_error).__name__}: {fallback_error}"
                )
                print(
                    "Warning: audio emotion model unavailable; continuing with text+prosody only. "
                    f"Reason: {AUDIO_MODEL_ERROR}"
                )
                return None

    return AUDIO_MODEL


# =========================================================
# FEATURES
# =========================================================
FEATURE_NAMES = [
    "duration_sec",
    "rms_mean",
    "rms_std",
    "rms_peak",
    "burst_ratio",
    "zcr_mean",
    "centroid_mean",
    "flatness_mean",
    "rolloff_mean",
    "f0_mean",
    "f0_std",
    "voiced_ratio",
    "harmonic_ratio",
]


def extract_features(file_path: str) -> Dict[str, float]:
    y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

    if len(y) == 0:
        raise ValueError("Empty audio")

    duration_sec = len(y) / sr

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = float(np.mean(np.abs(y_harm))) if len(y_harm) > 0 else 0.0
    perc_energy = float(np.mean(np.abs(y_perc))) if len(y_perc) > 0 else 0.0
    harmonic_ratio = harm_energy / (harm_energy + perc_energy + 1e-8)

    f0 = librosa.yin(y, fmin=50, fmax=600, sr=sr)
    finite_mask = np.isfinite(f0) & (f0 > 0)
    if np.any(finite_mask):
        f0_valid = f0[finite_mask]
        f0_mean = float(np.mean(f0_valid))
        f0_std = float(np.std(f0_valid))
        voiced_ratio = float(np.mean(finite_mask))
    else:
        f0_mean = 0.0
        f0_std = 0.0
        voiced_ratio = 0.0

    rms_mean = float(np.mean(rms)) if len(rms) > 0 else 0.0
    rms_std = float(np.std(rms)) if len(rms) > 0 else 0.0
    rms_peak = float(np.max(rms)) if len(rms) > 0 else 0.0
    burst_ratio = rms_peak / (rms_mean + 1e-8)

    return {
        "duration_sec": float(duration_sec),
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "rms_peak": rms_peak,
        "burst_ratio": float(burst_ratio),
        "zcr_mean": float(np.mean(zcr)) if len(zcr) > 0 else 0.0,
        "centroid_mean": float(np.mean(centroid)) if len(centroid) > 0 else 0.0,
        "flatness_mean": float(np.mean(flatness)) if len(flatness) > 0 else 0.0,
        "rolloff_mean": float(np.mean(rolloff)) if len(rolloff) > 0 else 0.0,
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "voiced_ratio": voiced_ratio,
        "harmonic_ratio": float(harmonic_ratio),
    }


def fit_dataset_scaler(items: List[dict]) -> Dict[str, Dict[str, float]]:
    values = {k: [] for k in FEATURE_NAMES}

    for item in items:
        try:
            feats = extract_features(item["file_path"])
            for k in FEATURE_NAMES:
                values[k].append(float(feats[k]))
        except Exception:
            continue

    scaler = {}
    for k, arr in values.items():
        if not arr:
            scaler[k] = {"mean": 0.0, "std": 1.0}
            continue
        a = np.asarray(arr, dtype=np.float64)
        m = float(np.mean(a))
        s = float(np.std(a))
        if s < 1e-8:
            s = 1.0
        scaler[k] = {"mean": m, "std": s}
    return scaler


def zscore_feature(value: float, mean: float, std: float) -> float:
    return (float(value) - float(mean)) / (float(std) + 1e-8)


# =========================================================
# TRANSCRIPT
# =========================================================
def transcribe_if_needed(file_path: str) -> Dict[str, Optional[float]]:
    global WHISPER_MODEL, WHISPER_MODEL_DISABLED, WHISPER_MODEL_ERROR, WHISPER_RUNTIME

    model = load_whisper()
    if model is None:
        return {"transcript": "", "avg_logprob": None, "no_speech_prob": None}

    audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
    audio = audio.astype(np.float32, copy=False)

    try:
        segments, _info = model.transcribe(
            audio,
            language=CFG["models"]["asr"]["language"],
            beam_size=int(CFG["models"]["asr"]["beam_size"]),
            vad_filter=True,
            condition_on_previous_text=False,
        )
    except Exception as e:
        if WHISPER_RUNTIME.get("device") == "cuda" and is_cuda_runtime_error(e):
            print(
                "Warning: ASR transcription failed on CUDA; retrying on CPU. "
                f"Reason: {type(e).__name__}: {e}"
            )
            WHISPER_MODEL = None
            WHISPER_RUNTIME = {}
            model = load_whisper(force_cpu=True)
            if model is None:
                return {"transcript": "", "avg_logprob": None, "no_speech_prob": None}

            try:
                segments, _info = model.transcribe(
                    audio,
                    language=CFG["models"]["asr"]["language"],
                    beam_size=int(CFG["models"]["asr"]["beam_size"]),
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
            except Exception as retry_error:
                WHISPER_MODEL_DISABLED = True
                WHISPER_MODEL_ERROR = f"{type(retry_error).__name__}: {retry_error}"
                print(
                    "Warning: ASR transcription unavailable after CPU retry; using empty transcript. "
                    f"Reason: {WHISPER_MODEL_ERROR}"
                )
                return {"transcript": "", "avg_logprob": None, "no_speech_prob": None}
        else:
            WHISPER_MODEL_DISABLED = True
            WHISPER_MODEL_ERROR = f"{type(e).__name__}: {e}"
            print(
                "Warning: ASR transcription unavailable; using empty transcript. "
                f"Reason: {WHISPER_MODEL_ERROR}"
            )
            return {"transcript": "", "avg_logprob": None, "no_speech_prob": None}

    texts = []
    avg_logprobs = []
    no_speech_probs = []

    for s in segments:
        t = s.text.strip()
        if t:
            texts.append(t)
        if getattr(s, "avg_logprob", None) is not None:
            avg_logprobs.append(float(s.avg_logprob))
        if getattr(s, "no_speech_prob", None) is not None:
            no_speech_probs.append(float(s.no_speech_prob))

    return {
        "transcript": " ".join(texts).strip(),
        "avg_logprob": float(np.mean(avg_logprobs)) if avg_logprobs else None,
        "no_speech_prob": float(np.mean(no_speech_probs)) if no_speech_probs else None,
    }


# =========================================================
# AUDIO BRANCH
# =========================================================
def audio_emotion_scores(file_path: str) -> Dict[str, float]:
    model = load_audio_model()
    if model is None:
        return {}

    if isinstance(model, dict) and model.get("backend") == "transformers_audio":
        feature_extractor = model["feature_extractor"]
        hf_model = model["model"]
        torch = model["torch"]
        device = model["device"]
        target_sr = int(getattr(feature_extractor, "sampling_rate", TARGET_SR) or TARGET_SR)
        waveform = load_wav_mono_float32(file_path, target_sr)
        inputs = feature_extractor(
            waveform,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = hf_model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        result = {}
        for idx, prob in enumerate(probs):
            label = str(model["id2label"].get(idx, idx)).strip().lower()
            result[label] = float(prob)
    else:
        # Try to get posterior probabilities
        result = {}

        try:
            # SpeechBrain 1.1.x split_path() only recognizes forward slashes.
            # On Windows, classify_file("<LOCAL_PATH>") can therefore become the
            # invalid drive-relative path "D:folder...". Load the waveform
            # ourselves and classify the tensor so Unicode and drive roots are
            # preserved exactly.
            import torch

            model_sample_rate = int(getattr(model.hparams, "sample_rate", TARGET_SR) or TARGET_SR)
            waveform = load_wav_mono_float32(str(Path(file_path).resolve()), model_sample_rate)
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).to(model.device)
            relative_length = torch.ones(1, device=model.device)
            with torch.inference_mode():
                if all(
                    hasattr(model.mods, module_name)
                    for module_name in ("wav2vec2", "avg_pool", "output_mlp")
                ):
                    # speechbrain/emotion-recognition-wav2vec2-IEMOCAP ships
                    # Wav2Vec2/avg_pool/output_mlp modules but no matching
                    # generic AudioClassifier interface. Run its declared
                    # module graph directly.
                    embeddings = model.mods.wav2vec2(waveform_tensor)
                    pooled = model.mods.avg_pool(embeddings, relative_length)
                    logits = model.mods.output_mlp(pooled)
                    out_prob = model.hparams.softmax(logits).squeeze(1)
                    score, index = torch.max(out_prob, dim=-1)
                    try:
                        text_lab = model.hparams.label_encoder.decode_torch(index)
                    except (KeyError, IndexError):
                        # Some SpeechBrain/Hugging Face version combinations
                        # do not hydrate label_encoder.txt. This checkpoint's
                        # declared IEMOCAP output order is stable.
                        iemocap_labels = ["neu", "ang", "hap", "sad"]
                        text_lab = [iemocap_labels[int(index.item())]]
                else:
                    out_prob, score, index, text_lab = model.classify_batch(
                        waveform_tensor, relative_length
                    )
        except Exception as e:
            raise RuntimeError(f"SpeechBrain classify_batch failed: {e}")

        # Build raw label->score dict safely
        try:
            logits_or_probs = out_prob.squeeze().detach().float().cpu()
            # SpeechBrain classifiers commonly return log-posteriors. Convert
            # any non-probability vector before fusion.
            if (
                torch.any(logits_or_probs < 0)
                or not torch.isclose(logits_or_probs.sum(), torch.tensor(1.0), atol=1e-3)
            ):
                logits_or_probs = torch.softmax(logits_or_probs, dim=-1)
            probs = logits_or_probs.numpy()

            # Try common SpeechBrain label structures
            labels = None

            if hasattr(model.hparams, "label_encoder"):
                le = model.hparams.label_encoder

                if hasattr(le, "ind2lab"):
                    ind2lab = le.ind2lab
                    if isinstance(ind2lab, dict):
                        labels = [str(ind2lab[i]) for i in sorted(ind2lab.keys())]
                    else:
                        labels = [str(x) for x in ind2lab]

            if labels is None or len(labels) != len(probs):
                if len(probs) == 4:
                    # speechbrain/emotion-recognition-wav2vec2-IEMOCAP
                    labels = ["neu", "ang", "hap", "sad"]
                else:
                    # fallback: predicted label only
                    pred = str(text_lab).strip().lower()
                    result[pred] = float(score)

            if labels is not None and len(labels) == len(probs):
                for lab, p in zip(labels, probs):
                    result[str(lab).strip().lower()] = float(p)

        except Exception:
            pred = str(text_lab).strip().lower()
            result[pred] = float(score)

    # Map model labels to your canonical labels
    audio_map = CFG["label_maps"]["audio_to_canonical"]
    speechbrain_aliases = {
        "ang": "Angry",
        "anger": "Angry",
        "angry": "Angry",
        "hap": "Happy",
        "happy": "Happy",
        "happiness": "Happy",
        "neu": "Neutral",
        "neutral": "Neutral",
        "sad": "Sad",
        "sadness": "Sad",
    }
    mapped = empty_active_scores()

    for src_label, val in result.items():
        dst = audio_map.get(src_label) or speechbrain_aliases.get(src_label)
        add_score(mapped, dst, float(val))

    return normalize_to_active_scores(mapped)


# =========================================================
# TEXT BRANCH
# =========================================================
def text_emotion_scores(text: str) -> Dict[str, float]:
    global TEXT_MODEL_DISABLED, TEXT_MODEL_ERROR

    if not text.strip():
        return {}

    clf = load_text_model()
    if clf is None:
        return {}

    try:
        tokenizer = clf["tokenizer"]
        model = clf["model"]
        torch = clf["torch"]
        device = clf["device"]
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=min(int(getattr(tokenizer, "model_max_length", 512) or 512), 512),
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits[0]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        results = [
            {
                "label": clf["id2label"].get(idx, str(idx)),
                "score": float(prob),
            }
            for idx, prob in enumerate(probs)
        ]
    except Exception as e:
        TEXT_MODEL_DISABLED = True
        TEXT_MODEL_ERROR = f"{type(e).__name__}: {e}"
        print(
            "Warning: text emotion inference failed; continuing without text branch. "
            f"Reason: {TEXT_MODEL_ERROR}"
        )
        return {}

    raw = {}
    for item in results:
        raw[str(item["label"]).strip().lower()] = float(item["score"])

    text_map = CFG["label_maps"]["text_to_canonical"]
    mapped = empty_active_scores()

    for src_label, val in raw.items():
        dst = text_map.get(src_label)
        add_score(mapped, dst, float(val))

    mapped = normalize_to_active_scores(mapped)
    if not mapped:
        return {}
    return normalize_to_active_scores(calibrate_text_scores(text, mapped))


def load_wav_mono_float32(file_path: str, target_sr: int) -> np.ndarray:
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    sr, data = wavfile.read(file_path)
    if data.ndim > 1:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.integer):
        scale = max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / float(scale)
    else:
        data = data.astype(np.float32)

    if sr != target_sr:
        gcd = math.gcd(int(sr), int(target_sr))
        up = int(target_sr // gcd)
        down = int(sr // gcd)
        data = resample_poly(data, up, down).astype(np.float32)

    return np.ascontiguousarray(data, dtype=np.float32)


# =========================================================
# QUALITY / COMPATIBILITY
# =========================================================
def transcript_reliability(avg_logprob: Optional[float], no_speech_prob: Optional[float]) -> float:
    # no guessed thresholds here; convert available metadata to a soft reliability score
    parts = []

    if avg_logprob is not None:
        # map logprob smoothly
        parts.append(1.0 / (1.0 + math.exp(-2.0 * (avg_logprob + 1.0))))

    if no_speech_prob is not None:
        parts.append(1.0 - max(0.0, min(1.0, no_speech_prob)))

    if not parts:
        return 0.5

    return float(np.mean(parts))


def sfx_rejection_gate(
    feats: Dict[str, float],
    scaler: Dict[str, Dict[str, float]],
) -> Tuple[bool, List[str]]:
    """Aggressive SFX/effect rejection before model inference (inspired by hierarchical)."""
    reasons = []
    z_voiced = zscore_feature(
        feats["voiced_ratio"], scaler["voiced_ratio"]["mean"], scaler["voiced_ratio"]["std"]
    )
    z_flat = zscore_feature(
        feats["flatness_mean"], scaler["flatness_mean"]["mean"], scaler["flatness_mean"]["std"]
    )
    z_harm = zscore_feature(
        feats["harmonic_ratio"], scaler["harmonic_ratio"]["mean"], scaler["harmonic_ratio"]["std"]
    )
    z_rms = zscore_feature(feats["rms_mean"], scaler["rms_mean"]["mean"], scaler["rms_mean"]["std"])
    z_zcr = zscore_feature(feats["zcr_mean"], scaler["zcr_mean"]["mean"], scaler["zcr_mean"]["std"])
    z_burst = zscore_feature(
        feats["burst_ratio"], scaler["burst_ratio"]["mean"], scaler["burst_ratio"]["std"]
    )

    # Relative SFX patterns
    if z_voiced < -2.5:
        reasons.append("very_low_voiced")
    if z_flat > 2.0 and z_harm < -1.8:
        reasons.append("noise_like_high_flat_low_harm")
    if z_rms < -2.2 and z_voiced < -1.5:
        reasons.append("quiet_nonvoiced")
    if z_burst > 2.0 and feats["duration_sec"] < 1.5:
        reasons.append("short_bursty_impact")
    if z_zcr > 2.0 and z_flat > 1.5:
        reasons.append("high_zcr_flat_noise")

    return (len(reasons) > 0), reasons


def low_quality_flag(
    feats: Dict[str, float],
    scaler: Dict[str, Dict[str, float]],
) -> Tuple[bool, List[str]]:
    """Enhanced general quality check after SFX gate."""
    reasons = []
    z_voiced = zscore_feature(
        feats["voiced_ratio"], scaler["voiced_ratio"]["mean"], scaler["voiced_ratio"]["std"]
    )
    z_flat = zscore_feature(
        feats["flatness_mean"], scaler["flatness_mean"]["mean"], scaler["flatness_mean"]["std"]
    )
    z_harm = zscore_feature(
        feats["harmonic_ratio"], scaler["harmonic_ratio"]["mean"], scaler["harmonic_ratio"]["std"]
    )
    z_rms = zscore_feature(feats["rms_mean"], scaler["rms_mean"]["mean"], scaler["rms_mean"]["std"])

    if z_voiced < -2.0:
        reasons.append("very_low_voiced_ratio_vs_dataset")
    if z_flat > 2.0 and z_harm < -1.5:
        reasons.append("noise_like_vs_dataset")
    if z_rms < -2.0:
        reasons.append("too_quiet_vs_dataset")

    return (len(reasons) > 0), reasons


# =========================================================
# FUSION
# =========================================================
def prosody_emotion_scores(
    feats: Dict[str, float],
    scaler: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    prosody = empty_active_scores()

    z_f0_std = zscore_feature(feats["f0_std"], scaler["f0_std"]["mean"], scaler["f0_std"]["std"])
    z_rms = zscore_feature(feats["rms_mean"], scaler["rms_mean"]["mean"], scaler["rms_mean"]["std"])
    z_burst = zscore_feature(feats["burst_ratio"], scaler["burst_ratio"]["mean"], scaler["burst_ratio"]["std"])
    z_f0_mean = zscore_feature(feats["f0_mean"], scaler["f0_mean"]["mean"], scaler["f0_mean"]["std"])
    z_flat = zscore_feature(feats["flatness_mean"], scaler["flatness_mean"]["mean"], scaler["flatness_mean"]["std"])
    z_harm = zscore_feature(feats["harmonic_ratio"], scaler["harmonic_ratio"]["mean"], scaler["harmonic_ratio"]["std"])
    z_zcr = zscore_feature(feats["zcr_mean"], scaler["zcr_mean"]["mean"], scaler["zcr_mean"]["std"])

    if z_rms > 1.0 or z_burst > 1.0:
        # Loud/bursty anime speech is often panic or surprise. Keep Angry only for less high-pitch panic-like cases.
        if z_f0_mean < 0.75 and z_f0_std < 1.35:
            add_score(prosody, "Angry", 0.30)
            add_score(prosody, "Fear", 0.12)
        else:
            add_score(prosody, "Angry", 0.08)
            add_score(prosody, "Fear", 0.32)
            add_score(prosody, "Pleasant_Surprise", 0.18)
    if z_rms > 0.35 and z_burst > 0.35 and z_f0_std > 0.35:
        add_score(prosody, "Happy", 0.18)
        add_score(prosody, "Pleasant_Surprise", 0.16)
    if z_f0_mean > 1.0 and z_f0_std > 1.0:
        add_score(prosody, "Pleasant_Surprise", 0.45)
        add_score(prosody, "Fear", 0.30)
    if z_f0_std > 0.75 and z_rms < 0.25:
        add_score(prosody, "Fear", 0.22)
    if z_rms < -0.5 and z_f0_mean < -0.5:
        add_score(prosody, "Sad", 0.38)
    if abs(z_rms) < 0.55 and abs(z_f0_std) < 0.55 and abs(z_burst) < 0.85:
        add_score(prosody, "Neutral", 0.50)
    if z_flat > 0.9 and z_harm < -0.4 and z_zcr > 0.35:
        add_score(prosody, "Disgust", 0.12)

    return normalize_evidence_scores(prosody)


def branch_top_count(label: str, *branches: Dict[str, float]) -> int:
    return sum(1 for branch in branches if branch and top2(branch)[0][0] == label)


def strong_branch_support(label: str, text_scores: Dict[str, float], prosody: Dict[str, float]) -> bool:
    if label == "Angry":
        return float(text_scores.get(label, 0.0)) >= 0.22 or float(prosody.get(label, 0.0)) >= 0.34
    return float(text_scores.get(label, 0.0)) >= 0.22 or float(prosody.get(label, 0.0)) >= 0.28


def relaxed_branch_support(label: str, text_scores: Dict[str, float], prosody: Dict[str, float]) -> bool:
    if label == "Angry":
        return float(text_scores.get(label, 0.0)) >= 0.12 or float(prosody.get(label, 0.0)) >= 0.24
    return float(text_scores.get(label, 0.0)) >= 0.12 or float(prosody.get(label, 0.0)) >= 0.16


def auto_label_decision(
    top_label: str,
    top_score: float,
    gap: float,
    text_scores: Dict[str, float],
    prosody: Dict[str, float],
    transcript_rel: float,
) -> Tuple[bool, str]:
    policy = auto_label_policy()
    if policy == "volume":
        if transcript_rel < 0.35:
            return False, "weak_transcript_for_volume_policy"
        thresholds = {
            "Neutral": (0.32, 0.04),
            "Happy": (0.32, 0.04),
            "Fear": (0.32, 0.04),
            "Sad": (0.32, 0.04),
            "Pleasant_Surprise": (0.32, 0.04),
            "Angry": (0.32, 0.04),
            "Disgust": (0.32, 0.04),
        }
        min_score, min_gap = thresholds.get(top_label, (0.55, 0.12))
        if top_score >= min_score and gap >= min_gap:
            return True, f"volume_{top_label.lower()}_top_score"
        if top_score >= min_score + 0.08 and relaxed_branch_support(top_label, text_scores, prosody):
            return True, f"volume_{top_label.lower()}_branch_supported"
        return False, f"uncertain_{top_label.lower()}_volume"

    if policy == "balanced":
        if transcript_rel < 0.40:
            return False, "weak_transcript_for_balanced_policy"
        if top_label in ("Sad", "Angry"):
            if top_score >= 0.68 and gap >= 0.18 and relaxed_branch_support(top_label, text_scores, prosody):
                return True, f"balanced_{top_label.lower()}_supported"
            return False, "uncertain_dominant_emotion"

        if top_label in ("Disgust", "Pleasant_Surprise"):
            if top_score >= 0.66 and gap >= 0.18 and relaxed_branch_support(top_label, text_scores, prosody):
                return True, f"balanced_{top_label.lower()}_supported"
            return False, "uncertain_difficult_emotion"

        if top_label == "Neutral":
            if top_score >= 0.52 and gap >= 0.10 and (float(prosody.get("Neutral", 0.0)) >= 0.18 or transcript_rel >= 0.60):
                return True, "balanced_neutral_supported"
            return False, "uncertain_neutral"

        if top_label in ("Happy", "Fear"):
            if top_score >= 0.54 and gap >= 0.10 and relaxed_branch_support(top_label, text_scores, prosody):
                return True, f"balanced_{top_label.lower()}_supported"
            return False, f"uncertain_{top_label.lower()}"

        return False, "uncertain_fusion"

    if top_label in ("Sad", "Angry"):
        if top_score >= 0.82 and gap >= 0.28 and strong_branch_support(top_label, text_scores, prosody):
            return True, f"strict_{top_label.lower()}_supported"
        return False, "uncertain_dominant_emotion"

    if top_label in ("Disgust", "Pleasant_Surprise"):
        if top_score >= 0.82 and gap >= 0.24 and strong_branch_support(top_label, text_scores, prosody):
            return True, f"strict_{top_label.lower()}_supported"
        return False, "uncertain_difficult_emotion"

    if top_label == "Neutral":
        if top_score >= 0.58 and gap >= 0.12 and (float(prosody.get("Neutral", 0.0)) >= 0.24 or transcript_rel >= 0.65):
            return True, "neutral_supported"
        return False, "uncertain_neutral"

    if top_label in ("Happy", "Fear"):
        if top_score >= 0.62 and gap >= 0.14 and strong_branch_support(top_label, text_scores, prosody):
            return True, f"{top_label.lower()}_supported"
        return False, f"uncertain_{top_label.lower()}"

    return False, "uncertain_fusion"


def weighted_fusion(
    audio_scores: Dict[str, float],
    text_scores: Dict[str, float],
    feats: Dict[str, float],
    scaler: Dict[str, Dict[str, float]],
    transcript_rel: float,
) -> Dict[str, float]:
    weights = CFG["fusion"]["base_weights"]
    w_audio = min(float(weights["audio"]), 0.20)
    w_text = max(float(weights["text"]), 0.30)
    w_prosody = max(float(weights["prosody"]), 0.42)

    audio_scores = soften_distribution(normalize_to_active_scores(audio_scores), cap=0.56, floor=0.02)
    text_scores = soften_distribution(normalize_to_active_scores(text_scores), cap=0.72, floor=0.01)
    fused = empty_active_scores()

    audio_branch = dict(audio_scores)
    audio_top = top2(audio_scores)[0]
    if audio_branch and AUDIO_RUNTIME.get("is_fallback"):
        audio_branch = blend_with_uniform(audio_branch, 0.62)
        w_audio *= 0.65

    prosody = prosody_emotion_scores(feats, scaler)
    branch_tops = [top2(branch)[0][0] for branch in (audio_branch, text_scores, prosody) if branch]
    for dominant_label in ("Sad", "Angry"):
        if audio_top[0] == dominant_label and branch_tops.count(dominant_label) < 2:
            audio_branch[dominant_label] *= 0.20
            add_score(audio_branch, "Neutral", 0.15)
            audio_branch = normalize_scores(audio_branch)

    # audio branch
    for k, v in audio_branch.items():
        fused[k] += w_audio * float(v)

    # text branch scaled by transcript reliability
    for k, v in text_scores.items():
        fused[k] += (w_text * transcript_rel) * float(v)

    for k, v in prosody.items():
        fused[k] += w_prosody * float(v)

    for label in active_emotion_labels():
        support = branch_top_count(label, audio_branch, text_scores, prosody)
        if label in ("Sad", "Angry") and support < 2:
            fused[label] *= 0.45
        elif label in ("Happy", "Neutral", "Fear", "Pleasant_Surprise") and support >= 1:
            fused[label] *= 1.15
        if label in ("Happy", "Neutral", "Fear", "Pleasant_Surprise") and support >= 2:
            fused[label] *= 1.25

    return normalize_to_active_scores(fused)


# =========================================================
# OUTPUT
# =========================================================
def build_output_name(folder: str, idx: int, file_name: str, top1: Tuple[str, float], top2v: Tuple[str, float]) -> str:
    base, ext = os.path.splitext(file_name)
    return (
        f"{folder}_{idx:05d}"
        f"__pred_{safe_name(top1[0])}_{top1[1]:.3f}"
        f"__alt_{safe_name(top2v[0])}_{top2v[1]:.3f}"
        f"__{safe_name(base)}{ext}"
    )


def build_unique_output_path(folder: str, file_name: str, idx: int, top1: Tuple[str, float], top2v: Tuple[str, float]) -> str:
    folder_path = os.path.join(OUTPUT_BASE, folder)

    if PRESERVE_ORIGINAL_FILENAME:
        base_name, ext = os.path.splitext(file_name)
        candidate = os.path.join(folder_path, file_name)
        if not os.path.exists(candidate):
            return candidate

        suffix_idx = 1
        while True:
            candidate_name = f"{base_name}_copy_{suffix_idx:02d}{ext}"
            candidate = os.path.join(folder_path, candidate_name)
            if not os.path.exists(candidate):
                return candidate
            suffix_idx += 1

    out_name = build_output_name(folder, idx, file_name, top1, top2v)
    return os.path.join(folder_path, out_name)


def materialize_output_file(src: str, dst: str) -> None:
    """Hardlink same-volume outputs to avoid duplicating WAV data."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ensure_dirs()

    items = collect_dialogue_segments()
    if not items:
        print("No wav files found in configured input folders.")
        return

    source_counts = {}
    for item in items:
        source_counts[item["source_group"]] = source_counts.get(item["source_group"], 0) + 1

    print(f"Found {len(items)} wav files")
    print(f"Source distribution: {source_counts}")
    print(f"Emotion set: {emotion_set_name()} ({', '.join(active_emotion_labels())})")
    print("Label helper: ensemble pseudo-labeling (audio + text + prosody, optional GPT judge)")
    print("Fitting dataset-relative feature scaler...")

    scaler_items = [item for item in items if should_classify_source(item["source_group"])]
    if not scaler_items:
        scaler_items = items
    scaler = fit_dataset_scaler(scaler_items)

    output_folders = active_output_folders()
    counts = {k: 0 for k in output_folders}
    rows = []

    min_final_conf = float(CFG["fusion"]["min_final_confidence"])
    min_final_gap = float(CFG["fusion"]["min_final_gap"])
    max_auto_duration = float(CFG.get("review", {}).get("max_auto_duration_sec", 10.0))

    for item in items:
        file_path = item["file_path"]
        file_name = item["file_name"]
        source_group = item["source_group"]
        speaker = ""
        character = ""
        character_score = ""
        character_margin = ""
        gpt_judge_label = ""
        gpt_judge_confidence = ""
        gpt_judge_reason = ""
        gpt_judge_model = ""

        try:
            audio_scores = {}
            text_scores = {}
            tr_rel = None
            if not should_classify_source(source_group):
                pred_folder = "Rejected_LowQuality"
                final_scores = {}
                reason = f"source_folder_rejected:{source_group}"
                transcript = ""
                avg_logprob = None
                no_speech_prob = None
                feats = {}
            else:
                manifest_path = item.get("manifest_path") or ""
                manifest_info = read_manifest_transcript(manifest_path, file_name)
                transcript = manifest_info["transcript"]
                speaker = str(manifest_info.get("speaker") or "")
                character = str(manifest_info.get("character") or "")
                character_score = str(manifest_info.get("character_score") or "")
                character_margin = str(manifest_info.get("character_margin") or "")

                if not transcript:
                    transcribed = transcribe_if_needed(file_path)
                    transcript = transcribed["transcript"]
                    avg_logprob = transcribed["avg_logprob"]
                    no_speech_prob = transcribed["no_speech_prob"]
                else:
                    avg_logprob = manifest_info["avg_logprob"]
                    no_speech_prob = manifest_info["no_speech_prob"]

                feats = extract_features(file_path)

                # Step 1: Aggressive SFX rejection before expensive models
                is_sfx, sfx_reasons = sfx_rejection_gate(feats, scaler)
                if is_sfx:
                    pred_folder = "Rejected_LowQuality"
                    final_scores = {}
                    reason = "SFX_rejection | " + " | ".join(sfx_reasons)
                else:
                    is_low_quality, lq_reasons = low_quality_flag(feats, scaler)

                    if is_low_quality:
                        pred_folder = "Rejected_LowQuality"
                        final_scores = {}
                        reason = "Low_quality | " + " | ".join(lq_reasons)
                    else:
                        text_scores = text_emotion_scores(transcript)
                        audio_scores = audio_emotion_scores(file_path)
                        tr_rel = transcript_reliability(avg_logprob, no_speech_prob)

                        final_scores = weighted_fusion(
                            audio_scores=audio_scores,
                            text_scores=text_scores,
                            feats=feats,
                            scaler=scaler,
                            transcript_rel=tr_rel,
                        )

                        (top_label, top_score), (second_label, second_score) = top2(final_scores)
                        gap = float(top_score - second_score)
                        prosody_scores = prosody_emotion_scores(feats, scaler)
                        auto_ok, auto_reason = auto_label_decision(
                            top_label=top_label,
                            top_score=top_score,
                            gap=gap,
                            text_scores=normalize_to_active_scores(text_scores),
                            prosody=prosody_scores,
                            transcript_rel=tr_rel,
                        )

                        if float(feats.get("duration_sec", 0.0)) > max_auto_duration:
                            pred_folder = "Needs_Review"
                            reason = f"duration_too_long:{float(feats.get('duration_sec', 0.0)):.2f}s"
                        elif top_score < min_final_conf or gap < min_final_gap or not auto_ok:
                            pred_folder = "Needs_Review"
                            reason = auto_reason if not auto_ok else "uncertain_fusion"
                            gpt_judge = openai_uncertain_emotion_judge(
                                transcript=transcript,
                                final_scores=final_scores,
                                audio_scores=audio_scores,
                                text_scores=text_scores,
                                feats=feats,
                                reason=reason,
                            )
                            if gpt_judge:
                                gpt_judge_label = str(gpt_judge.get("emotion", ""))
                                gpt_judge_confidence = f"{float(gpt_judge.get('confidence', 0.0) or 0.0):.3f}"
                                gpt_judge_reason = str(gpt_judge.get("reason", ""))
                                gpt_judge_model = str(gpt_judge.get("model", ""))
                            if gpt_judge.get("accepted"):
                                pred_folder = str(gpt_judge["emotion"])
                                final_scores = forced_label_scores(
                                    pred_folder,
                                    float(gpt_judge.get("confidence", 0.0) or 0.0),
                                    final_scores,
                                )
                                reason = (
                                    "gpt_judge_uncertain_fusion"
                                    f":{pred_folder}:confidence={gpt_judge_confidence}"
                                )
                        else:
                            pred_folder = top_label
                            reason = f"ensemble_pseudo_label:{emotion_set_name()}emotion_audio_text_prosody:{auto_reason}"

            if final_scores:
                (top_label, top_score), (second_label, second_score) = top2(final_scores)
                gap = float(top_score - second_score)
            else:
                top_label, top_score = "none", 0.0
                second_label, second_score = "none", 0.0
                gap = 0.0

            counts[pred_folder] += 1
            out_path = build_unique_output_path(
                pred_folder,
                file_name,
                counts[pred_folder],
                (top_label, top_score),
                (second_label, second_score),
            )

            if COPY_FILES:
                materialize_output_file(file_path, out_path)
            else:
                shutil.move(file_path, out_path)

            feature_summary = build_feature_summary(feats)
            review_status, review_reason = review_decision(pred_folder, top_score, gap, tr_rel)
            row = {
                "source_file": file_path,
                "source_group": source_group,
                "speaker": speaker,
                "character": character,
                "character_score": character_score,
                "character_margin": character_margin,
                "pred_folder": pred_folder,
                "reason": reason,
                "review_status": review_status,
                "review_reason": review_reason,
                "gpt_judge_label": gpt_judge_label,
                "gpt_judge_confidence": gpt_judge_confidence,
                "gpt_judge_reason": gpt_judge_reason,
                "gpt_judge_model": gpt_judge_model,
                "top_label": top_label,
                "top_score": top_score,
                "second_label": second_label,
                "second_score": second_score,
                "top_gap": gap,
                "transcript_reliability": tr_rel,
                "transcript": compact_text_for_csv(transcript),
                "avg_logprob": avg_logprob,
                "no_speech_prob": no_speech_prob,
                "output_file": out_path,
            }
            row.update(build_score_columns(final_scores, "final_score"))
            row.update(feature_summary)
            row["details_json"] = json.dumps(
                {
                    "final_scores": final_scores,
                    "features": feats,
                    "audio_scores": audio_scores,
                    "text_scores": text_scores,
                    "emotion_set": emotion_set_name(),
                    "label_helper": CFG.get("classification", {}).get("label_helper", "ensemble_pseudo_labeling"),
                    "feature_extractors": CFG.get("classification", {}).get("feature_extractors", []),
                },
                ensure_ascii=False,
            )
            rows.append(row)

            print(f"{file_name} -> {pred_folder} ({reason})")

        except Exception as e:
            import traceback
            if getattr(e, "errno", None) == 28:
                print(
                    "Disk is full while writing classifier output. "
                    f"Free space on the output drive or move the output folder, then rerun. Last file: {file_name}"
                )
                break
            print(f"Failed: {file_name} | {e}")
            traceback.print_exc()

    manifest_csv = os.path.join(OUTPUT_BASE, "_multimodal_emotion_manifest.csv")
    manifest_jsonl = os.path.join(OUTPUT_BASE, "_multimodal_emotion_manifest_details.jsonl")
    review_summary = write_review_exports(OUTPUT_BASE, rows)
    score_fieldnames = [
        f"final_score_{label.lower()}"
        for label in CANONICAL_LABELS
        if label != "Needs_Review"
    ]
    feature_fieldnames = [
        "duration_sec",
        "voiced_ratio",
        "harmonic_ratio",
        "flatness_mean",
        "zcr_mean",
        "rms_mean",
        "burst_ratio",
    ]
    csv_fieldnames = [
        "source_file",
        "source_group",
        "speaker",
        "character",
        "character_score",
        "character_margin",
        "pred_folder",
        "reason",
        "review_status",
        "review_reason",
        "gpt_judge_label",
        "gpt_judge_confidence",
        "gpt_judge_reason",
        "gpt_judge_model",
        "top_label",
        "top_score",
        "second_label",
        "second_score",
        "top_gap",
        "transcript_reliability",
        "transcript",
        "avg_logprob",
        "no_speech_prob",
    ] + score_fieldnames + feature_fieldnames + [
        "output_file",
    ]
    with open(manifest_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in csv_fieldnames} for row in rows])
    with open(manifest_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nDone.")
    for folder in output_folders:
        print(f"{folder}: {counts[folder]}")
    print(f"\nManifest saved: {manifest_csv}")
    print(f"Manifest details saved: {manifest_jsonl}")
    print(f"High-confidence review manifest: {review_summary['high_confidence_csv']}")
    print(f"Manual review queue: {review_summary['manual_review_csv']}")
    print(f"High-confidence clips: {review_summary['high_confidence_count']}")
    print(f"Manual review clips: {review_summary['manual_review_count']}")
    print(f"Auto-labeled emotion folders: {review_summary['auto_labeled_emotion_count']}")
    print(f"Needs_Review folder clips: {review_summary['needs_review_folder_count']}")
    print(f"Low-confidence emotion-folder review clips: {review_summary['low_confidence_emotion_review_count']}")

    return {
        "counts": counts,
        "manifest_csv": manifest_csv,
        "manifest_jsonl": manifest_jsonl,
        **review_summary,
    }


if __name__ == "__main__":
    main()
