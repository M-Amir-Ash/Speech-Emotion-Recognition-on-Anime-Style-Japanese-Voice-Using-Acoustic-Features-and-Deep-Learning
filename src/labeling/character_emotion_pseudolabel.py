import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = BASE_DIR / "output" / "pipeline_workspace"
LABELING_CONFIG_PATH = BASE_DIR / "character_labeling_config.json"
CSV_ENCODING = "utf-8-sig"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant_Surprise", "Sad"]
EXTRA_FOLDERS = ["Needs_Review", "Rejected"]
HIGH_PRIORITY_REVIEW = "needs_review_high_priority"
DIFFICULT_EMOTIONS = {"Disgust", "Pleasant_Surprise"}
DOMINANT_AUDIO_EMOTIONS = {"Angry", "Sad"}
JAPANESE_SER_SUPPORTED_LABELS = {"Angry", "Happy", "Neutral", "Sad"}
MISSING_CANDIDATE_LABELS = {"Fear", "Disgust", "Pleasant_Surprise"}
CANDIDATE_FOLDERS = {
    "Fear": "Candidate_Fear",
    "Disgust": "Candidate_Disgust",
    "Pleasant_Surprise": "Candidate_Pleasant_Surprise",
}
SEED_4CLASS_LABELS = {"Angry", "Happy", "Neutral", "Sad"}
MINORITY_CLASS_BONUS = {
    "Fear": 1.0,
    "Disgust": 1.0,
    "Pleasant_Surprise": 1.0,
    "Angry": 0.6,
    "Happy": 0.4,
    "Sad": 0.4,
    "Neutral": 0.2,
}
REVIEW_CLASS_ORDER = {
    "Disgust": 0,
    "Pleasant_Surprise": 1,
    "Fear": 2,
    "Angry": 3,
    "Happy": 4,
    "Sad": 5,
    "Neutral": 6,
}
REQUIRED_7_EMOTIONS = {
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Pleasant_Surprise",
    "Sad",
}
ANGRY_TEXT_CUES = (
    "\u6012",      # anger/angry
    "\u30e0\u30ab",  # muka / irritated
    "\u3075\u3056\u3051",  # stop messing around
    "\u3046\u308b\u3055\u3044",  # noisy/shut up
    "\u9ed9\u308c",  # shut up
    "\u8a31\u3055",  # not forgive
    "\u8a31\u305b",  # cannot forgive
    "\u3070\u304b",  # idiot
    "\u30d0\u30ab",
    "\u3042\u307b",
    "\u30a2\u30db",
    "\u304f\u305d",
    "\u30af\u30bd",
    "\u304a\u524d",  # confrontational "you"
    "\u3066\u3081\u3048",
)


def load_labeling_config() -> dict:
    if not LABELING_CONFIG_PATH.is_file():
        return {}
    with LABELING_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def config_section(config: dict, *keys: str) -> dict:
    cur = config
    for key in keys:
        if not isinstance(cur, dict):
            return {}
        cur = cur.get(key, {})
    return cur if isinstance(cur, dict) else {}


def resolve_config_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else BASE_DIR / path


def load_audio_float32(path: str, target_sr: int) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    y, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1).astype(np.float32)
    if int(sample_rate) != int(target_sr):
        gcd = int(np.gcd(int(sample_rate), int(target_sr)))
        y = resample_poly(y, int(target_sr) // gcd, int(sample_rate) // gcd).astype(np.float32)
    return np.asarray(y, dtype=np.float32)


def load_s3prl_ser_model(model_name: str, config: dict):
    raise NotImplementedError("S3PRL backend is optional and not configured.")


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(value).strip()) or "Unknown"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def materialize_file(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def empty_scores() -> Dict[str, float]:
    return {label: 0.0 for label in EMOTIONS}


def normalize(scores: Dict[str, float]) -> Dict[str, float]:
    out = {label: max(0.0, float(scores.get(label, 0.0))) for label in EMOTIONS}
    total = sum(out.values())
    if total <= 1e-8:
        return out
    return {k: v / total for k, v in out.items()}


def soften_scores(scores: Dict[str, float], cap: float = 0.62, floor: float = 0.015) -> Dict[str, float]:
    if not scores or sum(scores.values()) <= 1e-8:
        return {}
    softened = {label: min(float(scores.get(label, 0.0)), cap) + floor for label in EMOTIONS}
    return normalize(softened)


def add_score(scores: Dict[str, float], label: str, value: float) -> None:
    if label in scores:
        scores[label] += float(value)


def top2(scores: Dict[str, float]) -> Tuple[Tuple[str, float], Tuple[str, float]]:
    if not scores or sum(float(v) for v in scores.values()) <= 1e-8:
        return ("none", 0.0), ("none", 0.0)
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    first = items[0]
    second = items[1] if len(items) > 1 else ("none", 0.0)
    return first, second


def is_review_status(status: str) -> bool:
    return status in {"needs_review", HIGH_PRIORITY_REVIEW}


def review_priority_score(top_label: str, top_score: float, top_gap: float, evidence_count: int) -> float:
    minority_bonus = MINORITY_CLASS_BONUS.get(top_label, 0.0)
    return round(
        0.40 * max(0.0, float(top_score))
        + 0.20 * min(max(0.0, float(top_gap)) / 0.30, 1.0)
        + 0.20 * min(max(0, int(evidence_count)) / 3.0, 1.0)
        + 0.20 * minority_bonus,
        6,
    )


def copy_unique(src: Path, dst_dir: Path) -> Path:
    ensure_dir(dst_dir)
    candidate = dst_dir / src.name
    if not candidate.exists():
        materialize_file(src, candidate)
        return candidate
    idx = 2
    while True:
        candidate = dst_dir / f"{src.stem}__dup{idx}{src.suffix}"
        if not candidate.exists():
            materialize_file(src, candidate)
            return candidate
        idx += 1


def load_character_manifest(output_root: Path) -> Dict[str, dict]:
    path = output_root / "character_dialogue" / "_character_dialogue_manifest.csv"
    if not path.is_file():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: could not read character dialogue manifest: {type(exc).__name__}: {exc}")
        return {}
    rows = {}
    for _, row in df.iterrows():
        out_file = str(row.get("output_file", ""))
        if out_file:
            rows[str(Path(out_file).resolve())] = row.to_dict()
    return rows


def iter_character_clips(output_root: Path, input_root: Optional[Path] = None) -> List[dict]:
    char_root = input_root or (output_root / "character_dialogue")
    metadata = load_character_manifest(output_root)
    clips = []
    for character_dir in sorted(p for p in char_root.iterdir() if p.is_dir()):
        character = character_dir.name
        for wav in sorted(character_dir.glob("*.wav")):
            meta = metadata.get(str(wav.resolve()), {})
            clips.append(
                {
                    "clip_id": meta.get("clip_id", wav.stem),
                    "file_path": str(wav),
                    "source_file": meta.get("source_file", wav.name),
                    "character_name": meta.get("character_name", character),
                    "speaker_cluster": meta.get("speaker_cluster", ""),
                    "transcript": meta.get("transcript", ""),
                    "duration_sec": float(meta.get("duration_sec") or 0.0),
                    "source_media": meta.get("source_media", ""),
                    "start_sec": meta.get("start_sec", ""),
                    "end_sec": meta.get("end_sec", ""),
                }
            )
    return clips


def fit_scaler(clips: List[dict], mec) -> Dict[str, Dict[str, float]]:
    return mec.fit_dataset_scaler([{"file_path": clip["file_path"]} for clip in clips])


def audio_quality_status(feats: Dict[str, float], scaler: Dict[str, Dict[str, float]], mec) -> Tuple[str, str]:
    try:
        is_sfx, sfx_reasons = mec.sfx_rejection_gate(feats, scaler)
        is_low, low_reasons = mec.low_quality_flag(feats, scaler)
        is_music, music_reasons = music_scene_rejection_gate(feats)
        if is_sfx or is_low:
            reasons = list(sfx_reasons or []) + list(low_reasons or [])
            return "reject", "|".join(reasons)
        if is_music:
            return "reject", "|".join(music_reasons)
    except Exception as exc:
        print(f"Warning: quality gate failed; using simple fallback. Reason: {type(exc).__name__}: {exc}")
        if float(feats.get("duration_sec", 0.0)) < 0.65:
            return "reject", "too_short"
        if float(feats.get("voiced_ratio", 0.0)) < 0.05:
            return "reject", "low_voiced_ratio"
    return "pass", ""


def music_scene_rejection_gate(feats: Dict[str, float]) -> Tuple[bool, List[str]]:
    duration = float(feats.get("duration_sec", 0.0))
    voiced = float(feats.get("voiced_ratio", 0.0))
    harmonic = float(feats.get("harmonic_ratio", 0.0))
    flatness = float(feats.get("flatness_mean", 0.0))
    zcr = float(feats.get("zcr_mean", 0.0))
    centroid = float(feats.get("centroid_mean", 0.0))
    rolloff = float(feats.get("rolloff_mean", 0.0))
    rms_mean = float(feats.get("rms_mean", 0.0))
    rms_std = float(feats.get("rms_std", 0.0))
    rms_cv = rms_std / (rms_mean + 1e-8)
    burst = float(feats.get("burst_ratio", 0.0))

    reasons = []
    sustained_music = duration >= 2.5 and harmonic > 0.58 and flatness < 0.16 and rms_cv < 1.25
    live_band_mix = duration >= 1.5 and harmonic > 0.48 and rolloff > 5200.0 and centroid > 2400.0 and burst < 8.0
    crowd_noise = flatness > 0.36 and zcr > 0.16 and voiced < 0.45
    if sustained_music:
        reasons.append("music_dominant_scene")
    if live_band_mix:
        reasons.append("band_or_live_music_scene")
    if crowd_noise:
        reasons.append("crowd_or_noise_scene")
    return bool(reasons), reasons


def normalize_label_key(label: str) -> str:
    key = str(label).strip().lower()
    key = key.replace("emotion:", "").replace("label:", "").strip()
    key = key.replace("-", "_").replace(" ", "_")
    return key


def canonical_audio_scores(raw: Dict[str, float], manual_map: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    """
    Convert raw model emotion scores into canonical 7-emotion scores.

    Canonical labels:
    - Angry
    - Disgust
    - Fear
    - Happy
    - Neutral
    - Pleasant_Surprise
    - Sad
    """
    mapped = empty_scores()

    aliases = {
        # Angry
        "ang": "Angry",
        "angry": "Angry",
        "anger": "Angry",
        "angriness": "Angry",

        # Disgust
        "dis": "Disgust",
        "disgust": "Disgust",
        "disgusted": "Disgust",
        "disgusting": "Disgust",

        # Fear
        "fea": "Fear",
        "fear": "Fear",
        "fearful": "Fear",
        "scared": "Fear",
        "afraid": "Fear",
        "anxiety": "Fear",
        "anxious": "Fear",

        # Happy
        "hap": "Happy",
        "happy": "Happy",
        "happiness": "Happy",
        "joy": "Happy",
        "joyful": "Happy",
        "excited": "Happy",

        # Neutral
        "neu": "Neutral",
        "neutral": "Neutral",
        "calm": "Neutral",
        "normal": "Neutral",

        # Pleasant Surprise
        "sur": "Pleasant_Surprise",
        "surprise": "Pleasant_Surprise",
        "surprised": "Pleasant_Surprise",
        "pleasant_surprise": "Pleasant_Surprise",
        "pleasant surprise": "Pleasant_Surprise",
        "anticipation": "Pleasant_Surprise",
        "ps": "Pleasant_Surprise",

        # Sad
        "sad": "Sad",
        "sadness": "Sad",
        "sorrow": "Sad",
        "unhappy": "Sad",
        "depressed": "Sad",
    }

    normalized_manual = {
        normalize_label_key(k): v
        for k, v in (manual_map or {}).items()
        if str(v) in EMOTIONS
    }

    for label, score in (raw or {}).items():
        key = normalize_label_key(label)
        dst = normalized_manual.get(key) or aliases.get(key) or aliases.get(key.replace("_", " "))

        if not dst:
            continue

        try:
            mapped[dst] += float(score)
        except (TypeError, ValueError):
            continue

    return normalize(mapped)


def prosody_scores(feats: Dict[str, float], scaler: Dict[str, Dict[str, float]], mec) -> Dict[str, float]:
    scores = empty_scores()
    z = lambda key: mec.zscore_feature(feats.get(key, 0.0), scaler.get(key, {}).get("mean", 0.0), scaler.get(key, {}).get("std", 1.0))
    z_f0_mean = z("f0_mean")
    z_f0_std = z("f0_std")
    z_rms = z("rms_mean")
    z_rms_std = z("rms_std")
    z_burst = z("burst_ratio")
    z_zcr = z("zcr_mean")
    z_flat = z("flatness_mean")
    z_voiced = z("voiced_ratio")
    z_harm = z("harmonic_ratio")

    if z_rms > 0.8 and z_burst > 0.6 and z_f0_std > 0.6:
        # Loud, bursty anime speech is often panic/surprise, not anger by itself.
        if z_f0_mean < 0.75 and z_f0_std < 1.35:
            add_score(scores, "Angry", 0.30)
            add_score(scores, "Fear", 0.12)
        else:
            add_score(scores, "Angry", 0.08)
            add_score(scores, "Fear", 0.34)
            add_score(scores, "Pleasant_Surprise", 0.20)
    if z_rms > 0.45 and z_burst > 0.35 and z_f0_std > 0.35:
        add_score(scores, "Happy", 0.18)
        add_score(scores, "Pleasant_Surprise", 0.12)
    if z_f0_mean > 0.8 and z_f0_std > 0.8:
        add_score(scores, "Pleasant_Surprise", 0.35)
        add_score(scores, "Fear", 0.25)
    if z_f0_mean > 0.35 and z_f0_std > 0.35 and z_rms > -0.35:
        add_score(scores, "Happy", 0.18)
        add_score(scores, "Pleasant_Surprise", 0.18)
    if z_f0_std > 0.75 and z_rms < 0.25:
        add_score(scores, "Fear", 0.20)
    if z_rms < -0.5 and z_f0_mean < -0.4:
        add_score(scores, "Sad", 0.45)
    if z_rms < -0.25 and abs(z_f0_std) < 0.65:
        add_score(scores, "Neutral", 0.20)
    if abs(z_rms) < 0.55 and abs(z_f0_std) < 0.55 and abs(z_burst) < 0.8:
        add_score(scores, "Neutral", 0.45)
    if z_flat > 1.0 and z_harm < -0.5 and z_zcr > 0.5:
        add_score(scores, "Disgust", 0.12)
    if z_voiced < -1.2:
        add_score(scores, "Fear", 0.12)
    if z_rms_std > 1.0:
        add_score(scores, "Fear", 0.10)
        add_score(scores, "Angry", 0.04)

    return normalize(scores)


class JapaneseSERBranch:
    def __init__(self, config: dict):
        self.config = config
        self.available = False
        self.model_name = ""
        self.backend = ""
        self.device = "cpu"
        self.status_rows: List[dict] = []
        self._load()

    def _actual_device(self, requested: str) -> str:
        try:
            import torch

            return "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _status(self, model_name: str, backend: str, status: str, error: str = "", used: bool = False, level: int = 0) -> None:
        self.status_rows.append(
            {
                "branch_name": "japanese_audio_ser",
                "model_name": model_name,
                "backend": backend,
                "status": status,
                "device": self.device,
                "error_message": error,
                "used_for_inference": "yes" if used else "no",
                "fallback_level": level,
                "supported_labels": "Angry|Happy|Neutral|Sad" if used else "",
                "unsupported_labels": "Fear|Disgust|Pleasant_Surprise" if used else "",
            }
        )

    def _parse_model_entry(self, entry) -> Tuple[str, str]:
        if isinstance(entry, dict):
            return str(entry.get("name", "")).strip(), str(entry.get("backend", "transformers")).strip().lower()
        return str(entry).strip(), "transformers"

    def _load(self) -> None:
        cfg = config_section(self.config, "models", "japanese_audio_ser")
        if not cfg.get("enabled", True):
            self._status("", "", "inactive", "disabled")
            return
        models = [self._parse_model_entry(x) for x in cfg.get("preferred_models", [])]
        models = [(name, backend) for name, backend in models if name]
        if not models:
            self._status("", "", "inactive", "no models configured")
            return
        self.device = self._actual_device(str(cfg.get("device", "cuda")).lower())
        for level, (model_name, backend) in enumerate(models, start=1):
            if backend == "s3prl":
                try:
                    load_s3prl_ser_model(model_name, self.config)
                except NotImplementedError:
                    msg = "S3PRL backend is optional and not configured."
                    print(
                        f"Model {model_name} requires S3PRL backend. Skipping because S3PRL inference is not configured.",
                        flush=True,
                    )
                    self._status(model_name, backend, "skipped_requires_s3prl", msg, level=level)
                    continue
            if backend != "transformers":
                msg = f"unsupported backend: {backend}"
                print(f"Warning: Japanese SER model skipped: {model_name} | {msg}", flush=True)
                self._status(model_name, backend, "failed", msg, level=level)
                continue
            try:
                import torch
                from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

                self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
                loaded = AutoModelForAudioClassification.from_pretrained(model_name, output_loading_info=True)
                if isinstance(loaded, tuple):
                    model, loading_info = loaded
                else:
                    model, loading_info = loaded, {}
                load_warning = ""
                missing = [str(x) for x in loading_info.get("missing_keys", [])]
                unexpected = [str(x) for x in loading_info.get("unexpected_keys", [])]
                classifier_mismatch = [
                    key for key in missing + unexpected
                    if "classifier" in key or "projector" in key
                ]
                if classifier_mismatch:
                    load_warning = "classifier/projector load mismatch: " + ";".join(classifier_mismatch[:12])
                    print(f"Warning: Japanese SER model loaded with classifier mismatch: {model_name}", flush=True)
                model = model.to(self.device)
                model.eval()
                self.torch = torch
                labels = [str(v) for v in getattr(model.config, "id2label", {}).values()]
                if labels and all(re.fullmatch(r"LABEL_\d+", label, flags=re.I) for label in labels):
                    manual_map = config_section(self.config, "manual_label_maps", model_name)
                    if not manual_map:
                        msg = "generic LABEL_n labels and no manual label map configured"
                        print(f"Warning: Japanese SER model skipped: {model_name} | {msg}", flush=True)
                        self._status(model_name, backend, "skipped_generic_labels", msg, level=level)
                        continue
                self.model = model
                self.model_name = model_name
                self.backend = backend
                self.available = True
                print(f"Japanese SER model loaded: {model_name} ({self.device})", flush=True)
                status = "active_but_unreliable" if load_warning else "active"
                self._status(model_name, backend, status, load_warning, used=True, level=level)
                for rest_level, (rest_name, rest_backend) in enumerate(models[level:], start=level + 1):
                    if rest_backend == "s3prl":
                        msg = "S3PRL backend is optional and not configured."
                        self._status(rest_name, rest_backend, "skipped_requires_s3prl", msg, level=rest_level)
                return
            except Exception as exc:
                self.available = False
                msg = f"{type(exc).__name__}: {exc}"
                print(f"Warning: Japanese SER model failed: {model_name} | {msg}", flush=True)
                self._status(model_name, backend, "failed", msg, level=level)

    def scores(self, wav_path: str) -> Dict[str, float]:
        if not self.available:
            return {}
        try:
            sampling_rate = int(getattr(self.extractor, "sampling_rate", 16000) or 16000)
            y = load_audio_float32(wav_path, sampling_rate)
            inputs = self.extractor(y, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self.torch.no_grad():
                logits = self.model(**inputs).logits[0]
                probs = self.torch.softmax(logits, dim=-1).detach().cpu().numpy()
            id2label = getattr(self.model.config, "id2label", {}) or {}
            raw = {str(id2label.get(i, f"LABEL_{i}")): float(prob) for i, prob in enumerate(probs)}
            manual_map = config_section(self.config, "manual_label_maps", self.model_name)
            scores = canonical_audio_scores(raw, manual_map=manual_map)
            return scores if sum(scores.values()) > 1e-8 else {}
        except Exception as exc:
            print(f"Warning: Japanese SER scoring failed for {Path(wav_path).name}: {type(exc).__name__}: {exc}")
            return {}


class JapaneseSSLEmbeddingBranch:
    def __init__(self, config: dict):
        self.config = config
        self.available = False
        self.model_name = ""
        self.status_rows: List[dict] = []
        self._load()

    def _default_device(self, requested: str) -> str:
        try:
            import torch

            return "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load(self) -> None:
        cfg = config_section(self.config, "models", "japanese_ssl_embedding")
        if not cfg.get("enabled", True):
            self.status_rows.append(
                {
                    "branch_name": "japanese_ssl_embedding",
                    "model_name": "",
                    "backend": "transformers",
                    "status": "inactive",
                    "device": "",
                    "error_message": "disabled",
                    "used_for_inference": "no",
                    "fallback_level": "",
                }
            )
            return
        classifier_path = resolve_config_path(str(cfg.get("classifier_path", "models/japanese_ssl_emotion_classifier.joblib")))
        if not classifier_path.is_file():
            print(f"Warning: Japanese SSL classifier not found; branch disabled: {classifier_path}")
            self.status_rows.append(
                {
                    "branch_name": "japanese_ssl_embedding",
                    "model_name": "",
                    "backend": "transformers",
                    "status": "skipped_missing_classifier",
                    "device": str(cfg.get("device", "cuda")),
                    "error_message": str(classifier_path),
                    "used_for_inference": "no",
                    "fallback_level": "",
                }
            )
            return
        try:
            import joblib
            import torch
            from transformers import AutoFeatureExtractor, AutoModel

            self.classifier = joblib.load(classifier_path)
            self.torch = torch
            self.device = self._default_device(str(cfg.get("device", "cuda")).lower())
            for model_name in [str(x) for x in cfg.get("preferred_models", []) if str(x).strip()]:
                try:
                    self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
                    self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
                    self.model_name = model_name
                    self.available = True
                    print(f"Japanese SSL embedding model loaded: {model_name} ({self.device})")
                    return
                except Exception as exc:
                    print(f"Warning: Japanese SSL model failed: {model_name} | {type(exc).__name__}: {exc}")
        except Exception as exc:
            print(f"Warning: Japanese SSL branch unavailable. Reason: {type(exc).__name__}: {exc}")

    def scores(self, wav_path: str) -> Dict[str, float]:
        if not self.available:
            return {}
        try:
            import librosa

            y, sr = librosa.load(wav_path, sr=16000, mono=True)
            inputs = self.extractor(y, sampling_rate=sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self.torch.no_grad():
                out = self.model(**inputs)
            emb = out.last_hidden_state.mean(dim=1).detach().cpu().numpy()
            if hasattr(self.classifier, "predict_proba"):
                probs = self.classifier.predict_proba(emb)[0]
                classes = getattr(self.classifier, "classes_", EMOTIONS)
                return canonical_audio_scores({str(label): float(prob) for label, prob in zip(classes, probs)})
            label = self.classifier.predict(emb)[0]
            return canonical_audio_scores({str(label): 1.0})
        except Exception as exc:
            print(f"Warning: Japanese SSL scoring failed for {Path(wav_path).name}: {type(exc).__name__}: {exc}")
            return {}


class General7ClassAudioSupportBranch:
    def __init__(self, config: dict):
        self.config = config
        self.available = False
        self.model_name = ""
        self.backend = "transformers"
        self.device = "cpu"
        self.status_rows: List[dict] = []
        self._load()

    def _actual_device(self, requested: str) -> str:
        try:
            import torch

            return "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _status(self, model_name: str, status: str, error: str = "", used: bool = False, level: int = 0) -> None:
        self.status_rows.append(
            {
                "branch_name": "general_7class_audio_support",
                "model_name": model_name,
                "backend": "transformers",
                "status": status,
                "device": self.device,
                "error_message": error,
                "used_for_inference": "yes" if used else "no",
                "fallback_level": level,
                "supported_labels": "Angry|Disgust|Fear|Happy|Neutral|Pleasant_Surprise|Sad" if used else "",
                "unsupported_labels": "",
            }
        )

    def _load(self) -> None:
        cfg = config_section(self.config, "models", "general_7class_audio_support")
        if not cfg.get("enabled", True):
            self._status("", "inactive", "disabled")
            return
        entries = cfg.get("preferred_models", [])
        models = []
        for entry in entries:
            if isinstance(entry, dict):
                models.append((str(entry.get("name", "")).strip(), str(entry.get("backend", "transformers")).lower()))
            else:
                models.append((str(entry).strip(), "transformers"))
        models = [(name, backend) for name, backend in models if name]
        self.device = self._actual_device(str(cfg.get("device", "cuda")).lower())
        if not models:
            self._status("", "inactive", "no models configured")
            return
        for level, (model_name, backend) in enumerate(models, start=1):
            if backend != "transformers":
                self._status(model_name, "failed", f"unsupported backend: {backend}", level=level)
                continue
            try:
                import torch
                from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

                loaded = AutoModelForAudioClassification.from_pretrained(model_name, output_loading_info=True)
                if isinstance(loaded, tuple):
                    model, loading_info = loaded
                else:
                    model, loading_info = loaded, {}
                model = model.to(self.device).eval()
                self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
                self.model = model
                self.torch = torch
                self.model_name = model_name
                self.available = True
                labels = [str(v) for v in getattr(model.config, "id2label", {}).values()]
                if labels and all(re.fullmatch(r"LABEL_\d+", label, flags=re.I) for label in labels):
                    self.available = False
                    self._status(model_name, "skipped_generic_labels", "generic LABEL_n labels", level=level)
                    continue
                warning = ""
                missing = [str(x) for x in loading_info.get("missing_keys", [])]
                unexpected = [str(x) for x in loading_info.get("unexpected_keys", [])]
                if missing or unexpected:
                    warning = "load mismatch: " + ";".join((missing + unexpected)[:12])
                print(f"General 7-class support model loaded: {model_name} ({self.device})", flush=True)
                self._status(model_name, "active", warning, used=True, level=level)
                return
            except Exception as exc:
                self.available = False
                msg = f"{type(exc).__name__}: {exc}"
                print(f"Warning: general 7-class support model failed: {model_name} | {msg}", flush=True)
                self._status(model_name, "failed", msg, level=level)

    def scores(self, wav_path: str) -> Dict[str, float]:
        if not self.available:
            return {}
        try:
            sampling_rate = int(getattr(self.extractor, "sampling_rate", 16000) or 16000)
            y = load_audio_float32(wav_path, sampling_rate)
            inputs = self.extractor(y, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self.torch.no_grad():
                logits = self.model(**inputs).logits[0]
                probs = self.torch.softmax(logits, dim=-1).detach().cpu().numpy()
            id2label = getattr(self.model.config, "id2label", {}) or {}
            raw = {str(id2label.get(i, f"LABEL_{i}")): float(prob) for i, prob in enumerate(probs)}
            scores = canonical_audio_scores(raw)
            return scores if sum(scores.values()) > 1e-8 else {}
        except Exception as exc:
            print(f"Warning: general 7-class support scoring failed for {Path(wav_path).name}: {type(exc).__name__}: {exc}")
            return {}


def agreement_count(branch_scores: Dict[str, Dict[str, float]], final_top: str) -> int:
    count = 0
    for scores in branch_scores.values():
        if not scores:
            continue
        if top2(scores)[0][0] == final_top:
            count += 1
    return count


def branch_bias_from_tops(branch_tops: Dict[str, List[str]]) -> Dict[str, dict]:
    report: Dict[str, dict] = {}
    for branch, tops in branch_tops.items():
        valid = [label for label in tops if label in EMOTIONS]
        if not valid:
            report[branch] = {
                "dominant_label": "",
                "dominant_label_count": 0,
                "total_predictions": 0,
                "dominant_label_ratio": 0.0,
                "biased_for_this_run": False,
            }
            continue
        counts = defaultdict(int)
        for label in valid:
            counts[label] += 1
        dominant, count = max(counts.items(), key=lambda item: item[1])
        ratio = count / max(1, len(valid))
        report[branch] = {
            "dominant_label": dominant,
            "dominant_label_count": count,
            "total_predictions": len(valid),
            "dominant_label_ratio": round(float(ratio), 6),
            "biased_for_this_run": ratio > 0.70,
        }
    return report


def filtered_agreement_count(branch_scores: Dict[str, Dict[str, float]], final_top: str, biased_branches: Dict[str, dict]) -> int:
    count = 0
    for branch, scores in branch_scores.items():
        if biased_branches.get(branch, {}).get("biased_for_this_run"):
            continue
        if not scores:
            continue
        if top2(scores)[0][0] == final_top:
            count += 1
    return count


def has_text_cue(transcript: str, cues: Tuple[str, ...]) -> bool:
    text = str(transcript or "")
    return any(cue in text for cue in cues)


def missing_emotion_evidence(
    label: str,
    clip_type: str,
    transcript: str,
    prosody_top: str,
    general_top: str,
    japanese_top: str,
    feats: Dict[str, float],
) -> Tuple[int, List[str]]:
    sources: List[str] = []
    duration = float(feats.get("duration_sec", 0.0) or 0.0)
    f0_mean = float(feats.get("f0_mean", 0.0) or 0.0)
    f0_std = float(feats.get("f0_std", 0.0) or 0.0)
    rms_std = float(feats.get("rms_std", 0.0) or 0.0)
    burst = float(feats.get("burst_ratio", 0.0) or 0.0)

    if label == "Fear":
        if clip_type == "short_vocal_reaction":
            sources.append("short_vocal_reaction")
        if prosody_top == "Fear":
            sources.append("prosody_top_fear")
        if f0_mean > 260.0 and f0_std > 40.0:
            sources.append("high_unstable_f0")
        if rms_std > 0.025:
            sources.append("unstable_rms")
        if has_text_cue(transcript, ("怖", "こわい", "やばい", "助け", "逃げ", "無理", "やめて")):
            sources.append("fear_text_cue")
        if general_top == "Fear":
            sources.append("general_7class_fear")
        if has_text_cue(transcript, ("怖", "こわい", "やばい", "助け", "逃げ", "無理", "やめて", "危ない")):
            sources.append("fear_text_cue")
    elif label == "Disgust":
        disgust_text = has_text_cue(transcript, ("気持ち悪", "最悪", "嫌", "うざ", "汚", "吐き気"))
        if disgust_text:
            sources.append("disgust_text_cue")
        if general_top == "Disgust":
            sources.append("general_7class_disgust")
        if japanese_top == "Angry" and disgust_text:
            sources.append("angry_ser_with_disgust_text")
        if has_text_cue(transcript, ("気持ち悪", "最悪", "嫌", "うざ", "汚", "吐き気", "キモ")):
            sources.append("disgust_text_cue")
    elif label == "Pleasant_Surprise":
        if clip_type == "short_vocal_reaction":
            sources.append("short_vocal_reaction")
        if prosody_top == "Pleasant_Surprise":
            sources.append("prosody_top_pleasant_surprise")
        if f0_mean > 260.0 and f0_std > 40.0:
            sources.append("high_unstable_f0")
        if burst > 3.5:
            sources.append("bursty_onset")
        if 0.5 <= duration <= 3.0:
            sources.append("short_duration")
        if has_text_cue(transcript, ("えっ", "あっ", "おお", "うそ", "まさか", "すごい", "びっくり", "驚")):
            sources.append("surprise_text_cue")
        if general_top == "Pleasant_Surprise":
            sources.append("general_7class_surprise")
        if has_text_cue(transcript, ("えっ", "あっ", "おお", "うそ", "まさか", "すごい", "びっくり", "驚", "やった")):
            sources.append("surprise_text_cue")
    return len(set(sources)), sorted(set(sources))


def fuse_scores(
    japanese_audio_ser_scores: Dict[str, float],
    japanese_ssl_embedding_scores: Dict[str, float],
    general_7class_scores: Dict[str, float],
    audio_baseline_scores: Dict[str, float],
    text_scores: Dict[str, float],
    prosody_branch: Dict[str, float],
    clip_type: str = "normal_dialogue",
    biased_branches: Optional[Dict[str, dict]] = None,
) -> Dict[str, float]:
    biased_branches = biased_branches or {}
    japanese_audio_ser_scores = soften_scores(japanese_audio_ser_scores, cap=0.82, floor=0.004) if japanese_audio_ser_scores else {}
    japanese_ssl_embedding_scores = soften_scores(japanese_ssl_embedding_scores, cap=0.76, floor=0.005) if japanese_ssl_embedding_scores else {}
    general_7class_scores = soften_scores(general_7class_scores, cap=0.72, floor=0.005) if general_7class_scores else {}
    audio_baseline_scores = soften_scores(audio_baseline_scores, cap=0.55, floor=0.02) if audio_baseline_scores else {}
    text_scores = soften_scores(text_scores, cap=0.55, floor=0.005) if text_scores else {}
    prosody_branch = soften_scores(prosody_branch, cap=0.74, floor=0.005) if prosody_branch else {}

    branch_scores = [japanese_audio_ser_scores, japanese_ssl_embedding_scores, general_7class_scores, audio_baseline_scores, text_scores, prosody_branch]
    branch_tops = [top2(scores)[0][0] for scores in branch_scores if scores and sum(scores.values()) > 1e-8]
    audio_top = top2(audio_baseline_scores)[0][0] if audio_baseline_scores else ""
    if audio_top in DOMINANT_AUDIO_EMOTIONS and branch_tops.count(audio_top) < 2:
        audio_baseline_scores = dict(audio_baseline_scores)
        audio_baseline_scores[audio_top] *= 0.18
        audio_baseline_scores["Neutral"] += 0.20
        audio_baseline_scores = normalize(audio_baseline_scores)

    if japanese_audio_ser_scores:
        weights = {
            "japanese_audio_ser": 0.45,
            "japanese_ssl_embedding": 0.0,
            "general_7class_audio_support": 0.20 if clip_type != "normal_dialogue" else 0.15,
            "audio_baseline_fallback": 0.05 if clip_type != "normal_dialogue" else 0.05,
            "prosody": 0.45 if clip_type == "short_vocal_reaction" else 0.40 if clip_type == "low_transcript_reliability" else 0.30,
            "text": 0.00 if clip_type != "normal_dialogue" else 0.05,
        }
        weights["japanese_audio_ser"] = 0.30 if clip_type == "short_vocal_reaction" else 0.35 if clip_type == "low_transcript_reliability" else 0.45
    elif japanese_ssl_embedding_scores:
        weights = {
            "japanese_audio_ser": 0.0,
            "japanese_ssl_embedding": 0.45,
            "general_7class_audio_support": 0.15,
            "audio_baseline_fallback": 0.10,
            "prosody": 0.45 if clip_type != "normal_dialogue" else 0.35,
            "text": 0.00 if clip_type != "normal_dialogue" else 0.10,
        }
    else:
        weights = {
            "japanese_audio_ser": 0.0,
            "japanese_ssl_embedding": 0.0,
            "general_7class_audio_support": 0.25 if clip_type != "normal_dialogue" else 0.20,
            "audio_baseline_fallback": 0.05 if clip_type != "normal_dialogue" else 0.10,
            "prosody": 0.70 if clip_type != "normal_dialogue" else 0.60,
            "text": 0.0 if clip_type == "short_vocal_reaction" else 0.10,
        }

    if biased_branches.get("audio_baseline_fallback", {}).get("biased_for_this_run"):
        weights["audio_baseline_fallback"] = min(float(weights.get("audio_baseline_fallback", 0.0)), 0.02)

    branches = {
        "japanese_audio_ser": japanese_audio_ser_scores,
        "japanese_ssl_embedding": japanese_ssl_embedding_scores,
        "general_7class_audio_support": general_7class_scores,
        "audio_baseline_fallback": audio_baseline_scores,
        "text": text_scores,
        "prosody": prosody_branch,
    }
    available_weight = sum(weights[name] for name, scores in branches.items() if scores and sum(scores.values()) > 1e-8)
    if available_weight <= 1e-8:
        return empty_scores()

    fused = empty_scores()
    for name, scores in branches.items():
        if not scores or sum(scores.values()) <= 1e-8:
            continue
        weight = weights[name] / available_weight
        for label in EMOTIONS:
            fused[label] += weight * scores.get(label, 0.0)

    # Disgust is intentionally conservative: keep it only when at least two branches agree.
    disgust_votes = sum(1 for scores in branch_scores if scores and top2(scores)[0][0] == "Disgust")
    if disgust_votes < 2:
        fused["Disgust"] *= 0.45
    for label in EMOTIONS:
        if branch_tops.count(label) < 2:
            fused[label] *= 0.70
    for label in DOMINANT_AUDIO_EMOTIONS:
        if branch_tops.count(label) < 2:
            fused[label] *= 0.45
    for label in ("Happy", "Neutral", "Fear", "Pleasant_Surprise"):
        if branch_tops.count(label) >= 1:
            fused[label] *= 1.18
        if branch_tops.count(label) >= 2:
            fused[label] *= 1.28
    return normalize(fused)


def confidence_decision(
    top_label: str,
    top_score: float,
    top_gap: float,
    agree: int,
    quality: str,
    speaker_cluster: str,
    character_name: str,
    audio_top: str = "",
    text_top: str = "",
    prosody_top: str = "",
    clip_type: str = "normal_dialogue",
    japanese_audio_ser_top: str = "",
    general_7class_top: str = "",
    evidence_count: int = 0,
    has_valid_branch: bool = False,
) -> Tuple[str, str]:
    known_character = bool(character_name and safe_name(character_name) != "Unknown")
    if quality != "pass":
        return "reject", "rejected_low_quality"
    if not known_character:
        return "reject", "rejected_unknown_speaker"
    if top_label not in EMOTIONS:
        return "reject", "rejected_unknown_label"
    if not has_valid_branch:
        return "reject", "rejected_no_valid_branch_scores"
    if top_score < 0.35 and evidence_count < 1:
        return "reject", "rejected_low_confidence"

    if top_label == "Disgust":
        if top_score >= 0.90 and top_gap >= 0.35 and evidence_count >= 2:
            return "auto_accept", "auto_accept_disgust_strict"
        if evidence_count >= 2 and top_score >= 0.50:
            return HIGH_PRIORITY_REVIEW, "candidate_disgust"
        if evidence_count >= 1 or top_score >= 0.35:
            return "needs_review", "candidate_disgust"
        return "reject", "rejected_low_confidence"

    if top_label == "Pleasant_Surprise":
        if top_score >= 0.86 and top_gap >= 0.30 and evidence_count >= 2:
            return "auto_accept", "auto_accept_pleasant_surprise"
        if evidence_count >= 2 and top_score >= 0.50:
            return HIGH_PRIORITY_REVIEW, "candidate_pleasant_surprise"
        if evidence_count >= 1 or top_score >= 0.35:
            return "needs_review", "candidate_pleasant_surprise"
        return "reject", "rejected_low_confidence"

    if top_label == "Fear":
        if top_score >= 0.84 and top_gap >= 0.30 and evidence_count >= 2:
            return "auto_accept", "auto_accept_fear"
        if evidence_count >= 2 and top_score >= 0.50:
            return HIGH_PRIORITY_REVIEW, "candidate_fear"
        if evidence_count >= 1 or top_score >= 0.35:
            return "needs_review", "candidate_fear"
        return "reject", "rejected_low_confidence"

    if top_label in JAPANESE_SER_SUPPORTED_LABELS:
        if (
            japanese_audio_ser_top == top_label
            and top_score >= 0.80
            and top_gap >= 0.25
            and evidence_count >= 2
        ):
            return "auto_accept", "auto_accept_supported_japanese_ser"
        if top_score >= 0.55:
            return "needs_review", "needs_review_supported_emotion"
        if top_score >= 0.35 or evidence_count >= 1:
            return "needs_review", "needs_review_low_confidence_candidate"
        return "reject", "rejected_low_confidence"

    if quality == "pass" and known_character and top_label == "Neutral":
        if top_score >= 0.62 and top_gap >= 0.35 and prosody_top == "Neutral":
            return "auto_accept", "auto_high_confidence"

    if quality == "pass" and known_character and top_label == "Sad":
        if not (top_score >= 0.90 and top_gap >= 0.35 and text_top == "Sad" and prosody_top == "Sad"):
            if top_score >= 0.55:
                return "needs_review", "needs_manual_review"
            return "needs_review", "needs_review_low_confidence_candidate"

    if quality == "pass" and known_character and top_label == "Angry":
        if not (
            top_score >= 0.86
            and top_gap >= 0.30
            and (text_top == "Angry" or prosody_top == "Angry")
        ):
            if top_score >= 0.55:
                return "needs_review", "needs_manual_review"
            return "needs_review", "needs_review_low_confidence_candidate"

    min_top_score = 0.88 if top_label in DIFFICULT_EMOTIONS else 0.80
    min_top_gap = 0.35 if top_label in DIFFICULT_EMOTIONS else 0.25
    if (
        top_score >= min_top_score
        and top_gap >= min_top_gap
        and agree >= 2
        and quality == "pass"
        and known_character
    ):
        return "auto_accept", "auto_high_confidence"
    if top_score >= 0.55 and quality == "pass" and known_character:
        return "needs_review", "needs_manual_review"
    if (top_score >= 0.35 or evidence_count >= 1) and quality == "pass" and known_character:
        return "needs_review", "needs_review_low_confidence_candidate"
    if quality != "pass":
        return "reject", "rejected_low_quality"
    if known_character:
        return "reject", "rejected_low_quality"
    return "reject", "rejected_unknown_speaker"


def output_bucket(confidence_status: str, top_label: str) -> str:
    if confidence_status == "auto_accept":
        return top_label
    if is_review_status(confidence_status):
        return "Needs_Review"
    return "Rejected"


def reset_speaker_emotion_folders(output_root: Path) -> None:
    speaker_root = output_root / "speaker_clusters"
    if not speaker_root.is_dir():
        return
    for speaker_dir in sorted(p for p in speaker_root.iterdir() if p.is_dir() and p.name.startswith("SPEAKER_")):
        for folder in EMOTIONS + EXTRA_FOLDERS:
            target = speaker_dir / folder
            if target.exists():
                shutil.rmtree(target)
            ensure_dir(target)


def speaker_emotion_bucket(confidence_status: str, top_label: str) -> str:
    if confidence_status == "auto_accept" and top_label in EMOTIONS:
        return top_label
    if is_review_status(confidence_status):
        return "Needs_Review"
    return "Rejected"


def run_character_emotion(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root or DEFAULT_WORKSPACE).resolve()
    output_name = safe_name(getattr(args, "classification_output_name", "character_emotion_classified"))
    character_root = output_root / output_name
    global_root = character_root / "_global"
    all_top_label_root = character_root / "_all_top_label"
    speaker_cluster_root = output_root / "speaker_clusters"

    if len(EMOTIONS) != 7 or set(EMOTIONS) != REQUIRED_7_EMOTIONS:
        raise RuntimeError(f"Character-aware labeling must use exactly 7 emotions, got: {EMOTIONS}")

    if not args.skip_emotion_labeling:
        if character_root.exists():
            shutil.rmtree(character_root)
    ensure_dir(character_root)
    ensure_dir(global_root)
    ensure_dir(all_top_label_root)

    if args.skip_emotion_labeling:
        manifest_path = character_root / "_character_emotion_manifest.csv"
        if not manifest_path.is_file():
            print(f"Character emotion manifest not found for shortlist refresh: {manifest_path}", flush=True)
            return 1
        with manifest_path.open("r", encoding=CSV_ENCODING, newline="") as f:
            rows = [normalize_review_row(row) for row in csv.DictReader(f)]
        write_review_queues(character_root, rows, int(getattr(args, "target_per_emotion", 100)))
        shortlist_count = write_review_shortlist(
            character_root,
            rows,
            int(getattr(args, "review_sample_per_class", 50)),
            int(getattr(args, "max_review_total", 500)),
            bool(getattr(args, "include_rejected_review", False)),
        )
        write_distribution_report(character_root, rows)
        write_count_consistency_report(character_root, rows)
        write_candidate_summary_report(character_root, rows)
        write_rejection_reason_report(character_root, rows)
        ensure_model_status_report_schema(character_root)
        original_review = sum(1 for row in rows if is_review_status(str(row.get("confidence_status", ""))))
        reduction = 0.0 if original_review == 0 else 100.0 * (1.0 - (shortlist_count / original_review))
        print(f"Original Needs_Review: {original_review}", flush=True)
        print(f"Review shortlist: {shortlist_count}", flush=True)
        print(f"Reduction: {reduction:.1f}%", flush=True)
        print(f"Review shortlist saved: {character_root / '_review_shortlist'}", flush=True)
        return 0

    reset_speaker_emotion_folders(output_root)

    import multimodal_emotion_classifier as mec
    import text_guided_emotion_classifier as tge

    labeling_config = load_labeling_config()
    japanese_ser = JapaneseSERBranch(labeling_config)
    japanese_ssl = JapaneseSSLEmbeddingBranch(labeling_config)
    general_7class = General7ClassAudioSupportBranch(labeling_config)
    active_japanese = japanese_ser.model_name if japanese_ser.available else "unavailable"
    print("Active labeling models:", flush=True)
    print(f"japanese_audio_ser: {active_japanese}", flush=True)
    print(f"general_7class_audio_support: {general_7class.model_name if general_7class.available else 'unavailable'}", flush=True)
    print("audio_baseline_fallback: lazy-load on first valid clip", flush=True)
    print("text: koshin2001/Japanese-to-emotions", flush=True)
    print("prosody: enabled", flush=True)
    print("s3prl_models: skipped unless configured", flush=True)
    if not japanese_ser.available:
        print("Warning: Japanese audio SER is unavailable. Labeling quality may be weak.", flush=True)
        print("Warning: No Japanese audio emotion model is active. Do not run the full dataset yet.", flush=True)
    print(
        "Warning: Japanese SER model only supports 4 classes. Fear, Disgust, and Pleasant_Surprise are candidate labels and require additional evidence.",
        flush=True,
    )
    if not general_7class.available:
        print("Warning: general 7-class support model is unavailable; missing-emotion candidates rely on prosody/text only.", flush=True)
    os.environ["EMOTION_SET"] = "7"
    mec.CFG.setdefault("classification", {})["emotion_set"] = "7"
    input_root = Path(args.input_root).resolve() if args.input_root else output_root / "character_dialogue"
    clips = iter_character_clips(output_root, input_root)
    if getattr(args, "classify_limit", None):
        clips = clips[: max(0, int(args.classify_limit))]
    if not clips:
        print(f"No character clips found: {output_root / 'character_dialogue'}")
        return 1
    if getattr(args, "classify_limit", None):
        print(f"Character emotion classify limit active: {len(clips)} clips")

    for character in sorted(set(safe_name(str(c["character_name"])) for c in clips)):
        for folder in EMOTIONS + EXTRA_FOLDERS + list(CANDIDATE_FOLDERS.values()):
            ensure_dir(character_root / character / folder)
    for folder in EMOTIONS + EXTRA_FOLDERS + list(CANDIDATE_FOLDERS.values()):
        ensure_dir(global_root / folder)
    for folder in EMOTIONS:
        ensure_dir(all_top_label_root / folder)
    seed_root = character_root / "_seed_4class"
    for folder in sorted(SEED_4CLASS_LABELS):
        ensure_dir(seed_root / folder)
    print(f"Character-aware emotion labels: {', '.join(EMOTIONS)}")

    scaler = fit_scaler(clips, mec)
    rows = []
    debug_rows = []
    branch_names = ["japanese_audio_ser", "general_7class_audio_support"]
    if japanese_ssl.available:
        branch_names.append("japanese_ssl_embedding")
    branch_names.extend(["audio_baseline_fallback", "prosody", "text", "final"])
    branch_totals = {name: empty_scores() for name in branch_names}
    branch_top_history = {name: [] for name in branch_names if name != "final"}
    biased_branches: Dict[str, dict] = {}

    for idx, clip in enumerate(clips, start=1):
        src = Path(clip["file_path"])
        character = safe_name(str(clip["character_name"]))
        speaker_cluster = str(clip.get("speaker_cluster", ""))
        transcript = str(clip.get("transcript", ""))
        audio_top = "none"
        text_top = "none"
        prosody_top = "none"
        japanese_audio_ser_top = "none"
        japanese_ssl_embedding_top = "none"
        general_7class_top = "none"
        audio_baseline_top = "none"
        clip_type = "normal_dialogue"
        try:
            feats = mec.extract_features(str(src))
            duration = float(feats.get("duration_sec", clip.get("duration_sec") or 0.0))
            clip_type = tge.detect_clip_type(transcript, duration, 0.50)
            quality, quality_reason = audio_quality_status(feats, scaler, mec)
            if clip_type == "short_vocal_reaction" and duration >= 0.5 and quality_reason in {"too_short", "very_low_harmonic_ratio"}:
                quality, quality_reason = "pass", ""
            if quality != "pass":
                clip_type = "bad_audio"
            japanese_audio_ser_scores = japanese_ser.scores(str(src)) if quality == "pass" else {}
            japanese_ssl_embedding_scores = japanese_ssl.scores(str(src)) if quality == "pass" else {}
            general_7class_scores = general_7class.scores(str(src)) if quality == "pass" else {}
            audio_baseline_scores = canonical_audio_scores(mec.audio_emotion_scores(str(src))) if quality == "pass" else {}
            if quality == "pass" and clip_type == "normal_dialogue":
                text_scores = normalize(mec.text_emotion_scores(transcript))
            else:
                text_scores = {}
            prosody_branch = tge.prosody_scores(feats, scaler, transcript, clip_type) if quality == "pass" else {}
            for history_name, history_scores in (
                ("japanese_audio_ser", japanese_audio_ser_scores),
                ("japanese_ssl_embedding", japanese_ssl_embedding_scores),
                ("general_7class_audio_support", general_7class_scores),
                ("audio_baseline_fallback", audio_baseline_scores),
                ("prosody", prosody_branch),
                ("text", text_scores),
            ):
                if history_name in branch_top_history and history_scores and sum(history_scores.values()) > 1e-8:
                    branch_top_history[history_name].append(top2(history_scores)[0][0])
            rolling_bias = branch_bias_from_tops(branch_top_history)
            audio_bias = rolling_bias.get("audio_baseline_fallback", {})
            if audio_bias.get("total_predictions", 0) >= 10 and audio_bias.get("biased_for_this_run"):
                biased_branches["audio_baseline_fallback"] = audio_bias
            final_scores = fuse_scores(
                japanese_audio_ser_scores,
                japanese_ssl_embedding_scores,
                general_7class_scores,
                audio_baseline_scores,
                text_scores,
                prosody_branch,
                clip_type=clip_type,
                biased_branches=biased_branches,
            )
            (top_label, top_score), (second_label, second_score) = top2(final_scores)
            top_gap = float(top_score) - float(second_score)
            branches = {
                "japanese_audio_ser": japanese_audio_ser_scores,
                "japanese_ssl_embedding": japanese_ssl_embedding_scores,
                "general_7class_audio_support": general_7class_scores,
                "audio_baseline_fallback": audio_baseline_scores,
                "text": text_scores,
                "prosody": prosody_branch,
            }
            agree = filtered_agreement_count(branches, top_label, biased_branches)
            japanese_audio_ser_top = top2(japanese_audio_ser_scores)[0][0] if japanese_audio_ser_scores else "unavailable"
            japanese_ssl_embedding_top = top2(japanese_ssl_embedding_scores)[0][0] if japanese_ssl_embedding_scores else "unavailable"
            general_7class_top = top2(general_7class_scores)[0][0] if general_7class_scores else "unavailable"
            audio_baseline_top = top2(audio_baseline_scores)[0][0] if audio_baseline_scores else "unavailable"
            audio_top = japanese_audio_ser_top if japanese_audio_ser_scores else audio_baseline_top
            text_top = top2(text_scores)[0][0] if text_scores else "disabled"
            prosody_top = top2(prosody_branch)[0][0]
            if top_label == "Angry" and not has_text_cue(transcript, ANGRY_TEXT_CUES):
                panic_like = prosody_top in {"Fear", "Pleasant_Surprise"} or general_7class_top in {"Fear", "Pleasant_Surprise"}
                weak_angry_support = japanese_audio_ser_top != "Angry" or general_7class_top in {"Fear", "Pleasant_Surprise"}
                if panic_like or weak_angry_support:
                    final_scores = dict(final_scores)
                    final_scores["Angry"] *= 0.35
                    if prosody_top in {"Fear", "Pleasant_Surprise"}:
                        final_scores[prosody_top] += 0.12
                    elif general_7class_top in {"Fear", "Pleasant_Surprise"}:
                        final_scores[general_7class_top] += 0.12
                    final_scores = normalize(final_scores)
                    (top_label, top_score), (second_label, second_score) = top2(final_scores)
                    top_gap = float(top_score) - float(second_score)
            evidence_count, evidence_sources = missing_emotion_evidence(
                top_label,
                clip_type,
                transcript,
                prosody_top,
                general_7class_top,
                japanese_audio_ser_top,
                feats,
            )
            if top_label in JAPANESE_SER_SUPPORTED_LABELS:
                supported_sources = []
                angry_text_cue = has_text_cue(transcript, ANGRY_TEXT_CUES)
                for source_name, source_top in (
                    ("japanese_audio_ser", japanese_audio_ser_top),
                    ("general_7class_audio_support", general_7class_top),
                    ("prosody", prosody_top),
                    ("text", text_top),
                ):
                    if source_top != top_label:
                        continue
                    if top_label == "Angry" and source_name == "prosody" and not angry_text_cue and japanese_audio_ser_top != "Angry" and general_7class_top != "Angry":
                        continue
                    if top_label == "Angry" and source_name == "text" and not angry_text_cue:
                        continue
                    if source_top == top_label:
                        supported_sources.append(source_name)
                if top_label == "Angry" and angry_text_cue:
                    supported_sources.append("angry_text_cue")
                evidence_sources = sorted(set(supported_sources))
                evidence_count = len(evidence_sources)
            has_valid_branch = any(
                scores and sum(scores.values()) > 1e-8
                for scores in [
                    japanese_audio_ser_scores,
                    general_7class_scores,
                    {} if biased_branches.get("audio_baseline_fallback", {}).get("biased_for_this_run") else audio_baseline_scores,
                    prosody_branch,
                    text_scores,
                ]
            )
            valid_branch_count = sum(
                1
                for scores in [
                    japanese_audio_ser_scores,
                    general_7class_scores,
                    {} if biased_branches.get("audio_baseline_fallback", {}).get("biased_for_this_run") else audio_baseline_scores,
                    prosody_branch,
                    text_scores,
                ]
                if scores and sum(scores.values()) > 1e-8
            )
            confidence_status, label_source = confidence_decision(
                top_label,
                top_score,
                top_gap,
                agree,
                quality,
                speaker_cluster,
                character,
                audio_top,
                text_top,
                prosody_top,
                clip_type,
                japanese_audio_ser_top,
                general_7class_top,
                evidence_count,
                has_valid_branch,
            )
        except Exception as exc:
            print(f"Warning: failed emotion pseudo-label for {src.name}: {type(exc).__name__}: {exc}")
            feats = {}
            quality = "reject"
            quality_reason = f"{type(exc).__name__}:{exc}"
            japanese_audio_ser_scores = japanese_ssl_embedding_scores = general_7class_scores = audio_baseline_scores = text_scores = prosody_branch = final_scores = empty_scores()
            top_label, top_score = "none", 0.0
            second_label, second_score = "none", 0.0
            top_gap = 0.0
            agree = 0
            evidence_count = 0
            evidence_sources = []
            has_valid_branch = False
            valid_branch_count = 0
            confidence_status, label_source = "reject", "rejected_unknown_label"

        bucket = output_bucket(confidence_status, top_label)
        speaker_bucket = speaker_emotion_bucket(confidence_status, top_label)
        char_dst = copy_unique(src, character_root / character / bucket) if args.copy_files else character_root / character / bucket / src.name
        global_dst = copy_unique(src, global_root / bucket) if args.copy_files else global_root / bucket / src.name
        all_top_dst = ""
        if top_label in EMOTIONS:
            all_top_dst = str(
                copy_unique(src, all_top_label_root / top_label)
                if args.copy_files
                else all_top_label_root / top_label / src.name
            )
        speaker_dst = ""
        if speaker_cluster and speaker_cluster != "SPEAKER_UNKNOWN":
            speaker_dst_path = speaker_cluster_root / safe_name(speaker_cluster) / speaker_bucket
            speaker_dst = str(copy_unique(src, speaker_dst_path) if args.copy_files else speaker_dst_path / src.name)
        candidate_emotion = top_label if top_label in MISSING_CANDIDATE_LABELS and confidence_status != "auto_accept" else ""
        candidate_output_file = ""
        global_candidate_output_file = ""
        if candidate_emotion and is_review_status(confidence_status):
            candidate_folder = CANDIDATE_FOLDERS[candidate_emotion]
            candidate_output_file = str(copy_unique(src, character_root / character / candidate_folder) if args.copy_files else character_root / character / candidate_folder / src.name)
            global_candidate_output_file = str(copy_unique(src, global_root / candidate_folder) if args.copy_files else global_root / candidate_folder / src.name)
        seed_4class_output_file = ""
        if (
            top_label in SEED_4CLASS_LABELS
            and japanese_audio_ser_top == top_label
            and float(top_score) >= 0.65
            and float(top_gap) >= 0.15
            and quality == "pass"
        ):
            seed_4class_output_file = str(copy_unique(src, seed_root / top_label) if args.copy_files else seed_root / top_label / src.name)
        priority = review_priority_score(top_label, float(top_score), float(top_gap), int(evidence_count))

        row = {
            "clip_id": clip.get("clip_id", src.stem),
            "source_file": clip.get("source_file", src.name),
            "output_file": str(char_dst),
            "global_output_file": str(global_dst),
            "all_top_label_output_file": all_top_dst,
            "speaker_emotion_output_file": speaker_dst,
            "candidate_output_file": candidate_output_file,
            "global_candidate_output_file": global_candidate_output_file,
            "seed_4class_output_file": seed_4class_output_file,
            "character_name": character,
            "speaker_cluster": speaker_cluster,
            "transcript": transcript,
            "transcript_reliability": "has_transcript" if str(transcript).strip() else "empty_transcript",
            "duration_sec": clip.get("duration_sec", feats.get("duration_sec", "")),
            "audio_quality_status": quality,
            "audio_quality_reason": quality_reason,
            "clip_type": clip_type,
            "top_label": top_label,
            "top_score": round(float(top_score), 6),
            "second_label": second_label,
            "second_score": round(float(second_score), 6),
            "top_gap": round(float(top_gap), 6),
            "model_agreement_count": agree,
            "confidence_status": confidence_status,
            "label_source": label_source,
            "review_priority_score": priority,
            "candidate_emotion": candidate_emotion,
            "audio_model_top": audio_top,
            "text_model_top": text_top,
            "prosody_top": prosody_top,
            "japanese_audio_ser_top": japanese_audio_ser_top,
            "japanese_ssl_embedding_top": japanese_ssl_embedding_top,
            "general_7class_audio_support_top": general_7class_top,
            "audio_baseline_top": audio_baseline_top,
            "japanese_audio_ser_model": japanese_ser.model_name,
            "japanese_ssl_embedding_model": japanese_ssl.model_name,
            "general_7class_audio_support_model": general_7class.model_name,
            "supported_by_japanese_ser": str(top_label in JAPANESE_SER_SUPPORTED_LABELS and japanese_audio_ser_top == top_label).lower(),
            "missing_class_candidate": str(top_label in MISSING_CANDIDATE_LABELS).lower(),
            "evidence_count": evidence_count,
            "evidence_sources": "|".join(evidence_sources),
            "has_valid_branch_scores": str(bool(has_valid_branch)).lower(),
            "valid_branch_count": valid_branch_count,
            "source_media": clip.get("source_media", ""),
            "start_sec": clip.get("start_sec", ""),
            "end_sec": clip.get("end_sec", ""),
        }
        for label in EMOTIONS:
            row[f"final_score_{label.lower()}"] = round(float(final_scores.get(label, 0.0)), 6)
        rows.append(row)
        for branch_name, scores in (
            ("japanese_audio_ser", japanese_audio_ser_scores),
            ("japanese_ssl_embedding", japanese_ssl_embedding_scores),
            ("general_7class_audio_support", general_7class_scores),
            ("audio_baseline_fallback", audio_baseline_scores),
            ("prosody", prosody_branch),
            ("text", text_scores),
            ("final", final_scores),
        ):
            if branch_name not in branch_totals:
                continue
            for label in EMOTIONS:
                branch_totals[branch_name][label] += float(scores.get(label, 0.0)) if scores else 0.0
        debug_rows.append(
            {
                "clip_id": row["clip_id"],
                "source_file": row["source_file"],
                "character_name": character,
                "speaker_cluster": speaker_cluster,
                "transcript": transcript,
                "clip_type": clip_type,
                "japanese_audio_ser_scores": japanese_audio_ser_scores,
                "japanese_ssl_embedding_scores": japanese_ssl_embedding_scores,
                "general_7class_audio_support_scores": general_7class_scores,
                "audio_baseline_scores": audio_baseline_scores,
                "prosody_scores": prosody_branch,
                "text_scores": text_scores,
                "final_scores": final_scores,
                "top_label": top_label,
                "top_score": top_score,
                "top_gap": top_gap,
                "confidence_status": confidence_status,
                "label_source": label_source,
                "review_priority_score": priority,
                "candidate_emotion": candidate_emotion,
                "supported_by_japanese_ser": top_label in JAPANESE_SER_SUPPORTED_LABELS and japanese_audio_ser_top == top_label,
                "missing_class_candidate": top_label in MISSING_CANDIDATE_LABELS,
                "evidence_count": evidence_count,
                "evidence_sources": evidence_sources,
            }
        )
        if idx % 100 == 0:
            print(f"Pseudo-labeled {idx}/{len(clips)} clips")

    final_branch_bias = branch_bias_from_tops(branch_top_history)
    audio_bias = final_branch_bias.get("audio_baseline_fallback", {})
    if audio_bias.get("biased_for_this_run"):
        biased_branches["audio_baseline_fallback"] = audio_bias
        print(
            f"Warning: audio_baseline_fallback is label-biased toward {audio_bias.get('dominant_label')}. Its fusion weight was reduced.",
            flush=True,
        )

    apply_label_collapse_protection(rows, character_root, global_root)

    manifest_path = character_root / "_character_emotion_manifest.csv"
    fields = [
        "clip_id",
        "source_file",
        "output_file",
        "global_output_file",
        "all_top_label_output_file",
        "speaker_emotion_output_file",
        "candidate_output_file",
        "global_candidate_output_file",
        "seed_4class_output_file",
        "character_name",
        "speaker_cluster",
        "transcript",
        "transcript_reliability",
        "duration_sec",
        "audio_quality_status",
        "clip_type",
        "top_label",
        "top_score",
        "second_label",
        "second_score",
        "top_gap",
        "model_agreement_count",
        "confidence_status",
        "label_source",
        "review_priority_score",
        "candidate_emotion",
        "final_score_angry",
        "final_score_disgust",
        "final_score_fear",
        "final_score_happy",
        "final_score_neutral",
        "final_score_pleasant_surprise",
        "final_score_sad",
        "audio_model_top",
        "text_model_top",
        "prosody_top",
        "japanese_audio_ser_top",
        "japanese_ssl_embedding_top",
        "general_7class_audio_support_top",
        "audio_baseline_top",
        "japanese_audio_ser_model",
        "japanese_ssl_embedding_model",
        "general_7class_audio_support_model",
        "supported_by_japanese_ser",
        "missing_class_candidate",
        "evidence_count",
        "evidence_sources",
        "has_valid_branch_scores",
        "valid_branch_count",
        "source_media",
        "start_sec",
        "end_sec",
    ]
    with manifest_path.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in rows])

    target_per_emotion = int(args.target_per_emotion)
    write_review_queues(character_root, rows, target_per_emotion)
    write_balanced_review_sample(character_root, rows, int(getattr(args, "review_sample_per_class", 20)))
    shortlist_count = write_review_shortlist(
        character_root,
        rows,
        int(getattr(args, "review_sample_per_class", 50)),
        int(getattr(args, "max_review_total", 500)),
        bool(getattr(args, "include_rejected_review", False)),
    )
    write_distribution_report(character_root, rows)
    write_count_consistency_report(character_root, rows)
    write_branch_report(character_root, branch_totals)
    write_branch_bias_report(character_root, final_branch_bias)
    write_candidate_summary_report(character_root, rows)
    write_rejection_reason_report(character_root, rows)
    model_status_rows = list(japanese_ser.status_rows) + list(japanese_ssl.status_rows) + list(general_7class.status_rows)
    model_status_rows.extend(
        [
            {
                "branch_name": "audio_baseline_fallback",
                "model_name": "speechbrain/emotion-recognition-wav2vec2-IEMOCAP; superb/wav2vec2-base-superb-er",
                "backend": "speechbrain/transformers",
                "status": "active" if sum(branch_totals["audio_baseline_fallback"].values()) > 1e-8 else "inactive",
                "device": "cuda",
                "error_message": "",
                "used_for_inference": "yes" if sum(branch_totals["audio_baseline_fallback"].values()) > 1e-8 else "no",
                "fallback_level": 6,
                "biased_for_this_run": str(audio_bias.get("biased_for_this_run", False)).lower(),
                "dominant_label": audio_bias.get("dominant_label", ""),
                "dominant_label_ratio": audio_bias.get("dominant_label_ratio", ""),
            },
            {
                "branch_name": "text",
                "model_name": "koshin2001/Japanese-to-emotions",
                "backend": "transformers",
                "status": "active" if sum(branch_totals["text"].values()) > 1e-8 else "inactive",
                "device": "cuda",
                "error_message": "",
                "used_for_inference": "yes" if sum(branch_totals["text"].values()) > 1e-8 else "no",
                "fallback_level": "",
            },
            {
                "branch_name": "prosody",
                "model_name": "prosody_rules",
                "backend": "acoustic_rules",
                "status": "active",
                "device": "cpu",
                "error_message": "",
                "used_for_inference": "yes",
                "fallback_level": "",
            },
        ]
    )
    write_model_status_report(character_root, model_status_rows)
    with (character_root / "_debug_scores.jsonl").open("w", encoding="utf-8") as f:
        for row in debug_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print_counts(rows, target_per_emotion)
    original_review = sum(1 for row in rows if is_review_status(str(row.get("confidence_status", ""))))
    reduction = 0.0 if original_review == 0 else 100.0 * (1.0 - (shortlist_count / original_review))
    print(f"Original Needs_Review: {original_review}", flush=True)
    print(f"Review shortlist: {shortlist_count}", flush=True)
    print(f"Reduction: {reduction:.1f}%", flush=True)
    print(f"Character emotion output: {character_root}")
    print(f"Global emotion output: {global_root}")
    print(f"All top-label emotion output: {all_top_label_root}")
    print(f"Manifest saved: {manifest_path}")
    return 0


def write_review_queues(root: Path, rows: List[dict], target_per_emotion: int) -> None:
    review_rows = [
        row
        for row in rows
        if is_review_status(row["confidence_status"])
        and row["audio_quality_status"] == "pass"
        and 1.0 <= float(row.get("duration_sec") or 0.0) <= 8.0
    ]
    minority_rank = {label: 0 for label in MISSING_CANDIDATE_LABELS}
    review_rows.sort(
        key=lambda r: (
            0 if r.get("confidence_status") == HIGH_PRIORITY_REVIEW else 1,
            -float(r.get("review_priority_score") or 0.0),
            minority_rank.get(r.get("top_label"), 1),
            str(r["character_name"]),
        )
    )

    accepted_counts = defaultdict(int)
    for row in rows:
        if row["confidence_status"] == "auto_accept":
            accepted_counts[row["top_label"]] += 1
    low_emotions = {label for label in EMOTIONS if accepted_counts[label] < target_per_emotion}
    low_rows = [row for row in review_rows if row["top_label"] in low_emotions]
    low_rows.sort(key=lambda r: (accepted_counts[r["top_label"]], -float(r.get("review_priority_score") or 0.0)))

    fields = list(rows[0].keys()) if rows else []
    for name, data in [
        ("_character_review_queue.csv", review_rows),
        ("_review_queue_low_count_emotions.csv", low_rows),
    ]:
        with (root / name).open("w", encoding=CSV_ENCODING, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{k: row.get(k, "") for k in fields} for row in data])


def write_balanced_review_sample(root: Path, rows: List[dict], per_class: int) -> None:
    sample_root = root / "_balanced_review_sample"
    if sample_root.exists():
        shutil.rmtree(sample_root)
    per_class = max(0, int(per_class))
    for label in EMOTIONS:
        ensure_dir(sample_root / label)
    if per_class == 0:
        return
    review_rows = [
        row for row in rows
        if is_review_status(str(row.get("confidence_status", ""))) and row.get("audio_quality_status") == "pass" and row.get("top_label") in EMOTIONS
    ]
    by_label: Dict[str, List[dict]] = defaultdict(list)
    for row in review_rows:
        by_label[str(row.get("top_label"))].append(row)
    for label, label_rows in by_label.items():
        label_rows.sort(key=lambda r: -float(r.get("review_priority_score") or 0.0))
        for row in label_rows[:per_class]:
            src = Path(str(row.get("output_file", "")))
            if src.is_file():
                copy_unique(src, sample_root / label)


def review_label(row: dict) -> str:
    candidate = str(row.get("candidate_emotion", "")).strip()
    label = candidate or str(row.get("top_label", "")).strip()
    return label if label in EMOTIONS else ""


def row_has_valid_branch_scores(row: dict) -> bool:
    explicit = str(row.get("has_valid_branch_scores", "")).strip().lower()
    if explicit in {"true", "1", "yes"}:
        return True
    if explicit in {"false", "0", "no"}:
        return False
    try:
        if int(float(row.get("valid_branch_count") or 0)) >= 1:
            return True
    except (TypeError, ValueError):
        pass
    try:
        if float(row.get("top_score") or 0.0) > 0.0:
            return True
    except (TypeError, ValueError):
        pass
    try:
        return int(float(row.get("evidence_count") or 0)) >= 1
    except (TypeError, ValueError):
        return False


def normalize_review_row(row: dict) -> dict:
    label = review_label(row) or str(row.get("top_label", ""))
    if not str(row.get("candidate_emotion", "")).strip() and label in EMOTIONS:
        row["candidate_emotion"] = label
    try:
        evidence_count = int(float(row.get("evidence_count") or 0))
    except (TypeError, ValueError):
        evidence_count = 0
        row["evidence_count"] = 0
    if not str(row.get("review_priority_score", "")).strip():
        row["review_priority_score"] = review_priority_score(
            label,
            float(row.get("top_score") or 0.0),
            float(row.get("top_gap") or 0.0),
            evidence_count,
        )
    if not str(row.get("transcript_reliability", "")).strip():
        row["transcript_reliability"] = "has_transcript" if str(row.get("transcript", "")).strip() else "empty_transcript"
    if not str(row.get("has_valid_branch_scores", "")).strip():
        row["has_valid_branch_scores"] = str(row_has_valid_branch_scores(row)).lower()
    if not str(row.get("evidence_sources", "")).strip():
        row["evidence_sources"] = "none"
    return row


def shortlist_candidate_rows(rows: List[dict], include_rejected: bool = False) -> List[dict]:
    candidates = []
    for row in rows:
        row = normalize_review_row(row)
        label = review_label(row)
        transcript = str(row.get("transcript", "")).strip()
        clip_type = str(row.get("clip_type", ""))
        if not label:
            continue
        status = str(row.get("confidence_status", ""))
        if not is_review_status(status) and not (include_rejected and status == "reject"):
            continue
        if row.get("audio_quality_status") != "pass":
            continue
        if not row_has_valid_branch_scores(row):
            continue
        if not transcript and clip_type != "short_vocal_reaction":
            continue
        candidates.append(row)
    return candidates


def shortlist_sort_key(row: dict) -> tuple:
    label = review_label(row)
    return (
        REVIEW_CLASS_ORDER.get(label, 99),
        -float(row.get("review_priority_score") or 0.0),
        -int(float(row.get("evidence_count") or 0)),
        -float(row.get("top_score") or 0.0),
    )


def write_review_shortlist(root: Path, rows: List[dict], per_class: int, max_total: int, include_rejected: bool = False) -> int:
    shortlist_root = root / "_review_shortlist"
    by_character_root = root / "_review_shortlist_by_character"
    for target in (shortlist_root, by_character_root):
        if target.exists():
            shutil.rmtree(target)
    for label in EMOTIONS:
        ensure_dir(shortlist_root / label)

    per_class = max(0, int(per_class))
    max_total = max(0, int(max_total))
    candidates = shortlist_candidate_rows(rows, include_rejected=include_rejected)
    by_label: Dict[str, List[dict]] = defaultdict(list)
    for row in candidates:
        by_label[review_label(row)].append(row)

    selected: List[dict] = []
    for label in sorted(EMOTIONS, key=lambda item: REVIEW_CLASS_ORDER.get(item, 99)):
        label_rows = sorted(by_label.get(label, []), key=shortlist_sort_key)
        selected.extend(label_rows[:per_class] if per_class else label_rows)
    selected = sorted(selected, key=shortlist_sort_key)
    if max_total and len(selected) > max_total:
        selected = selected[:max_total]

    manifest_rows = []
    copied_by_clip = set()
    for row in selected:
        label = review_label(row)
        src = Path(str(row.get("output_file", "")))
        if not src.is_file():
            src = Path(str(row.get("global_output_file", "")))
        if not src.is_file():
            continue
        key = (str(src.resolve()), label)
        if key in copied_by_clip:
            continue
        copied_by_clip.add(key)
        character = safe_name(str(row.get("character_name", "Unknown")))
        global_copy = copy_unique(src, shortlist_root / label)
        ensure_dir(by_character_root / character / label)
        copy_unique(src, by_character_root / character / label)
        manifest_rows.append(
            {
                "clip_id": row.get("clip_id", ""),
                "source_file": row.get("source_file", ""),
                "output_file": str(global_copy),
                "character_name": row.get("character_name", ""),
                "speaker_cluster": row.get("speaker_cluster", ""),
                "candidate_emotion": row.get("candidate_emotion", "") or label,
                "final_top": row.get("top_label", ""),
                "confidence_status": row.get("confidence_status", ""),
                "review_priority_score": row.get("review_priority_score", ""),
                "evidence_count": row.get("evidence_count", ""),
                "evidence_sources": row.get("evidence_sources", ""),
                "top_score": row.get("top_score", ""),
                "top_gap": row.get("top_gap", ""),
                "transcript": row.get("transcript", ""),
                "clip_type": row.get("clip_type", ""),
                "audio_quality_status": row.get("audio_quality_status", ""),
            }
        )

    manifest_fields = [
        "clip_id",
        "source_file",
        "output_file",
        "character_name",
        "speaker_cluster",
        "candidate_emotion",
        "final_top",
        "confidence_status",
        "review_priority_score",
        "evidence_count",
        "evidence_sources",
        "top_score",
        "top_gap",
        "transcript",
        "clip_type",
        "audio_quality_status",
    ]
    with (root / "_review_shortlist_manifest.csv").open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    workload_rows = []
    selected_counts = defaultdict(int)
    original_counts = defaultdict(int)
    for row in candidates:
        original_counts[review_label(row)] += 1
    for row in manifest_rows:
        selected_counts[str(row.get("candidate_emotion", ""))] += 1
    for label in EMOTIONS:
        original = original_counts[label]
        shortlist = selected_counts[label]
        workload_rows.append(
            {
                "category": label,
                "original_count": original,
                "shortlist_count": shortlist,
                "reduced_by": original - shortlist,
                "notes": "minority class retained by priority" if label in {"Disgust", "Pleasant_Surprise"} else "",
            }
        )
    original_total = len(candidates)
    shortlist_total = len(manifest_rows)
    workload_rows.append(
        {
            "category": "TOTAL",
            "original_count": original_total,
            "shortlist_count": shortlist_total,
            "reduced_by": original_total - shortlist_total,
            "notes": f"max_review_total={max_total}; review_sample_per_class={per_class}",
        }
    )
    with (root / "_review_workload_report.csv").open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "original_count", "shortlist_count", "reduced_by", "notes"])
        writer.writeheader()
        writer.writerows(workload_rows)
    return shortlist_total


def move_output_to_review(row: dict, character_root: Path, global_root: Path) -> None:
    character = safe_name(str(row.get("character_name", "Unknown")))
    for key, root in (("output_file", character_root / character), ("global_output_file", global_root)):
        old_path = Path(str(row.get(key, "")))
        if not old_path.is_file():
            continue
        new_path = copy_unique(old_path, root / "Needs_Review")
        try:
            if old_path.resolve() != new_path.resolve():
                old_path.unlink(missing_ok=True)
        except Exception:
            pass
        row[key] = str(new_path)


def apply_label_collapse_protection(rows: List[dict], character_root: Path, global_root: Path) -> None:
    report_path = character_root / "_label_collapse_report.csv"
    auto_rows = [row for row in rows if row.get("confidence_status") == "auto_accept"]
    if not auto_rows:
        with report_path.open("w", encoding=CSV_ENCODING, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["label", "count", "ratio", "action_taken"])
            writer.writeheader()
        return
    counts = defaultdict(int)
    for row in auto_rows:
        counts[str(row.get("top_label", ""))] += 1
    dominant, count = max(counts.items(), key=lambda item: item[1])
    if dominant not in EMOTIONS or count / max(1, len(auto_rows)) <= 0.70:
        with report_path.open("w", encoding=CSV_ENCODING, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["label", "count", "ratio", "action_taken"])
            writer.writeheader()
            for label, label_count in sorted(counts.items()):
                writer.writerow({"label": label, "count": label_count, "ratio": round(label_count / max(1, len(auto_rows)), 6), "action_taken": "none"})
        return
    print(f"Warning: label collapse detected. Most clips are assigned to {dominant}.")
    changed = 0
    for row in auto_rows:
        if row.get("top_label") != dominant:
            continue
        top_score = float(row.get("top_score") or 0.0)
        top_gap = float(row.get("top_gap") or 0.0)
        if top_score >= 0.90 and top_gap >= 0.35:
            continue
        row["confidence_status"] = "needs_review"
        row["label_source"] = f"needs_review_label_collapse_{dominant}"
        move_output_to_review(row, character_root, global_root)
        changed += 1
    with report_path.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "count", "ratio", "action_taken"])
        writer.writeheader()
        writer.writerow(
            {
                "label": dominant,
                "count": count,
                "ratio": round(count / max(1, len(auto_rows)), 6),
                "action_taken": f"moved_{changed}_low_confidence_to_needs_review",
            }
        )


def write_distribution_report(root: Path, rows: List[dict]) -> None:
    out_rows = []
    for emotion in EMOTIONS + EXTRA_FOLDERS:
        if emotion in EMOTIONS:
            data = [row for row in rows if row.get("top_label") == emotion]
        elif emotion == "Needs_Review":
            data = [row for row in rows if is_review_status(str(row.get("confidence_status", "")))]
        else:
            data = [row for row in rows if row.get("confidence_status") == "reject"]
        out_rows.append(
            {
                "emotion": emotion,
                "auto_accept_count": sum(1 for row in data if row.get("confidence_status") == "auto_accept"),
                "needs_review_count": sum(1 for row in data if is_review_status(str(row.get("confidence_status", "")))),
                "needs_review_high_priority_count": sum(1 for row in data if row.get("confidence_status") == HIGH_PRIORITY_REVIEW),
                "rejected_count": sum(1 for row in data if row.get("confidence_status") == "reject"),
                "candidate_count": sum(
                    1
                    for row in data
                    if is_review_status(str(row.get("confidence_status", "")))
                    or str(row.get("missing_class_candidate", "")).lower() == "true"
                ),
                "seed_4class_count": sum(1 for row in data if str(row.get("seed_4class_output_file", "")).strip()),
                "total_count": len(data),
            }
        )
    with (root / "_emotion_distribution_report.csv").open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "emotion",
                "auto_accept_count",
                "needs_review_count",
                "needs_review_high_priority_count",
                "rejected_count",
                "candidate_count",
                "seed_4class_count",
                "total_count",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)


def manifest_count_for_label(rows: List[dict], label: str) -> int:
    if label in EMOTIONS:
        return sum(1 for row in rows if row.get("top_label") == label and row.get("confidence_status") == "auto_accept")
    if label == "Needs_Review":
        return sum(1 for row in rows if is_review_status(str(row.get("confidence_status", ""))))
    if label == "Rejected":
        return sum(1 for row in rows if row.get("confidence_status") == "reject")
    return 0


def report_count_for_label(root: Path, label: str) -> int:
    path = root / "_emotion_distribution_report.csv"
    if not path.is_file():
        return 0
    with path.open("r", encoding=CSV_ENCODING, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("emotion") != label:
                continue
            if label in EMOTIONS:
                return int(float(row.get("auto_accept_count") or 0))
            if label == "Needs_Review":
                return int(float(row.get("needs_review_count") or 0))
            if label == "Rejected":
                return int(float(row.get("rejected_count") or 0))
    return 0


def write_count_consistency_report(root: Path, rows: List[dict]) -> None:
    labels = EMOTIONS + EXTRA_FOLDERS
    output_keys = {
        **{label: "output_file" for label in EMOTIONS},
        "Needs_Review": "output_file",
        "Rejected": "output_file",
    }
    global_keys = {
        **{label: "global_output_file" for label in EMOTIONS},
        "Needs_Review": "global_output_file",
        "Rejected": "global_output_file",
    }
    report_rows = []
    for source, base, keys in (("character", root, output_keys), ("global", root / "_global", global_keys)):
        for label in labels:
            folder_count = len(list((base / label).glob("*.wav"))) if (base / label).is_dir() and source == "global" else 0
            if source == "character":
                folder_count = sum(
                    len(list((char_dir / label).glob("*.wav")))
                    for char_dir in root.iterdir()
                    if char_dir.is_dir() and not char_dir.name.startswith("_") and (char_dir / label).is_dir()
                )
            manifest_count = manifest_count_for_label(rows, label)
            report_count = report_count_for_label(root, label)
            difference = max(abs(folder_count - manifest_count), abs(report_count - manifest_count))
            status = "ok" if difference == 0 else "mismatch"
            if status != "ok":
                print(
                    f"Warning: count mismatch [{source}] {label}: folder={folder_count}, manifest={manifest_count}, report={report_count}",
                    flush=True,
                )
            report_rows.append(
                {
                    "source": source,
                    "emotion": label,
                    "folder_count": folder_count,
                    "manifest_count": manifest_count,
                    "report_count": report_count,
                    "status": status,
                    "difference": difference,
                }
            )
    with (root / "_count_consistency_report.csv").open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "emotion", "folder_count", "manifest_count", "report_count", "status", "difference"])
        writer.writeheader()
        writer.writerows(report_rows)


def write_branch_report(root: Path, branch_totals: Dict[str, Dict[str, float]]) -> None:
    rows = []
    for branch in branch_totals:
        row = {"branch": branch}
        row.update({label: round(float(branch_totals[branch].get(label, 0.0)), 6) for label in EMOTIONS})
        rows.append(row)
    with (root / "_branch_prediction_report.csv").open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["branch"] + EMOTIONS)
        writer.writeheader()
        writer.writerows(rows)


def write_branch_bias_report(root: Path, bias_rows: Dict[str, dict]) -> None:
    fields = ["branch", "dominant_label", "dominant_label_count", "total_predictions", "dominant_label_ratio", "biased_for_this_run", "action_taken"]
    with (root / "_branch_bias_report.csv").open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for branch, row in bias_rows.items():
            writer.writerow(
                {
                    "branch": branch,
                    "dominant_label": row.get("dominant_label", ""),
                    "dominant_label_count": row.get("dominant_label_count", 0),
                    "total_predictions": row.get("total_predictions", 0),
                    "dominant_label_ratio": row.get("dominant_label_ratio", 0.0),
                    "biased_for_this_run": str(bool(row.get("biased_for_this_run", False))).lower(),
                    "action_taken": "fusion_weight_reduced" if branch == "audio_baseline_fallback" and row.get("biased_for_this_run") else "",
                }
            )


def write_candidate_summary_report(root: Path, rows: List[dict]) -> None:
    fields = ["emotion", "candidate_count", "high_priority_count", "average_top_score", "average_evidence_count", "top_evidence_sources"]
    out_rows = []
    for emotion in EMOTIONS:
        data = [
            row for row in rows
            if row.get("top_label") == emotion and (is_review_status(str(row.get("confidence_status", ""))) or row.get("candidate_emotion"))
        ]
        evidence_counter = defaultdict(int)
        for row in data:
            for source in str(row.get("evidence_sources", "")).split("|"):
                if source:
                    evidence_counter[source] += 1
        top_sources = sorted(evidence_counter.items(), key=lambda item: item[1], reverse=True)[:5]
        out_rows.append(
            {
                "emotion": emotion,
                "candidate_count": len(data),
                "high_priority_count": sum(1 for row in data if row.get("confidence_status") == HIGH_PRIORITY_REVIEW),
                "average_top_score": round(float(np.mean([float(row.get("top_score") or 0.0) for row in data])) if data else 0.0, 6),
                "average_evidence_count": round(float(np.mean([float(row.get("evidence_count") or 0.0) for row in data])) if data else 0.0, 6),
                "top_evidence_sources": "|".join(f"{name}:{count}" for name, count in top_sources),
            }
        )
    with (root / "_candidate_summary_report.csv").open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)


def write_rejection_reason_report(root: Path, rows: List[dict]) -> None:
    counts = defaultdict(int)
    for row in rows:
        if row.get("confidence_status") == "reject":
            reason = str(row.get("label_source") or row.get("audio_quality_reason") or "reject")
            counts[reason] += 1
    with (root / "_rejection_reason_report.csv").open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["reject_reason", "count"])
        writer.writeheader()
        for reason, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
            writer.writerow({"reject_reason": reason, "count": count})


def write_model_status_report(root: Path, rows: List[dict]) -> None:
    fields = [
        "branch_name",
        "model_name",
        "backend",
        "status",
        "device",
        "error_message",
        "used_for_inference",
        "fallback_level",
        "supported_labels",
        "unsupported_labels",
        "biased_for_this_run",
        "dominant_label",
        "dominant_label_ratio",
        "reliability_warning",
    ]
    with (root / "_model_status_report.csv").open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {
                **{k: row.get(k, "") for k in fields},
                "reliability_warning": row.get("reliability_warning", row.get("error_message", "") if row.get("status") == "active_but_unreliable" else ""),
            }
            for row in rows
        ])


def ensure_model_status_report_schema(root: Path) -> None:
    path = root / "_model_status_report.csv"
    if not path.is_file():
        return
    with path.open("r", encoding=CSV_ENCODING, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    needed = [
        "branch_name",
        "model_name",
        "backend",
        "status",
        "device",
        "error_message",
        "used_for_inference",
        "fallback_level",
        "supported_labels",
        "unsupported_labels",
        "biased_for_this_run",
        "dominant_label",
        "dominant_label_ratio",
        "reliability_warning",
    ]
    changed = False
    for field in needed:
        if field not in fields:
            fields.append(field)
            changed = True
    for row in rows:
        if row.get("branch_name") == "japanese_audio_ser" and "classifier/projector load mismatch" in str(row.get("error_message", "")):
            if row.get("status") == "active":
                row["status"] = "active_but_unreliable"
                changed = True
            if not row.get("reliability_warning"):
                row["reliability_warning"] = row.get("error_message", "")
                changed = True
    if not changed:
        return
    with path.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in rows])


def print_counts(rows: List[dict], target_per_emotion: int = 100) -> None:
    counts = defaultdict(lambda: defaultdict(int))
    per_character_emotions = defaultdict(lambda: defaultdict(int))
    emotion_counts = {
        "auto_accept": defaultdict(int),
        "needs_review": defaultdict(int),
        "reject": defaultdict(int),
    }
    for row in rows:
        status = row["confidence_status"]
        label = row["top_label"]
        counts[row["character_name"]][status] += 1
        if status == "auto_accept" and label in EMOTIONS:
            per_character_emotions[row["character_name"]][label] += 1
        elif is_review_status(status):
            per_character_emotions[row["character_name"]]["Needs_Review"] += 1
        elif status == "reject":
            per_character_emotions[row["character_name"]]["Rejected"] += 1
        if status == HIGH_PRIORITY_REVIEW and label in EMOTIONS:
            emotion_counts["needs_review"][label] += 1
        elif status in emotion_counts and label in EMOTIONS:
            emotion_counts[status][label] += 1

    print("\nEmotion counts per character:")
    for character in sorted(per_character_emotions):
        print(f"Character: {character}")
        for label in EMOTIONS:
            print(f"  {label}: {per_character_emotions[character][label]}")
        print(f"  Needs_Review: {per_character_emotions[character]['Needs_Review']}")
        print(f"  Rejected: {per_character_emotions[character]['Rejected']}")

    print("\nGlobal emotion counts:")
    for label in EMOTIONS:
        print(
            f"  {label}: {emotion_counts['auto_accept'][label]}"
        )
    print(f"  Needs_Review: {sum(1 for row in rows if is_review_status(str(row.get('confidence_status', ''))))}")
    print(f"  Rejected: {sum(1 for row in rows if row.get('confidence_status') == 'reject')}")

    if any(emotion_counts["auto_accept"][label] < target_per_emotion for label in DIFFICULT_EMOTIONS):
        print(
            "Warning: Disgust/Pleasant_Surprise has low accepted sample count. "
            "This class may require manual review or should be discussed as a challenging minority class."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Character-aware 7-emotion pseudo-labeling.")
    parser.add_argument("--input-root", default=str(DEFAULT_WORKSPACE / "character_dialogue"))
    parser.add_argument("--output-root", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--target-per-emotion", type=int, default=100)
    parser.add_argument("--review-sample-per-class", type=int, default=50)
    parser.add_argument("--max-review-total", type=int, default=500)
    parser.add_argument("--include-rejected-review", action="store_true")
    parser.add_argument("--classify-limit", type=int, help="Limit clips for quick Step 9 testing.")
    parser.add_argument("--skip-emotion-labeling", action="store_true")
    parser.add_argument("--copy-files", action="store_true", default=True)
    parser.add_argument("--embedding-model", default="facebook/hubert-base-ls960")
    parser.add_argument("--cluster-embedding-device", choices=["cuda", "cpu"])
    parser.add_argument("--classification-output-name", default="character_emotion_classified")
    parser.add_argument("--trained-model-weights", action="store_true")
    return parser


def main() -> int:
    return run_character_emotion(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
