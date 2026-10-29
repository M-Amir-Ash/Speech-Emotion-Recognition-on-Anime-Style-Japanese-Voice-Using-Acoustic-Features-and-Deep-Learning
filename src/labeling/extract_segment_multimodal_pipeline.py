import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

import librosa
import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel
from moviepy import VideoFileClip
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Couldn't find ffmpeg or avconv.*",
        category=RuntimeWarning,
        module="pydub.utils",
    )
    from pydub import AudioSegment

try:
    import imageio_ffmpeg

    AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass


def patch_torchaudio_legacy_backend_api() -> None:
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
from pydub.silence import detect_nonsilent


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
HF_TOKEN_FILE = os.path.join(BASE_DIR, "TokenKeyHuggyFace")
HF_TOKEN_FILE_TXT = os.path.join(BASE_DIR, "TokenKeyHuggyFace.txt")

PIPELINE_WORKSPACE_ROOT = os.path.join(OUTPUT_DIR, "pipeline_workspace")
PIPELINE_WORKSPACE = PIPELINE_WORKSPACE_ROOT
EXTRACTED_ROOT = os.path.join(PIPELINE_WORKSPACE, "_extracted_audio")
EXTRACTED_AUDIO_PATH = os.path.join(EXTRACTED_ROOT, "audio.wav")
SEGMENT_ROOT = os.path.join(PIPELINE_WORKSPACE, "segmented")
JP_DIR = os.path.join(SEGMENT_ROOT, "japanese_dialogue")
VOCAL_EFFECT_DIR = os.path.join(SEGMENT_ROOT, "character_vocal_effect")
OTHER_DIR = os.path.join(SEGMENT_ROOT, "other_or_effect")
SPEAKER_DIR = os.path.join(SEGMENT_ROOT, "by_speaker")
MIXED_SPEAKER_DIR = os.path.join(SEGMENT_ROOT, "mixed_speaker_rejected")
TEMP_DIR = os.path.join(SEGMENT_ROOT, "_temp")
CLASSIFIED_DIR = os.path.join(PIPELINE_WORKSPACE, "multimodal_emotion_classified")
CURRENT_SOURCE_STEM = "audio"

TARGET_SR = 16000
MIN_SEGMENT_SEC = 0.65
MIN_EMOTION_CLIP_SEC = 0.65
MIN_VOICED_RATIO = 0.22
MIN_HARMONIC_RATIO = 0.18
MAX_FLATNESS = 0.32
MAX_ZCR = 0.22
MIN_RMS_MEAN = 0.0035
MAX_CENTROID_MEAN = 3800.0
MAX_DIALOGUE_MUSIC_RATIO = 0.58
TARGET_LANGUAGE = "ja"
MIN_LANGUAGE_PROB = 0.45
MIN_JAPANESE_CHAR_RATIO = 0.42
MIN_JAPANESE_CHAR_COUNT = 3
MAX_ASCII_CHAR_RATIO = 0.30
MAX_DIGIT_CHAR_RATIO = 0.25
WHISPER_SEGMENT_MODEL = "tiny"
WHISPERX_MODEL = "base"
PYANNOTE_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
WHISPER_SEGMENT_BEAM_SIZE = 1
WHISPER_SEGMENT_BATCH_SIZE = 8
SEGMENTATION_RUNTIME = {}
MAX_SEGMENT_SEC = 8.0
TARGET_CHUNK_SEC = 5.5
MIN_CHUNK_SEC = 0.60
SILENCE_MIN_MS = 220
KEEP_SILENCE_MS = 90
SILENCE_THRESH_OFFSET_DB = 18.0

CLASSIFIER_OUTPUT_FOLDERS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Pleasant_Surprise",
    "Sad",
    "Needs_Review",
    "Rejected_LowQuality",
]

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
}
ALLOWED_MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_classifier_output_tree() -> None:
    ensure_dir(CLASSIFIED_DIR)
    for folder in CLASSIFIER_OUTPUT_FOLDERS:
        ensure_dir(os.path.join(CLASSIFIED_DIR, folder))


def reset_wav_folder(path: str) -> None:
    ensure_dir(path)
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isfile(full_path):
            os.remove(full_path)


def safe_name(text: str, max_len: int = 40) -> str:
    if not text:
        return "no_text"
    cleaned = "".join(c if c.isalnum() or c in ("_", "-", " ") else "_" for c in text)
    cleaned = "_".join(cleaned.split())
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = "no_text"
    return cleaned[:max_len]


def japanese_char_ratio(text: str) -> float:
    if not text:
        return 0.0

    total = 0
    japanese = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        code = ord(ch)
        is_hiragana = 0x3040 <= code <= 0x309F
        is_katakana = 0x30A0 <= code <= 0x30FF
        is_katakana_ext = 0x31F0 <= code <= 0x31FF
        is_cjk = 0x4E00 <= code <= 0x9FFF
        if is_hiragana or is_katakana or is_katakana_ext or is_cjk:
            japanese += 1

    if total == 0:
        return 0.0
    return japanese / total


def text_char_stats(text: str) -> Dict[str, float]:
    if not text:
        return {
            "total": 0,
            "japanese": 0,
            "ascii_letters": 0,
            "digits": 0,
        }

    total = 0
    japanese = 0
    ascii_letters = 0
    digits = 0

    for ch in text:
        if ch.isspace():
            continue
        total += 1
        code = ord(ch)
        is_hiragana = 0x3040 <= code <= 0x309F
        is_katakana = 0x30A0 <= code <= 0x30FF
        is_katakana_ext = 0x31F0 <= code <= 0x31FF
        is_cjk = 0x4E00 <= code <= 0x9FFF
        if is_hiragana or is_katakana or is_katakana_ext or is_cjk:
            japanese += 1
        elif ch.isascii() and ch.isalpha():
            ascii_letters += 1
        elif ch.isdigit():
            digits += 1

    return {
        "total": total,
        "japanese": japanese,
        "ascii_letters": ascii_letters,
        "digits": digits,
    }


def looks_like_japanese_text(text: str, min_ratio: float = MIN_JAPANESE_CHAR_RATIO) -> bool:
    cleaned = (text or "").strip()
    if len(cleaned) < 2:
        return False
    stats = text_char_stats(cleaned)
    total = max(1, int(stats["total"]))
    jp_ratio = float(stats["japanese"]) / total
    ascii_ratio = float(stats["ascii_letters"]) / total
    digit_ratio = float(stats["digits"]) / total
    if int(stats["japanese"]) < MIN_JAPANESE_CHAR_COUNT:
        return False
    if jp_ratio < min_ratio:
        return False
    if ascii_ratio > MAX_ASCII_CHAR_RATIO:
        return False
    if digit_ratio > MAX_DIGIT_CHAR_RATIO:
        return False
    return True


def configure_run_workspace(input_path: str, run_index: int, total_runs: int) -> None:
    global PIPELINE_WORKSPACE, EXTRACTED_ROOT, EXTRACTED_AUDIO_PATH, SEGMENT_ROOT
    global JP_DIR, VOCAL_EFFECT_DIR, OTHER_DIR, SPEAKER_DIR, MIXED_SPEAKER_DIR, TEMP_DIR
    global CLASSIFIED_DIR, CURRENT_SOURCE_STEM

    stem = safe_name(os.path.splitext(os.path.basename(input_path))[0], max_len=64)
    CURRENT_SOURCE_STEM = stem or f"input_{run_index:03d}"
    PIPELINE_WORKSPACE = PIPELINE_WORKSPACE_ROOT
    EXTRACTED_ROOT = os.path.join(PIPELINE_WORKSPACE, "_extracted_audio")
    EXTRACTED_AUDIO_PATH = os.path.join(EXTRACTED_ROOT, f"{CURRENT_SOURCE_STEM}.wav")
    SEGMENT_ROOT = os.path.join(PIPELINE_WORKSPACE, "segmented")
    JP_DIR = os.path.join(SEGMENT_ROOT, "japanese_dialogue")
    VOCAL_EFFECT_DIR = os.path.join(SEGMENT_ROOT, "character_vocal_effect")
    OTHER_DIR = os.path.join(SEGMENT_ROOT, "other_or_effect")
    SPEAKER_DIR = os.path.join(SEGMENT_ROOT, "by_speaker")
    MIXED_SPEAKER_DIR = os.path.join(SEGMENT_ROOT, "mixed_speaker_rejected")
    TEMP_DIR = os.path.join(SEGMENT_ROOT, "_temp")
    CLASSIFIED_DIR = os.path.join(PIPELINE_WORKSPACE, "multimodal_emotion_classified")


def list_supported_inputs(input_dir: str) -> List[str]:
    candidates = []
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    for name in sorted(os.listdir(input_dir)):
        full_path = os.path.join(input_dir, name)
        if not os.path.isfile(full_path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in ALLOWED_MEDIA_EXTENSIONS:
            candidates.append(full_path)

    if not candidates:
        raise FileNotFoundError(
            f"No supported media found in {input_dir}. "
            f"Supported extensions: {sorted(ALLOWED_MEDIA_EXTENSIONS)}"
        )

    return candidates


def resolve_input_paths(input_arg: Optional[str]) -> List[str]:
    if not input_arg:
        return list_supported_inputs(INPUT_DIR)

    candidate = os.path.abspath(input_arg)
    if os.path.isdir(candidate):
        return list_supported_inputs(candidate)
    if os.path.isfile(candidate):
        ext = os.path.splitext(candidate)[1].lower()
        if ext not in ALLOWED_MEDIA_EXTENSIONS:
            raise FileNotFoundError(
                f"Unsupported input file: {candidate}. "
                f"Supported extensions: {sorted(ALLOWED_MEDIA_EXTENSIONS)}"
            )
        return [candidate]
    raise FileNotFoundError(f"Input path not found: {candidate}")


def normalize_audio_file(src_path: str, dst_path: str) -> None:
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_channels(1).set_frame_rate(TARGET_SR).set_sample_width(2)
    audio.export(dst_path, format="wav")


def preferred_torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def extract_audio(input_path: str, output_wav: str) -> str:
    ensure_dir(os.path.dirname(output_wav))
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".wav":
        normalize_audio_file(input_path, output_wav)
        return output_wav

    if ext != ".wav":
        cmd = [
            get_ffmpeg_exe(),
            "-y",
            "-i",
            input_path,
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(TARGET_SR),
            "-sample_fmt",
            "s16",
            output_wav,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "ffmpeg failed").strip())
        return output_wav

    normalize_audio_file(input_path, output_wav)
    return output_wav


def clip_features(wav_path: str) -> Dict[str, float]:
    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    if len(y) == 0:
        return {
            "duration_sec": 0.0,
            "voiced_ratio": 0.0,
            "harmonic_ratio": 0.0,
            "flatness_mean": 1.0,
            "zcr_mean": 1.0,
            "rms_mean": 0.0,
            "centroid_mean": 0.0,
        }

    duration_sec = len(y) / sr
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    f0 = librosa.yin(y, fmin=50, fmax=600, sr=sr)
    voiced_mask = np.isfinite(f0) & (f0 > 0)
    voiced_ratio = float(np.mean(voiced_mask)) if len(voiced_mask) > 0 else 0.0

    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = float(np.mean(np.abs(y_harm))) if len(y_harm) > 0 else 0.0
    perc_energy = float(np.mean(np.abs(y_perc))) if len(y_perc) > 0 else 0.0
    harmonic_ratio = harm_energy / (harm_energy + perc_energy + 1e-8)
    rms_mean = float(np.mean(rms)) if len(rms) > 0 else 0.0
    rms_std = float(np.std(rms)) if len(rms) > 0 else 0.0
    onset_mean = float(np.mean(onset_env)) if len(onset_env) > 0 else 0.0
    onset_std = float(np.std(onset_env)) if len(onset_env) > 0 else 0.0
    music_ratio = float(np.mean(np.abs(y_harm))) / (float(np.mean(np.abs(y))) + 1e-8)

    return {
        "duration_sec": float(duration_sec),
        "voiced_ratio": voiced_ratio,
        "harmonic_ratio": float(harmonic_ratio),
        "music_ratio": music_ratio,
        "flatness_mean": float(np.mean(flatness)) if len(flatness) > 0 else 1.0,
        "zcr_mean": float(np.mean(zcr)) if len(zcr) > 0 else 1.0,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "rms_cv": float(rms_std / (rms_mean + 1e-8)),
        "onset_mean": onset_mean,
        "onset_std": onset_std,
        "spectral_contrast_mean": float(np.mean(contrast)) if contrast.size else 0.0,
        "centroid_mean": float(np.mean(centroid)) if len(centroid) > 0 else 0.0,
        "rolloff_mean": float(np.mean(rolloff)) if len(rolloff) > 0 else 0.0,
    }


def looks_like_clean_speech(feats: Dict[str, float]) -> Tuple[bool, List[str]]:
    reasons = []
    if feats["duration_sec"] < MIN_SEGMENT_SEC:
        reasons.append("too_short")
    if feats["voiced_ratio"] < MIN_VOICED_RATIO:
        reasons.append("low_voiced_ratio")
    if feats["harmonic_ratio"] < MIN_HARMONIC_RATIO:
        reasons.append("low_harmonic_ratio")
    if feats["flatness_mean"] > MAX_FLATNESS:
        reasons.append("too_flat_noise_like")
    if feats["zcr_mean"] > MAX_ZCR:
        reasons.append("too_high_zcr")
    if feats.get("rms_mean", 0.0) < MIN_RMS_MEAN:
        reasons.append("too_quiet")
    if feats.get("centroid_mean", 0.0) > MAX_CENTROID_MEAN:
        reasons.append("too_bright_noise_like")
    return len(reasons) == 0, reasons


def music_scene_rejection_gate(feats: Dict[str, float], transcript: str = "") -> Tuple[bool, List[str]]:
    """Reject song/live/crowd/music-dominant clips; keep normal/rehearsal dialogue."""
    reasons = []
    duration = float(feats.get("duration_sec", 0.0))
    voiced = float(feats.get("voiced_ratio", 0.0))
    harmonic = float(feats.get("harmonic_ratio", 0.0))
    music_ratio = float(feats.get("music_ratio", harmonic))
    flatness = float(feats.get("flatness_mean", 0.0))
    zcr = float(feats.get("zcr_mean", 0.0))
    centroid = float(feats.get("centroid_mean", 0.0))
    rolloff = float(feats.get("rolloff_mean", 0.0))
    rms_cv = float(feats.get("rms_cv", 0.0))
    onset_mean = float(feats.get("onset_mean", 0.0))
    contrast = float(feats.get("spectral_contrast_mean", 0.0))
    text = str(transcript or "")

    lyric_like = bool(re.search(r"(らら|ラー|ララ|歌|唄|music|song|live|ライブ|オープニング|エンディング)", text, re.IGNORECASE))
    sustained_music = duration >= 2.5 and music_ratio > MAX_DIALOGUE_MUSIC_RATIO and harmonic > 0.52 and flatness < 0.18
    rhythmic_music = duration >= 2.0 and onset_mean > 1.05 and contrast > 18.0 and rms_cv < 1.15
    bright_band_mix = duration >= 1.5 and rolloff > 5200.0 and centroid > 2400.0 and harmonic > 0.45 and voiced > 0.55
    crowd_noise = flatness > 0.36 and zcr > 0.16 and voiced < 0.45

    if sustained_music and lyric_like:
        reasons.append("song_or_lyric_scene")
    elif sustained_music and rhythmic_music:
        reasons.append("music_dominant_scene")
    elif bright_band_mix and rhythmic_music:
        reasons.append("band_or_live_music_scene")
    if crowd_noise:
        reasons.append("crowd_or_noise_scene")

    return bool(reasons), reasons


def has_strong_japanese_speech_signal(result: Dict[str, Optional[float]], feats: Dict[str, float]) -> bool:
    text = (result.get("text") or "").strip()
    lang = result.get("lang")
    prob = float(result.get("prob") or 0.0)
    stats = text_char_stats(text)
    total = max(1, int(stats["total"]))
    jp_ratio = float(stats["japanese"]) / total

    if lang != TARGET_LANGUAGE or prob < 0.85 or len(text) < 3 or int(stats["japanese"]) < MIN_JAPANESE_CHAR_COUNT:
        return False
    if jp_ratio < 0.55:
        return False

    if feats["duration_sec"] < 0.45:
        return False
    if feats["voiced_ratio"] < 0.22:
        return False
    if feats["harmonic_ratio"] < 0.16:
        return False
    if feats.get("rms_mean", 0.0) < MIN_RMS_MEAN:
        return False
    if feats.get("centroid_mean", 0.0) > MAX_CENTROID_MEAN:
        return False

    return True


def load_segmentation_whisper(model_name: str) -> WhisperModel:
    global SEGMENTATION_RUNTIME

    attempts = [("cuda", "float16")]
    last_error = None

    for device, compute_type in attempts:
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
            SEGMENTATION_RUNTIME = {
                "device": device,
                "compute_type": compute_type,
            }
            print(f"Segmentation Whisper runtime: {device}/{compute_type}")
            return model
        except Exception as e:
            last_error = e
            raise RuntimeError(
                "Segmentation Whisper could not start on GPU. "
                "This fast pipeline is now GPU-only. "
                f"Reason: {type(e).__name__}: {e}"
            ) from e

    raise RuntimeError(f"Unable to load segmentation Whisper model on GPU: {last_error}")


def transcribe_audio_segments(
    model: WhisperModel,
    audio_wav: str,
    batch_size: int,
) -> List[Dict[str, Optional[float]]]:
    runner = BatchedInferencePipeline(model=model)
    segments, info = runner.transcribe(
        audio_wav,
        beam_size=WHISPER_SEGMENT_BEAM_SIZE,
        batch_size=batch_size,
        task="transcribe",
        vad_filter=True,
        condition_on_previous_text=False,
    )

    language = getattr(info, "language", TARGET_LANGUAGE) or TARGET_LANGUAGE
    language_prob = float(getattr(info, "language_probability", 1.0) or 1.0)
    results: List[Dict[str, Optional[float]]] = []

    for segment in segments:
        start = float(getattr(segment, "start", 0.0) or 0.0)
        end = float(getattr(segment, "end", 0.0) or 0.0)
        if end <= start:
            continue

        results.append({
            "start_sec": start,
            "end_sec": end,
            "duration_sec": end - start,
            "lang": language,
            "prob": language_prob,
            "text": segment.text.strip(),
            "avg_logprob": None if getattr(segment, "avg_logprob", None) is None else float(segment.avg_logprob),
            "no_speech_prob": None if getattr(segment, "no_speech_prob", None) is None else float(segment.no_speech_prob),
        })

    return results


def sentence_end(text: str) -> bool:
    stripped = str(text or "").strip()
    return bool(stripped) and stripped[-1] in {"。", "！", "？", "!", "?", "."}


def split_whisperx_segments_by_words(segments: List[Dict[str, object]]) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for segment in segments:
        words = [
            word for word in segment.get("words", [])
            if word.get("start") is not None and word.get("end") is not None
        ]
        if not words:
            if segment.get("start") is not None and segment.get("end") is not None:
                results.append(segment)
            continue

        current_words = []
        current_start = None
        current_end = None
        for word in words:
            text = str(word.get("word") or "")
            if current_start is None:
                current_start = float(word["start"])
            current_end = float(word["end"])
            current_words.append(text)

            duration = current_end - current_start
            if sentence_end(text) or duration >= TARGET_CHUNK_SEC:
                joined = "".join(current_words).strip()
                if joined and duration >= MIN_CHUNK_SEC:
                    results.append({
                        "start": current_start,
                        "end": current_end,
                        "text": joined,
                    })
                current_words = []
                current_start = None
                current_end = None

        if current_words and current_start is not None and current_end is not None:
            joined = "".join(current_words).strip()
            if joined and current_end - current_start >= MIN_CHUNK_SEC:
                results.append({
                    "start": current_start,
                    "end": current_end,
                    "text": joined,
                })

    return results


def transcribe_whisperx_segments(
    audio_wav: str,
    model_name: str = WHISPERX_MODEL,
    batch_size: int = WHISPER_SEGMENT_BATCH_SIZE,
) -> List[Dict[str, Optional[float]]]:
    try:
        import whisperx
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "WhisperX is not installed. Install it with: "
            f"\"{sys.executable}\" -m pip install whisperx"
        ) from e

    import torch

    device = preferred_torch_device()
    compute_type = "float16" if device == "cuda" else "float32"
    print(f"WhisperX transcription runtime: {device}/{compute_type}")
    model = None
    align_model = None
    effective_batch_size = min(batch_size, 4) if device == "cuda" else batch_size
    try:
        model = whisperx.load_model(
            model_name,
            device,
            compute_type=compute_type,
            language=TARGET_LANGUAGE,
            vad_method="silero",
        )
        result = model.transcribe(
            audio_wav,
            batch_size=effective_batch_size,
            language=TARGET_LANGUAGE,
        )
        segments = result.get("segments", [])
    except RuntimeError as e:
        is_cuda_oom = "out of memory" in str(e).lower() and device == "cuda"
        if is_cuda_oom and model_name != "tiny":
            if model is not None:
                del model
                model = None
            torch.cuda.empty_cache()
            print("WhisperX ran out of GPU memory; retrying with tiny model and smaller batch.")
            return transcribe_whisperx_segments(audio_wav, model_name="tiny", batch_size=2)
        raise
    finally:
        if model is not None:
            del model
        if device == "cuda":
            torch.cuda.empty_cache()

    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=TARGET_LANGUAGE,
            device=device,
        )
        aligned = whisperx.align(
            segments,
            align_model,
            metadata,
            audio_wav,
            device,
            return_char_alignments=False,
        )
        segments = split_whisperx_segments_by_words(aligned.get("segments", segments))
    except Exception as e:
        print(f"WhisperX alignment unavailable, using WhisperX segment timestamps. Reason: {type(e).__name__}: {e}")
    finally:
        if align_model is not None:
            del align_model
        if device == "cuda":
            torch.cuda.empty_cache()

    out: List[Dict[str, Optional[float]]] = []
    for segment in segments:
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", 0.0) or 0.0)
        text = str(segment.get("text") or "").strip()
        if end <= start:
            continue
        out.append({
            "start_sec": start,
            "end_sec": end,
            "duration_sec": end - start,
            "lang": TARGET_LANGUAGE,
            "prob": 1.0,
            "text": text,
            "avg_logprob": None,
            "no_speech_prob": None,
        })
    return out


def transcribe_chunk_metadata(
    model: WhisperModel,
    chunk_wav: str,
    batch_size: int = 1,
) -> Dict[str, Optional[float]]:
    parts = transcribe_audio_segments(model, chunk_wav, batch_size=batch_size)
    if not parts:
        return {
            "lang": TARGET_LANGUAGE,
            "prob": 0.0,
            "text": "",
            "avg_logprob": None,
            "no_speech_prob": None,
        }

    texts = [str(part.get("text") or "").strip() for part in parts if str(part.get("text") or "").strip()]
    avg_logprobs = [float(part["avg_logprob"]) for part in parts if part.get("avg_logprob") is not None]
    no_speech_probs = [float(part["no_speech_prob"]) for part in parts if part.get("no_speech_prob") is not None]
    probs = [float(part.get("prob") or 0.0) for part in parts]

    return {
        "lang": parts[0].get("lang") or TARGET_LANGUAGE,
        "prob": float(np.mean(probs)) if probs else 0.0,
        "text": " ".join(texts).strip(),
        "avg_logprob": float(np.mean(avg_logprobs)) if avg_logprobs else None,
        "no_speech_prob": float(np.mean(no_speech_probs)) if no_speech_probs else None,
    }


def split_long_clip_ranges(clip: AudioSegment, cut_mode: str = "sentence") -> List[Tuple[AudioSegment, int]]:
    duration_sec = len(clip) / 1000.0
    if duration_sec <= MAX_SEGMENT_SEC:
        return [(clip, 0)]

    silence_thresh = -40.0
    if clip.dBFS != float("-inf"):
        silence_thresh = float(clip.dBFS - SILENCE_THRESH_OFFSET_DB)

    nonsilent_ranges = detect_nonsilent(
        clip,
        min_silence_len=SILENCE_MIN_MS,
        silence_thresh=silence_thresh,
        seek_step=5,
    )

    chunks: List[Tuple[AudioSegment, int]] = []
    for start_ms, end_ms in nonsilent_ranges:
        start_ms = max(0, int(start_ms) - KEEP_SILENCE_MS)
        end_ms = min(len(clip), int(end_ms) + KEEP_SILENCE_MS)
        part = clip[start_ms:end_ms]
        if len(part) / 1000.0 < MIN_CHUNK_SEC:
            continue
        if len(part) / 1000.0 <= MAX_SEGMENT_SEC:
            chunks.append((part, start_ms))
            continue

        step_ms = int(TARGET_CHUNK_SEC * 1000)
        for rel_start_ms in range(0, len(part), step_ms):
            subpart = part[rel_start_ms:rel_start_ms + step_ms]
            if len(subpart) / 1000.0 < MIN_CHUNK_SEC:
                continue
            chunks.append((subpart, start_ms + rel_start_ms))

    if cut_mode == "silence":
        return chunks or [(clip[: int(TARGET_CHUNK_SEC * 1000)], 0)]

    if not chunks:
        step_ms = int(TARGET_CHUNK_SEC * 1000)
        for start_ms in range(0, len(clip), step_ms):
            part = clip[start_ms:start_ms + step_ms]
            if len(part) / 1000.0 >= MIN_CHUNK_SEC:
                chunks.append((part, start_ms))

    return chunks or [(clip[: int(TARGET_CHUNK_SEC * 1000)], 0)]


def is_character_vocal_effect(result: Dict[str, Optional[float]]) -> bool:
    text = (result.get("text") or "").strip().lower()
    lang = result.get("lang") or ""
    prob = float(result.get("prob") or 0.0)

    vocal_words = {
        "ah", "oh", "eh", "ugh", "huh", "haha", "ha", "aa", "aaa",
        "mm", "mmm", "hn", "hah", "ow", "wah", "woah", "whoa",
        "ehh", "ahh", "ohh", "ughh", "yah", "yeah", "hahaa",
    }
    jp_short_vocalizations = {
        "あ", "ああ", "あっ", "え", "えっ", "お", "おっ", "うっ",
        "はっ", "ふっ", "くっ", "ん", "んっ", "うわ", "うわっ",
        "わっ", "きゃ", "きゃっ",
    }

    if not text:
        return False

    cleaned = (
        text.replace("!", "")
        .replace("?", "")
        .replace(".", "")
        .replace(",", "")
        .strip()
    )
    words = cleaned.split()

    if cleaned in vocal_words:
        return True
    if cleaned in jp_short_vocalizations:
        return True
    if len(words) <= 2 and all(word in vocal_words for word in words):
        return True
    if len(cleaned) <= 4 and prob < 0.80:
        return True
    if lang != TARGET_LANGUAGE and prob < 0.70 and len(cleaned) <= 12:
        return True
    return False


def is_character_vocal_effect_strict(result: Dict[str, Optional[float]]) -> bool:
    text = (result.get("text") or "").strip().lower()
    lang = result.get("lang") or ""
    prob = float(result.get("prob") or 0.0)

    vocal_words = {
        "ah", "oh", "eh", "ugh", "huh", "haha", "ha", "aa", "aaa",
        "mm", "mmm", "hn", "hah", "ow", "wah", "woah", "whoa",
        "ehh", "ahh", "ohh", "ughh", "yah", "yeah", "hahaa",
    }
    jp_short_vocalizations = {
        "あ", "ああ", "あっ", "え", "えっ", "お", "おっ", "うっ",
        "はっ", "ふっ", "くっ", "ん", "んっ", "うわ", "うわっ",
        "わっ", "きゃ", "きゃっ", "はぁ", "へぇ", "おぉ", "うー",
    }

    if not text:
        return False

    cleaned = (
        text.replace("!", "")
        .replace("?", "")
        .replace("！", "")
        .replace("？", "")
        .replace(".", "")
        .replace(",", "")
        .replace("。", "")
        .replace("、", "")
        .strip()
    )
    words = cleaned.split()

    if cleaned in vocal_words or cleaned in jp_short_vocalizations:
        return True
    if lang == TARGET_LANGUAGE and len(cleaned) <= 4 and cleaned.endswith(("っ", "ー")):
        return True
    if len(words) <= 2 and all(word in vocal_words for word in words):
        return True
    if len(cleaned) <= 4 and prob < 0.80:
        return True
    if lang != TARGET_LANGUAGE and prob < 0.70 and len(cleaned) <= 12:
        return True
    return False


def reset_pipeline_dirs() -> None:
    ensure_dir(PIPELINE_WORKSPACE)
    ensure_dir(EXTRACTED_ROOT)
    ensure_dir(SEGMENT_ROOT)
    reset_wav_folder(JP_DIR)
    reset_wav_folder(VOCAL_EFFECT_DIR)
    reset_wav_folder(OTHER_DIR)
    reset_wav_folder(TEMP_DIR)
    if os.path.isdir(SPEAKER_DIR):
        shutil.rmtree(SPEAKER_DIR)
    ensure_dir(SPEAKER_DIR)
    reset_wav_folder(MIXED_SPEAKER_DIR)
    if os.path.isdir(CLASSIFIED_DIR):
        shutil.rmtree(CLASSIFIED_DIR)
    ensure_classifier_output_tree()
    for name in os.listdir(EXTRACTED_ROOT):
        full_path = os.path.join(EXTRACTED_ROOT, name)
        if os.path.isfile(full_path) and name.lower().endswith(".wav"):
            os.remove(full_path)


def write_manifest(folder: str, rows: List[Dict[str, object]], append: bool = False) -> None:
    manifest_path = os.path.join(folder, "_segments_manifest.csv")
    mode = "a" if append and os.path.isfile(manifest_path) and os.path.getsize(manifest_path) > 0 else "w"
    with open(manifest_path, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "source_media",
                "start_sec",
                "end_sec",
                "duration_sec",
                "lang",
                "language_prob",
                "japanese_char_ratio",
                "transcript",
                "avg_logprob",
                "no_speech_prob",
                "speech_gate",
                "speech_gate_reasons",
                "bucket_reason",
                "speaker",
                "character",
                "character_score",
                "character_margin",
                "speaker_purity",
                "other_speaker_overlap_ratio",
                "speaker_assignment_pass",
            ],
        )
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def load_speaker_turns(
    audio_wav: str,
    hf_token: Optional[str] = None,
    diarization_model: str = PYANNOTE_DIARIZATION_MODEL,
) -> List[Dict[str, object]]:
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        for token_path in (HF_TOKEN_FILE, HF_TOKEN_FILE_TXT):
            if os.path.isfile(token_path):
                with open(token_path, "r", encoding="utf-8-sig") as f:
                    content = f.read().strip()
                match = re.search(r"hf_[A-Za-z0-9_\-]+", content)
                token = match.group(0) if match else content
                break
    if not token:
        raise RuntimeError(
            "Speaker diarization needs a Hugging Face token. "
            "Set HF_TOKEN or pass --hf-token after accepting the pyannote model agreement."
        )

    try:
        import torchaudio

        # pyannote.audio currently still calls torchaudio backend helpers that
        # were removed from newer torchaudio releases. The pipeline receives a
        # preloaded waveform below, so these compatibility stubs are enough.
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: []
        if not hasattr(torchaudio, "get_audio_backend"):
            torchaudio.get_audio_backend = lambda: None
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda backend=None: None

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"\s*torchcodec is not installed correctly.*",
                category=UserWarning,
            )
            from pyannote.audio import Pipeline
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "pyannote.audio is not installed. Install it in this Python environment with: "
            f"\"{sys.executable}\" -m pip install pyannote.audio"
        ) from e

    try:
        pipeline = Pipeline.from_pretrained(diarization_model, token=token)
    except Exception as e:
        raise RuntimeError(
            f"Could not download/load {diarization_model}. "
            "Use a valid Hugging Face read token from an account that has accepted the model conditions. "
            f"Open https://huggingface.co/{diarization_model} and accept/request access, then create a new token. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e
    if pipeline is None:
        raise RuntimeError(
            f"Could not load {diarization_model}. "
            "Check that your Hugging Face token is valid and you accepted that model's user agreement."
        )

    import torch
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    import math

    sample_rate, data = wavfile.read(audio_wav)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        scale = max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / float(scale)
    else:
        data = data.astype(np.float32)
    if sample_rate != TARGET_SR:
        gcd = math.gcd(int(sample_rate), int(TARGET_SR))
        data = resample_poly(data, TARGET_SR // gcd, sample_rate // gcd).astype(np.float32)
        sample_rate = TARGET_SR
    waveform = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0)
    device = preferred_torch_device()
    try:
        pipeline.to(torch.device(device))
    except Exception as e:
        print(f"Warning: pyannote pipeline could not move to {device}; using default device. Reason: {type(e).__name__}: {e}")
    waveform = waveform.to(device)
    print(f"pyannote diarization runtime: {device}")

    diarization = None
    try:
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
        turns = iter_diarization_tracks(diarization)
    finally:
        if diarization is not None:
            del diarization
        del pipeline
        del waveform
        if device == "cuda":
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    return turns


def iter_diarization_tracks(diarization) -> List[Dict[str, object]]:
    if hasattr(diarization, "itertracks"):
        return [
            {
                "start_sec": float(turn.start),
                "end_sec": float(turn.end),
                "speaker": str(speaker),
            }
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

    annotation = getattr(diarization, "speaker_diarization", None)
    if annotation is not None and hasattr(annotation, "itertracks"):
        return [
            {
                "start_sec": float(turn.start),
                "end_sec": float(turn.end),
                "speaker": str(speaker),
            }
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]

    if hasattr(diarization, "serialize"):
        serialized = diarization.serialize()
        rows = serialized.get("diarization") or serialized.get("exclusive_diarization") or []
        return [
            {
                "start_sec": float(row["start"]),
                "end_sec": float(row["end"]),
                "speaker": str(row["speaker"]),
            }
            for row in rows
            if {"start", "end", "speaker"}.issubset(row)
        ]

    raise RuntimeError(f"Unsupported diarization output type: {type(diarization).__name__}")


def speaker_for_span(start_sec: float, end_sec: float, speaker_turns: List[Dict[str, object]]) -> str:
    if not speaker_turns:
        return "UNKNOWN_SPEAKER"

    best_speaker = "UNKNOWN_SPEAKER"
    best_overlap = 0.0
    for turn in speaker_turns:
        overlap = min(end_sec, float(turn["end_sec"])) - max(start_sec, float(turn["start_sec"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = str(turn["speaker"])
    return best_speaker if best_overlap > 0.0 else "UNKNOWN_SPEAKER"


def split_span_on_speaker_changes(
    start_sec: float,
    end_sec: float,
    speaker_turns: List[Dict[str, object]],
    min_duration_sec: float = MIN_SEGMENT_SEC,
    pad_sec: float = 0.015,
    merge_gap_sec: float = 0.08,
) -> List[Dict[str, object]]:
    if not speaker_turns or end_sec <= start_sec:
        return [{
            "start_sec": start_sec,
            "end_sec": end_sec,
            "speaker": "UNKNOWN_SPEAKER",
        }]

    spans: List[Dict[str, object]] = []
    for turn in sorted(speaker_turns, key=lambda item: (float(item["start_sec"]), float(item["end_sec"]))):
        overlap_start = max(start_sec, float(turn["start_sec"]))
        overlap_end = min(end_sec, float(turn["end_sec"]))
        if overlap_end - overlap_start < min_duration_sec:
            continue

        clean_start = max(start_sec, overlap_start - pad_sec)
        clean_end = min(end_sec, overlap_end + pad_sec)
        speaker = str(turn["speaker"])

        if (
            spans
            and spans[-1]["speaker"] == speaker
            and clean_start - float(spans[-1]["end_sec"]) <= merge_gap_sec
        ):
            spans[-1]["end_sec"] = max(float(spans[-1]["end_sec"]), clean_end)
            continue

        spans.append({
            "start_sec": clean_start,
            "end_sec": clean_end,
            "speaker": speaker,
        })

    if not spans:
        return []
    return spans


def dominant_speaker_clip_span(
    start_sec: float,
    end_sec: float,
    speaker_turns: List[Dict[str, object]],
    min_purity: float = 0.82,
    max_other_overlap: float = 0.12,
    pad_sec: float = 0.03,
) -> Optional[Dict[str, object]]:
    if not speaker_turns or end_sec <= start_sec:
        return {
            "speaker": "UNKNOWN_SPEAKER",
            "start_sec": start_sec,
            "end_sec": end_sec,
            "purity": 1.0,
            "other_overlap_ratio": 0.0,
        }

    overlaps: Dict[str, float] = {}
    speaker_bounds: Dict[str, List[float]] = {}
    for turn in speaker_turns:
        overlap_start = max(start_sec, float(turn["start_sec"]))
        overlap_end = min(end_sec, float(turn["end_sec"]))
        overlap = overlap_end - overlap_start
        if overlap <= 0.0:
            continue
        speaker = str(turn["speaker"])
        overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap
        if speaker not in speaker_bounds:
            speaker_bounds[speaker] = [overlap_start, overlap_end]
        else:
            speaker_bounds[speaker][0] = min(speaker_bounds[speaker][0], overlap_start)
            speaker_bounds[speaker][1] = max(speaker_bounds[speaker][1], overlap_end)

    if not overlaps:
        return None

    ranked = sorted(overlaps.items(), key=lambda item: item[1], reverse=True)
    speaker, best_overlap = ranked[0]
    other_overlap = sum(value for key, value in overlaps.items() if key != speaker)
    clip_duration = max(1e-6, end_sec - start_sec)
    purity = best_overlap / clip_duration
    other_ratio = other_overlap / clip_duration

    if purity < min_purity or other_ratio > max_other_overlap:
        return None

    clean_start, clean_end = speaker_bounds[speaker]
    clean_start = max(start_sec, clean_start - pad_sec)
    clean_end = min(end_sec, clean_end + pad_sec)
    if clean_end - clean_start < MIN_SEGMENT_SEC:
        return None

    return {
        "speaker": speaker,
        "start_sec": clean_start,
        "end_sec": clean_end,
        "purity": purity,
        "other_overlap_ratio": other_ratio,
    }


def dominant_speaker_clip_span_retry(
    start_sec: float,
    end_sec: float,
    speaker_turns: List[Dict[str, object]],
    min_purity: float = 0.80,
    max_other_overlap: float = 0.20,
) -> Tuple[Optional[Dict[str, object]], str]:
    info = dominant_speaker_clip_span(
        start_sec,
        end_sec,
        speaker_turns,
        min_purity=min_purity,
        max_other_overlap=max_other_overlap,
    )
    if info is not None:
        info["speaker_pass"] = "threshold"
        return info, "threshold"
    if end_sec - start_sec >= MIN_SEGMENT_SEC:
        return None, "below_speaker_threshold"
    return None, "mixed_rejected"


def export_speaker_copy(src_wav: str, speaker: str, filename: str) -> None:
    speaker_folder = os.path.join(SPEAKER_DIR, safe_name(speaker, 48))
    ensure_dir(speaker_folder)
    shutil.copy2(src_wav, os.path.join(speaker_folder, filename))


def load_wav_tensor(file_path: str) -> Tuple["torch.Tensor", int]:
    import torch
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    sample_rate, data = wavfile.read(file_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        scale = max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / float(scale)
    else:
        data = data.astype(np.float32)
    if sample_rate != TARGET_SR:
        gcd = math.gcd(int(sample_rate), int(TARGET_SR))
        data = resample_poly(data, TARGET_SR // gcd, sample_rate // gcd).astype(np.float32)
        sample_rate = TARGET_SR
    waveform = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0)
    return waveform, sample_rate


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector
    return vector / norm


def load_character_embedder():
    patch_torchaudio_legacy_backend_api()
    from speechbrain.inference.classifiers import EncoderClassifier

    device = preferred_torch_device()
    embedder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )
    print(f"Reference speaker embedder runtime: {device}")
    return embedder


def compute_speaker_embedding(embedder, wav_path: str) -> np.ndarray:
    import torch

    waveform, _ = load_wav_tensor(wav_path)
    device = preferred_torch_device()
    waveform = waveform.to(device)
    with torch.no_grad():
        embedding = embedder.encode_batch(waveform).detach().cpu().numpy()
    return normalize_embedding(embedding)


def build_character_reference_embeddings(reference_root: Optional[str]) -> Dict[str, np.ndarray]:
    if not reference_root:
        return {}

    root = os.path.abspath(reference_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Reference character folder not found: {root}")

    embedder = load_character_embedder()
    references: Dict[str, np.ndarray] = {}
    for character in sorted(os.listdir(root)):
        character_dir = os.path.join(root, character)
        if not os.path.isdir(character_dir):
            continue

        vectors = []
        for name in sorted(os.listdir(character_dir)):
            if not name.lower().endswith((".wav", ".flac", ".mp3", ".m4a", ".ogg")):
                continue
            path = os.path.join(character_dir, name)
            try:
                vectors.append(compute_speaker_embedding(embedder, path))
            except Exception as e:
                print(f"Reference embedding failed: {path} | {type(e).__name__}: {e}")

        if vectors:
            references[safe_name(character, 64)] = normalize_embedding(np.mean(vectors, axis=0))

    if not references:
        raise RuntimeError(f"No usable reference character audio found under: {root}")

    print("Loaded reference character voices:")
    for character in sorted(references):
        print(f"  - {character}")
    match_character.embedder = embedder
    return references


def match_character(
    wav_path: str,
    references: Dict[str, np.ndarray],
    min_score: float = 0.58,
    min_margin: float = 0.06,
) -> Dict[str, object]:
    if not references:
        return {"character": "", "score": "", "margin": ""}

    vector = compute_speaker_embedding(match_character.embedder, wav_path)
    scores = sorted(
        ((character, float(np.dot(vector, ref))) for character, ref in references.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    best_character, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else -1.0
    margin = best_score - second_score
    if best_score < min_score or margin < min_margin:
        return {
            "character": "Unknown_or_Review",
            "score": round(best_score, 4),
            "margin": round(margin, 4),
        }
    return {
        "character": best_character,
        "score": round(best_score, 4),
        "margin": round(margin, 4),
    }


match_character.embedder = None


def segment_and_filter(
    audio_wav: str,
    max_segments: Optional[int] = None,
    model_name: str = WHISPER_SEGMENT_MODEL,
    batch_size: int = WHISPER_SEGMENT_BATCH_SIZE,
    speaker_diarize: bool = False,
    hf_token: Optional[str] = None,
    diarization_model: str = PYANNOTE_DIARIZATION_MODEL,
    cut_mode: str = "sentence",
    min_emotion_clip_sec: float = MIN_EMOTION_CLIP_SEC,
    reference_characters: Optional[str] = None,
    character_min_score: float = 0.58,
    character_min_margin: float = 0.06,
    speaker_min_purity: float = 0.80,
    speaker_max_other_overlap: float = 0.20,
    transcript_backend: str = "faster-whisper",
    whisperx_model: str = WHISPERX_MODEL,
) -> Dict[str, int]:
    audio = AudioSegment.from_wav(audio_wav)
    character_references = build_character_reference_embeddings(reference_characters)
    speaker_turns: List[Dict[str, object]] = []
    if speaker_diarize:
        print("Running speaker diarization before emotion classification...")
        speaker_turns = load_speaker_turns(
            audio_wav,
            hf_token=hf_token,
            diarization_model=diarization_model,
        )
        speakers = sorted({str(turn["speaker"]) for turn in speaker_turns})
        print(f"Detected speaker clusters: {', '.join(speakers) if speakers else 'none'}")

    whisper = None
    if transcript_backend == "whisperx":
        raw_segments = transcribe_whisperx_segments(
            audio_wav,
            model_name=whisperx_model,
            batch_size=batch_size,
        )
    else:
        whisper = load_segmentation_whisper(model_name)
        raw_segments = transcribe_audio_segments(whisper, audio_wav, batch_size=batch_size)

    jp_rows: List[Dict[str, object]] = []
    vocal_rows: List[Dict[str, object]] = []
    other_rows: List[Dict[str, object]] = []
    mixed_rows: List[Dict[str, object]] = []
    jp_start = sum(1 for name in os.listdir(JP_DIR) if name.lower().endswith(".wav"))
    vocal_start = sum(1 for name in os.listdir(VOCAL_EFFECT_DIR) if name.lower().endswith(".wav"))
    other_start = sum(1 for name in os.listdir(OTHER_DIR) if name.lower().endswith(".wav"))
    jp_count = jp_start
    vocal_count = vocal_start
    other_count = other_start

    if max_segments is not None:
        raw_segments = raw_segments[:max_segments]

    segment_idx = 0
    for raw_idx, segment in enumerate(raw_segments, start=1):
        start = float(segment["start_sec"])
        end = float(segment["end_sec"])
        base_clip = audio[int(start * 1000):int(end * 1000)]
        base_ranges = split_long_clip_ranges(base_clip, cut_mode=cut_mode)
        subclip_ranges = []
        for base_subclip, offset_ms in base_ranges:
            range_start = start + (offset_ms / 1000.0)
            range_end = range_start + (len(base_subclip) / 1000.0)
            if not speaker_diarize:
                subclip_ranges.append((base_subclip, offset_ms, None, None))
                continue

            speaker_split_spans = split_span_on_speaker_changes(
                range_start,
                range_end,
                speaker_turns,
            )
            if not speaker_split_spans:
                subclip_ranges.append((base_subclip, offset_ms, None, None))
                continue

            for span in speaker_split_spans:
                span_start = float(span["start_sec"])
                span_end = float(span["end_sec"])
                if span_end <= span_start:
                    continue
                split_clip = audio[int(span_start * 1000):int(span_end * 1000)]
                split_offset_ms = int(round((span_start - start) * 1000))
                subclip_ranges.append((split_clip, split_offset_ms, span_start, span_end))

        for subclip, offset_ms, forced_start, forced_end in subclip_ranges:
            segment_idx += 1
            if forced_start is None or forced_end is None:
                sub_start = start + (offset_ms / 1000.0)
                sub_end = sub_start + (len(subclip) / 1000.0)
            else:
                sub_start = float(forced_start)
                sub_end = float(forced_end)
            speaker_info, speaker_pass = dominant_speaker_clip_span_retry(
                sub_start,
                sub_end,
                speaker_turns,
                min_purity=speaker_min_purity,
                max_other_overlap=speaker_max_other_overlap,
            )
            if speaker_diarize and speaker_info is None:
                mixed_name = (
                    f"mixed_{segment_idx:04d}_"
                    f"{safe_name(CURRENT_SOURCE_STEM, 32)}_"
                    f"{safe_name(segment.get('text') or 'no_text')}.wav"
                )
                subclip.export(os.path.join(MIXED_SPEAKER_DIR, mixed_name), format="wav")
                mixed_rows.append({
                    "filename": mixed_name,
                    "source_media": CURRENT_SOURCE_STEM,
                    "start_sec": round(sub_start, 3),
                    "end_sec": round(sub_end, 3),
                    "duration_sec": round(sub_end - sub_start, 3),
                    "lang": segment.get("lang"),
                    "language_prob": segment.get("prob"),
                    "japanese_char_ratio": round(japanese_char_ratio(segment.get("text") or ""), 4),
                    "transcript": segment.get("text"),
                    "avg_logprob": segment.get("avg_logprob"),
                    "no_speech_prob": segment.get("no_speech_prob"),
                    "speech_gate": False,
                    "speech_gate_reasons": "mixed_or_low_purity_speaker",
                    "bucket_reason": "mixed_or_low_purity_speaker",
                    "speaker": "MIXED",
                    "character": "Unknown_or_Review",
                    "character_score": "",
                    "character_margin": "",
                    "speaker_purity": "",
                    "other_speaker_overlap_ratio": "",
                })
                continue

            if speaker_diarize and speaker_info is not None:
                clipped_start = float(speaker_info["start_sec"])
                clipped_end = float(speaker_info["end_sec"])
                if clipped_end - clipped_start > MAX_SEGMENT_SEC:
                    clipped_end = clipped_start + MAX_SEGMENT_SEC
                subclip = audio[int(clipped_start * 1000):int(clipped_end * 1000)]
                sub_start = clipped_start
                sub_end = clipped_end

            speaker = str(speaker_info["speaker"]) if speaker_info else speaker_for_span(sub_start, sub_end, speaker_turns)
            speaker_purity = "" if not speaker_info else round(float(speaker_info["purity"]), 4)
            other_speaker_overlap_ratio = "" if not speaker_info else round(float(speaker_info["other_overlap_ratio"]), 4)
            temp_path = os.path.join(TEMP_DIR, f"segment_{raw_idx:04d}_{segment_idx:04d}.wav")
            subclip.export(temp_path, format="wav")

            character_match = match_character(
                temp_path,
                character_references,
                min_score=character_min_score,
                min_margin=character_min_margin,
            )
            character = str(character_match.get("character") or "")
            character_score = character_match.get("score", "")
            character_margin = character_match.get("margin", "")

            feats = clip_features(temp_path)
            speech_ok, speech_reasons = looks_like_clean_speech(feats)
            if len(subclip_ranges) == 1 and not speaker_diarize:
                result = segment
            else:
                if transcript_backend == "whisperx":
                    result = dict(segment)
                    result["start_sec"] = sub_start
                    result["end_sec"] = sub_end
                    result["duration_sec"] = sub_end - sub_start
                else:
                    if whisper is None:
                        whisper = load_segmentation_whisper(model_name)
                    result = transcribe_chunk_metadata(whisper, temp_path, batch_size=1)
                    result["start_sec"] = sub_start
                    result["end_sec"] = sub_end
                    result["duration_sec"] = sub_end - sub_start
            if speaker_diarize:
                result["start_sec"] = sub_start
                result["end_sec"] = sub_end
                result["duration_sec"] = sub_end - sub_start

            text_tag = safe_name(result.get("text") or "no_text")
            row = {
                "source_media": CURRENT_SOURCE_STEM,
                "start_sec": round(float(result["start_sec"]), 3),
                "end_sec": round(float(result["end_sec"]), 3),
                "duration_sec": round(float(result["duration_sec"]), 3),
                "lang": result.get("lang"),
                "language_prob": result.get("prob"),
                "japanese_char_ratio": round(japanese_char_ratio(result.get("text") or ""), 4),
                "transcript": result.get("text"),
                "avg_logprob": result.get("avg_logprob"),
                "no_speech_prob": result.get("no_speech_prob"),
                "speech_gate": speech_ok,
                "speech_gate_reasons": "|".join(speech_reasons),
                "bucket_reason": "",
                "speaker": speaker,
                "character": character,
                "character_score": character_score,
                "character_margin": character_margin,
                "speaker_purity": speaker_purity,
                "other_speaker_overlap_ratio": other_speaker_overlap_ratio,
                "speaker_assignment_pass": speaker_pass,
            }

            transcript = (result.get("text") or "").strip()
            music_reject, music_reasons = music_scene_rejection_gate(feats, transcript)
            jp_text_ok = looks_like_japanese_text(transcript)
            strong_japanese = has_strong_japanese_speech_signal(result, feats)
            long_enough_for_emotion = float(feats.get("duration_sec", 0.0)) >= min_emotion_clip_sec
            is_japanese = (
                (speech_ok or strong_japanese)
                and bool(transcript)
                and long_enough_for_emotion
                and not music_reject
                and jp_text_ok
                and (
                    (
                        result.get("lang") == TARGET_LANGUAGE
                        and float(result.get("prob") or 0.0) >= MIN_LANGUAGE_PROB
                    )
                    or strong_japanese
                )
            )

            if is_japanese:
                jp_count += 1
                filename = (
                    f"jp_{jp_count:04d}_"
                    f"{safe_name(character or speaker, 24)}_"
                    f"{safe_name(speaker, 24)}_"
                    f"{safe_name(CURRENT_SOURCE_STEM, 32)}_"
                    f"{text_tag}.wav"
                )
                row["filename"] = filename
                row["bucket_reason"] = "speech_ok_and_japanese_dialogue" if speech_ok else "strong_japanese_dialogue"
                jp_path = os.path.join(JP_DIR, filename)
                subclip.export(jp_path, format="wav")
                if speaker_diarize:
                    export_speaker_copy(jp_path, speaker, filename)
                jp_rows.append(row)
            elif bool(transcript) and jp_text_ok and not long_enough_for_emotion:
                other_count += 1
                filename = f"other_{other_count:04d}_{safe_name(CURRENT_SOURCE_STEM, 32)}_{text_tag}.wav"
                row["filename"] = filename
                row["bucket_reason"] = f"too_short_for_emotion_training_lt_{min_emotion_clip_sec:.2f}s"
                subclip.export(os.path.join(OTHER_DIR, filename), format="wav")
                other_rows.append(row)
            elif music_reject:
                other_count += 1
                filename = f"other_{other_count:04d}_{safe_name(CURRENT_SOURCE_STEM, 32)}_{text_tag}.wav"
                row["filename"] = filename
                row["speech_gate_reasons"] = "|".join(list(speech_reasons) + list(music_reasons))
                row["bucket_reason"] = "music_or_crowd_rejected:" + "|".join(music_reasons)
                subclip.export(os.path.join(OTHER_DIR, filename), format="wav")
                other_rows.append(row)
            elif is_character_vocal_effect_strict(result):
                vocal_count += 1
                filename = f"vocal_{vocal_count:04d}_{safe_name(CURRENT_SOURCE_STEM, 32)}_{text_tag}.wav"
                row["filename"] = filename
                row["bucket_reason"] = "vocal_effect_or_short_non_dialogue"
                subclip.export(os.path.join(VOCAL_EFFECT_DIR, filename), format="wav")
                vocal_rows.append(row)
            else:
                other_count += 1
                filename = f"other_{other_count:04d}_{safe_name(CURRENT_SOURCE_STEM, 32)}_{text_tag}.wav"
                row["filename"] = filename
                row["bucket_reason"] = "non_dialogue_or_noise"
                subclip.export(os.path.join(OTHER_DIR, filename), format="wav")
                other_rows.append(row)

            if os.path.exists(temp_path):
                os.remove(temp_path)

    write_manifest(JP_DIR, jp_rows, append=True)
    write_manifest(VOCAL_EFFECT_DIR, vocal_rows, append=True)
    write_manifest(OTHER_DIR, other_rows, append=True)
    write_manifest(MIXED_SPEAKER_DIR, mixed_rows, append=True)

    return {
        "raw_segments": len(raw_segments),
        "japanese_dialogue": jp_count - jp_start,
        "character_vocal_effect": vocal_count - vocal_start,
        "other_or_effect": other_count - other_start,
    }


def run_multimodal_classifier(
    input_base: str,
    output_base: str,
    classify_limit: Optional[int] = None,
    emotion_set: str = "7",
    auto_label_policy: str = "balanced",
    openai_classifier_judge: bool = False,
    openai_model: str = "gpt-4.1",
    openai_judge_limit: int = 0,
    openai_timeout_sec: int = 120,
) -> None:
    import multimodal_emotion_classifier as mec

    os.environ["OPENAI_EMOTION_JUDGE"] = "1" if openai_classifier_judge else "0"
    os.environ["OPENAI_EMOTION_MODEL"] = str(openai_model or "gpt-4.1")
    os.environ["OPENAI_EMOTION_JUDGE_LIMIT"] = str(int(openai_judge_limit or 0))
    os.environ["OPENAI_EMOTION_TIMEOUT_SEC"] = str(int(openai_timeout_sec or 120))
    os.environ["EMOTION_SET"] = "7" if str(emotion_set) == "7" else "5"
    os.environ["EMOTION_AUTO_LABEL_POLICY"] = str(auto_label_policy or "balanced").strip().lower()

    mec.INPUT_BASES = [input_base]
    mec.OUTPUT_BASE = output_base
    mec.CFG.setdefault("classification", {})["emotion_set"] = os.environ["EMOTION_SET"]
    mec.CFG.setdefault("classification", {})["auto_label_policy"] = os.environ["EMOTION_AUTO_LABEL_POLICY"]
    mec.CLASSIFY_SOURCE_FOLDERS = {"japanese_dialogue"}
    mec.REJECT_SOURCE_FOLDERS = set()
    mec.COPY_FILES = True
    mec.PRESERVE_ORIGINAL_FILENAME = True
    mec.CFG["models"]["asr"]["device"] = "cuda"
    mec.CFG["models"]["asr"]["compute_type"] = "float16"
    mec.CFG["models"]["audio_emotion"]["device"] = "cuda"

    if classify_limit is None:
        return mec.main()

    original_collect = mec.collect_dialogue_segments

    def limited_collect():
        items = original_collect()
        return items[:classify_limit]

    mec.collect_dialogue_segments = limited_collect
    try:
        return mec.main()
    finally:
        mec.collect_dialogue_segments = original_collect


def load_speaker_name_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}

    map_path = os.path.abspath(path)
    if not os.path.isfile(map_path):
        raise FileNotFoundError(f"Speaker name map not found: {map_path}")

    with open(map_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Speaker name map must be a JSON object, for example {\"SPEAKER_00\": \"Nanoha\"}")

    return {str(k): safe_name(str(v), 64) for k, v in data.items() if str(v).strip()}


def group_classified_by_character(classified_root: str, speaker_name_map_path: Optional[str] = None) -> Dict[str, int]:
    manifest_csv = os.path.join(classified_root, "_multimodal_emotion_manifest.csv")
    if not os.path.isfile(manifest_csv):
        return {}

    speaker_names = load_speaker_name_map(speaker_name_map_path)
    output_root = os.path.join(classified_root, "_by_character_emotion")
    if os.path.isdir(output_root):
        shutil.rmtree(output_root)
    ensure_dir(output_root)

    counts: Dict[str, int] = {}
    with open(manifest_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            speaker_id = (row.get("speaker") or "UNKNOWN_SPEAKER").strip() or "UNKNOWN_SPEAKER"
            manifest_character = (row.get("character") or "").strip()
            character_name = manifest_character or speaker_names.get(speaker_id, speaker_id)
            emotion = (row.get("pred_folder") or "Needs_Review").strip() or "Needs_Review"
            src_path = (row.get("output_file") or row.get("source_file") or "").strip()
            if not src_path or not os.path.isfile(src_path):
                continue

            dst_dir = os.path.join(output_root, safe_name(character_name, 64), safe_name(emotion, 64))
            ensure_dir(dst_dir)
            dst_path = os.path.join(dst_dir, os.path.basename(src_path))
            if os.path.exists(dst_path):
                stem, ext = os.path.splitext(os.path.basename(src_path))
                idx = 1
                while os.path.exists(dst_path):
                    dst_path = os.path.join(dst_dir, f"{stem}_copy_{idx:02d}{ext}")
                    idx += 1
            shutil.copy2(src_path, dst_path)
            key = f"{character_name}/{emotion}"
            counts[key] = counts.get(key, 0) + 1

    if speaker_names:
        map_copy_path = os.path.join(output_root, "_speaker_name_map_used.json")
        with open(map_copy_path, "w", encoding="utf-8") as f:
            json.dump(speaker_names, f, ensure_ascii=False, indent=2)

    return counts


def wav_count(folder: str) -> int:
    if not os.path.isdir(folder):
        return 0
    return sum(1 for name in os.listdir(folder) if name.lower().endswith(".wav"))


def write_batch_summary(batch_rows: List[Dict[str, object]]) -> str:
    ensure_dir(PIPELINE_WORKSPACE_ROOT)
    out_path = os.path.join(PIPELINE_WORKSPACE_ROOT, "_batch_summary.csv")
    fieldnames = [
        "input_media",
        "workspace",
        "audio_wav",
        "raw_segments",
        "japanese_dialogue",
        "character_vocal_effect",
        "other_or_effect",
        "classification_output",
        "high_confidence_count",
        "manual_review_count",
        "high_confidence_csv",
        "manual_review_csv",
        "status",
        "error",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(batch_rows)
    return out_path


def print_final_output_summary(skip_classify: bool) -> None:
    print("\nFinal grouped outputs:")
    print(f"  Dialogue candidates: {JP_DIR}")
    print(f"  Speaker clusters:    {SPEAKER_DIR}")
    print(f"  Mixed rejected:      {MIXED_SPEAKER_DIR}")
    print(f"  Vocal/effects:       {VOCAL_EFFECT_DIR}")
    print(f"  Other/noise:         {OTHER_DIR}")
    print(f"  Emotion root:        {CLASSIFIED_DIR}")
    print(f"  Character emotions:  {os.path.join(CLASSIFIED_DIR, '_by_character_emotion')}")
    if skip_classify:
        print("  Emotion folders prepared, classification not run.")
    else:
        for folder in CLASSIFIER_OUTPUT_FOLDERS:
            print(f"    - {os.path.join(CLASSIFIED_DIR, folder)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified pipeline: extract audio -> speech segmentation -> multimodal emotion classification"
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        help="Input video/audio file or folder. If omitted, all supported media in ./input are processed.",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Optional limit for the number of VAD segments to process. Useful for testing.",
    )
    parser.add_argument(
        "--classify-limit",
        type=int,
        default=None,
        help="Optional limit for the number of speech segments sent to the classifier.",
    )
    parser.add_argument(
        "--skip-classify",
        action="store_true",
        help="Only extract and segment speech candidates, skip multimodal emotion classification.",
    )
    parser.add_argument(
        "--no-reset-output",
        action="store_true",
        help="Append segmentation output to the existing pipeline workspace instead of clearing it first.",
    )
    parser.add_argument(
        "--segment-model",
        default=WHISPER_SEGMENT_MODEL,
        help="Whisper model used only for speech segmentation/transcript gating. Default: %(default)s",
    )
    parser.add_argument(
        "--transcript-backend",
        choices=["faster-whisper", "whisperx"],
        default="faster-whisper",
        help="Transcription/cutting backend. Use whisperx for aligned sentence-like cuts. Default: %(default)s",
    )
    parser.add_argument(
        "--whisperx-model",
        default=WHISPERX_MODEL,
        help="WhisperX ASR model. Default: %(default)s",
    )
    parser.add_argument(
        "--segment-batch-size",
        type=int,
        default=WHISPER_SEGMENT_BATCH_SIZE,
        help="GPU batch size for segmentation Whisper inference. Default: %(default)s",
    )
    parser.add_argument(
        "--speaker-diarize",
        action="store_true",
        help="Cluster clips by speaker before emotion classification using pyannote.audio.",
    )
    parser.add_argument(
        "--diarization-model",
        default=PYANNOTE_DIARIZATION_MODEL,
        help="pyannote diarization model. Default: %(default)s",
    )
    parser.add_argument(
        "--hf-token",
        help="Hugging Face token for pyannote diarization. If omitted, HF_TOKEN or HUGGINGFACE_TOKEN is used.",
    )
    parser.add_argument(
        "--cut-mode",
        choices=["sentence", "silence", "fixed"],
        default="sentence",
        help=(
            "How to split long Whisper utterances: sentence keeps Whisper sentence/utterance boundaries; "
            "silence splits by waveform silence only; fixed allows old duration fallback. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--min-emotion-clip-sec",
        type=float,
        default=MIN_EMOTION_CLIP_SEC,
        help="Minimum clip length kept for emotion classification. Default: %(default)s",
    )
    parser.add_argument(
        "--emotion-set",
        choices=["5", "7"],
        default="7",
        help="Classify with primary 5 emotions or extended 7 emotions. Default: %(default)s",
    )
    parser.add_argument(
        "--auto-label-policy",
        choices=["strict", "balanced", "volume"],
        default="balanced",
        help="Emotion auto-accept policy: strict is cleanest, balanced is default, volume produces more weak labels. Default: %(default)s",
    )
    parser.add_argument(
        "--speaker-name-map",
        help='Optional JSON map from speaker IDs to character names, for example {"SPEAKER_00": "Nanoha"}.',
    )
    parser.add_argument(
        "--reference-characters",
        help="Folder of reference voices: reference_characters/<CharacterName>/*.wav.",
    )
    parser.add_argument(
        "--character-min-score",
        type=float,
        default=0.58,
        help="Minimum cosine score for reference character match. Default: %(default)s",
    )
    parser.add_argument(
        "--character-min-margin",
        type=float,
        default=0.06,
        help="Minimum score gap between best and second-best character. Default: %(default)s",
    )
    parser.add_argument(
        "--speaker-min-purity",
        type=float,
        default=0.80,
        help="Minimum dominant-speaker overlap ratio required to place a clip in a speaker folder. Default: %(default)s",
    )
    parser.add_argument(
        "--speaker-max-other-overlap",
        type=float,
        default=0.20,
        help="Maximum overlap ratio from non-dominant speakers allowed in a speaker clip. Default: %(default)s",
    )
    parser.add_argument(
        "--openai-classifier-judge",
        action="store_true",
        help="Use OpenAI GPT as a judge for uncertain emotion clips during classification.",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4.1",
        help="OpenAI model for uncertain emotion judging. Default: %(default)s",
    )
    parser.add_argument(
        "--openai-judge-limit",
        type=int,
        default=40,
        help="Maximum OpenAI judge calls during classification. Default: %(default)s",
    )
    parser.add_argument(
        "--openai-timeout-sec",
        type=int,
        default=120,
        help="Timeout per OpenAI judge call in seconds. Default: %(default)s",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_paths = resolve_input_paths(args.input_path)
    print(f"Found {len(input_paths)} input media file(s)")
    if args.speaker_diarize:
        print("Pipeline mode: speaker diarization + emotion classification")
    else:
        print("Pipeline mode: normal emotion classification only (speaker/character grouping OFF)")
    if args.no_reset_output:
        ensure_dir(JP_DIR)
        ensure_dir(VOCAL_EFFECT_DIR)
        ensure_dir(OTHER_DIR)
        ensure_dir(TEMP_DIR)
        ensure_dir(MIXED_SPEAKER_DIR)
    else:
        reset_pipeline_dirs()

    batch_rows: List[Dict[str, object]] = []
    for run_index, input_path in enumerate(input_paths, start=1):
        configure_run_workspace(input_path, run_index, len(input_paths))
        print(f"\n[{run_index}/{len(input_paths)}] Input media: {input_path}")
        print(f"Output root: {PIPELINE_WORKSPACE}")

        batch_row: Dict[str, object] = {
            "input_media": input_path,
            "workspace": PIPELINE_WORKSPACE,
            "audio_wav": "",
            "raw_segments": 0,
            "japanese_dialogue": 0,
            "character_vocal_effect": 0,
            "other_or_effect": 0,
            "classification_output": "",
            "high_confidence_count": 0,
            "manual_review_count": 0,
            "high_confidence_csv": "",
            "manual_review_csv": "",
            "status": "pending",
            "error": "",
        }

        try:
            extracted_audio = extract_audio(input_path, EXTRACTED_AUDIO_PATH)
            print(f"Extracted audio temp: {extracted_audio}")

            segment_counts = segment_and_filter(
                extracted_audio,
                max_segments=args.max_segments,
                model_name=args.segment_model,
                batch_size=args.segment_batch_size,
                speaker_diarize=args.speaker_diarize,
                hf_token=args.hf_token,
                diarization_model=args.diarization_model,
                cut_mode=args.cut_mode,
                min_emotion_clip_sec=args.min_emotion_clip_sec,
                reference_characters=args.reference_characters,
                character_min_score=args.character_min_score,
                character_min_margin=args.character_min_margin,
                speaker_min_purity=args.speaker_min_purity,
                speaker_max_other_overlap=args.speaker_max_other_overlap,
                transcript_backend=args.transcript_backend,
                whisperx_model=args.whisperx_model,
            )
            batch_row.update(segment_counts)
            print(f"Segmentation summary: {segment_counts}")
            print(f"Speech candidates: {JP_DIR}")
            if args.speaker_diarize:
                print(f"Speaker grouped clips: {SPEAKER_DIR}")
            print(f"Rejected vocal/effect clips, not character labels: {VOCAL_EFFECT_DIR}")
            print(f"Rejected other/noise: {OTHER_DIR}")
            batch_row["status"] = "segmented"
            if os.path.exists(extracted_audio):
                os.remove(extracted_audio)

        except Exception as e:
            batch_row["status"] = "failed"
            batch_row["error"] = f"{type(e).__name__}: {e}"
            print(f"Failed: {input_path} | {batch_row['error']}")
        finally:
            batch_rows.append(batch_row)

    if args.skip_classify:
        ensure_classifier_output_tree()
        for batch_row in batch_rows:
            if batch_row["status"] == "segmented":
                batch_row["status"] = "segmented_only"
        print("Classification skipped.")
    elif wav_count(JP_DIR) == 0:
        ensure_classifier_output_tree()
        print("Classification skipped because no Japanese dialogue clips were created.")
    else:
        ensure_classifier_output_tree()
        classify_summary = run_multimodal_classifier(
            JP_DIR,
            CLASSIFIED_DIR,
                classify_limit=args.classify_limit,
                emotion_set=args.emotion_set,
                auto_label_policy=args.auto_label_policy,
                openai_classifier_judge=args.openai_classifier_judge,
            openai_model=args.openai_model,
            openai_judge_limit=args.openai_judge_limit,
            openai_timeout_sec=args.openai_timeout_sec,
        ) or {}
        character_counts = {}
        if args.speaker_diarize:
            character_counts = group_classified_by_character(
                CLASSIFIED_DIR,
                speaker_name_map_path=args.speaker_name_map,
            )
        print(f"\nClassification output: {CLASSIFIED_DIR}")
        if args.speaker_diarize:
            print(f"Character/emotion output: {os.path.join(CLASSIFIED_DIR, '_by_character_emotion')}")
            for key in sorted(character_counts):
                print(f"  {key}: {character_counts[key]}")
        for batch_row in batch_rows:
            if batch_row["status"] == "segmented":
                batch_row["classification_output"] = CLASSIFIED_DIR
                batch_row["high_confidence_count"] = classify_summary.get("high_confidence_count", 0)
                batch_row["manual_review_count"] = classify_summary.get("manual_review_count", 0)
                batch_row["high_confidence_csv"] = classify_summary.get("high_confidence_csv", "")
                batch_row["manual_review_csv"] = classify_summary.get("manual_review_csv", "")
                batch_row["status"] = "completed"

    batch_summary = write_batch_summary(batch_rows)
    print_final_output_summary(args.skip_classify)
    print(f"\nBatch summary saved: {batch_summary}")
    failed_count = sum(1 for row in batch_rows if row["status"] == "failed")
    if failed_count:
        print(f"Completed with {failed_count} failed input(s).")
    if failed_count == len(batch_rows) or (not args.skip_classify and wav_count(JP_DIR) == 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
