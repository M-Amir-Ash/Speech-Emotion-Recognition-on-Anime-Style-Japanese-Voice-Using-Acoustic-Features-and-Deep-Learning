import argparse
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import shutil
from pathlib import Path
from typing import List, Optional, Set


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_SCRIPT = os.path.join(BASE_DIR, "extract_segment_multimodal_pipeline.py")
RELABEL_SCRIPT = os.path.join(BASE_DIR, "iterative_emotion_relabel.py")
REVIEW_SCRIPT = os.path.join(BASE_DIR, "manual_review_helper.py")
CLUSTER_PLOT_SCRIPT = os.path.join(BASE_DIR, "plot_emotion_clusters.py")
CHARACTER_CLUSTER_ANALYSIS_SCRIPT = os.path.join(BASE_DIR, "character_cluster_analysis.py")
CLUSTER_CLEAN_TARGET_SCRIPT = os.path.join(BASE_DIR, "cluster_clean_target_selector.py")
SPEAKER_CLUSTER_SCRIPT = os.path.join(BASE_DIR, "speaker_embedding_cluster.py")
CHARACTER_MAP_SCRIPT = os.path.join(BASE_DIR, "character_cluster_mapper.py")
CHARACTER_EMOTION_SCRIPT = os.path.join(BASE_DIR, "character_emotion_pseudolabel.py")
TEXT_GUIDED_EMOTION_SCRIPT = os.path.join(BASE_DIR, "text_guided_emotion_classifier.py")
TRAIN_EMBEDDING_SCRIPT = os.path.join(BASE_DIR, "train_emotion_embedding_classifier.py")
SELF_TRAIN_SCRIPT = os.path.join(BASE_DIR, "self_train_emotion.py")
FINAL_DATASET_SCRIPT = os.path.join(BASE_DIR, "build_final_emotion_dataset.py")
PIPELINE_WORKSPACE = os.path.join(BASE_DIR, "output", "pipeline_workspace")
SEGMENTED_DIALOGUE_DIR = os.path.join(PIPELINE_WORKSPACE, "segmented", "japanese_dialogue")
CLASSIFIED_ROOT = os.path.join(PIPELINE_WORKSPACE, "multimodal_emotion_classified")
SEGMENTED_BY_SPEAKER_DIR = os.path.join(PIPELINE_WORKSPACE, "segmented", "by_speaker")
INPUT_DIR = os.path.join(BASE_DIR, "input")
DEFAULT_RAW_INPUT_DIR = Path(
    r"D:\Telkom University Kuliah\Kuliah S1\Tugas S1\Semester 6\Proposal\Training Data\Auto Split\input"
)
EXTRACTED_AUDIO_DIR = os.path.join(BASE_DIR, "output", "extracted_audio")
HF_TOKEN_FILE = os.path.join(BASE_DIR, "TokenKeyHuggyFace")
HF_TOKEN_FILE_TXT = os.path.join(BASE_DIR, "TokenKeyHuggyFace.txt")
OPENAI_KEY_FILES = [
    os.path.join(BASE_DIR, "OpenAIKey"),
    os.path.join(BASE_DIR, "OpenAIKey.txt"),
    os.path.join(BASE_DIR, "TokenKeyOpenAI"),
    os.path.join(BASE_DIR, "TokenKeyOpenAI.txt"),
]
OPTION8_PROGRESS_FILE = os.path.join(PIPELINE_WORKSPACE, "_option8_processed_files.txt")
TARGET_SR = 16000

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
AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
}
RAW_MEDIA_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
}


def run_python_script(script_path: str, extra_args: List[str]) -> int:
    cmd = [sys.executable, script_path] + list(extra_args)
    return subprocess.call(cmd, cwd=BASE_DIR)


def safe_name(text: str, max_len: int = 80) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("_", "-", " ") else "_" for c in text)
    cleaned = "_".join(cleaned.split()).strip("_")
    return (cleaned or "audio")[:max_len]


def unique_path(folder: str, filename: str) -> str:
    os.makedirs(folder, exist_ok=True)
    candidate = os.path.join(folder, filename)
    if not os.path.exists(candidate):
        return candidate

    stem, ext = os.path.splitext(filename)
    idx = 1
    while True:
        candidate = os.path.join(folder, f"{stem}_copy_{idx:02d}{ext}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def iter_files_by_extension(path: str, extensions: Set[str]) -> List[str]:
    if not path:
        return []

    abs_path = os.path.abspath(path)
    if os.path.isfile(abs_path):
        ext = os.path.splitext(abs_path)[1].lower()
        return [abs_path] if ext in extensions else []

    if not os.path.isdir(abs_path):
        return []

    results: List[str] = []
    for root, _, files in os.walk(abs_path):
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            if ext in extensions:
                results.append(os.path.join(root, name))
    return results


def normalize_audio_to_wav(src_path: str, dst_path: str) -> None:
    from pydub import AudioSegment

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_channels(1).set_frame_rate(TARGET_SR).set_sample_width(2)
    audio.export(dst_path, format="wav")


def get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def extract_video_audio_to_wav(src_path: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cmd = [
        get_ffmpeg_exe(),
        "-y",
        "-i",
        src_path,
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "-sample_fmt",
        "s16",
        dst_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg failed").strip())


def extract_audio_only(source_path: str, output_dir: str = EXTRACTED_AUDIO_DIR) -> int:
    videos = iter_files_by_extension(source_path, VIDEO_EXTENSIONS)
    if not videos:
        print(f"No supported video files found: {source_path}", flush=True)
        return 1

    print(f"Extracting audio from {len(videos)} video file(s)...", flush=True)
    failed = 0
    for video_path in videos:
        stem = safe_name(os.path.splitext(os.path.basename(video_path))[0])
        dst_path = unique_path(output_dir, f"{stem}.wav")
        try:
            extract_video_audio_to_wav(video_path, dst_path)
            print(f"Saved: {dst_path}", flush=True)
        except Exception as e:
            failed += 1
            print(f"Failed: {video_path} | {type(e).__name__}: {e}", flush=True)

    return 1 if failed else 0


def import_audio_folder(source_path: str, input_dir: str = INPUT_DIR) -> int:
    audio_files = iter_files_by_extension(source_path, AUDIO_EXTENSIONS)
    if not audio_files:
        print(f"No supported audio files found: {source_path}", flush=True)
        return 1

    print(f"Importing {len(audio_files)} audio file(s) into pipeline input...", flush=True)
    failed = 0
    for audio_path in audio_files:
        stem = safe_name(os.path.splitext(os.path.basename(audio_path))[0])
        dst_path = unique_path(input_dir, f"{stem}.wav")
        try:
            normalize_audio_to_wav(audio_path, dst_path)
            print(f"Saved input audio: {dst_path}", flush=True)
        except Exception as e:
            failed += 1
            print(f"Failed: {audio_path} | {type(e).__name__}: {e}", flush=True)

    return 1 if failed else 0


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix}: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


def run_pipeline(args: argparse.Namespace) -> int:
    extra_args: List[str] = []
    if args.input:
        extra_args.extend(["--input", args.input])
    if args.skip_classify:
        extra_args.append("--skip-classify")
    if getattr(args, "no_reset_output", False):
        extra_args.append("--no-reset-output")
    if args.max_segments is not None:
        extra_args.extend(["--max-segments", str(args.max_segments)])
    if args.classify_limit is not None:
        extra_args.extend(["--classify-limit", str(args.classify_limit)])
    if args.transcript_backend:
        extra_args.extend(["--transcript-backend", args.transcript_backend])
    if args.whisperx_model:
        extra_args.extend(["--whisperx-model", args.whisperx_model])
    if args.speaker_diarize:
        extra_args.append("--speaker-diarize")
    if args.diarization_model:
        extra_args.extend(["--diarization-model", args.diarization_model])
    if args.hf_token:
        extra_args.extend(["--hf-token", args.hf_token])
    if args.cut_mode:
        extra_args.extend(["--cut-mode", args.cut_mode])
    if args.min_emotion_clip_sec is not None:
        extra_args.extend(["--min-emotion-clip-sec", str(args.min_emotion_clip_sec)])
    if getattr(args, "emotion_set", None):
        extra_args.extend(["--emotion-set", str(args.emotion_set)])
    if getattr(args, "auto_label_policy", None):
        extra_args.extend(["--auto-label-policy", str(args.auto_label_policy)])
    if args.speaker_name_map:
        extra_args.extend(["--speaker-name-map", args.speaker_name_map])
    if args.reference_characters:
        extra_args.extend(["--reference-characters", args.reference_characters])
    if args.character_min_score is not None:
        extra_args.extend(["--character-min-score", str(args.character_min_score)])
    if args.character_min_margin is not None:
        extra_args.extend(["--character-min-margin", str(args.character_min_margin)])
    if getattr(args, "speaker_min_purity", None) is not None:
        extra_args.extend(["--speaker-min-purity", str(args.speaker_min_purity)])
    if getattr(args, "speaker_max_other_overlap", None) is not None:
        extra_args.extend(["--speaker-max-other-overlap", str(args.speaker_max_other_overlap)])
    if getattr(args, "openai_classifier_judge", False):
        extra_args.append("--openai-classifier-judge")
        openai_key = getattr(args, "openai_api_key", None)
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        if getattr(args, "openai_model", None):
            extra_args.extend(["--openai-model", args.openai_model])
        if getattr(args, "openai_judge_limit", None) is not None:
            extra_args.extend(["--openai-judge-limit", str(args.openai_judge_limit)])
        if getattr(args, "openai_timeout_sec", None) is not None:
            extra_args.extend(["--openai-timeout-sec", str(args.openai_timeout_sec)])
    return run_python_script(PIPELINE_SCRIPT, extra_args)


def args_with_menu_input(args: argparse.Namespace) -> argparse.Namespace:
    if args.input:
        return args

    prompt = (
        "Input video/audio file or folder path "
        f"(blank = use {INPUT_DIR}): "
    )
    raw_path = input(prompt).strip().strip('"')
    if not raw_path:
        return args

    run_args = argparse.Namespace(**vars(args))
    run_args.input = raw_path
    return run_args


def args_for_normal_emotion_pipeline(args: argparse.Namespace) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(args))
    run_args.speaker_diarize = False
    run_args.transcript_backend = "faster-whisper"
    run_args.hf_token = None
    run_args.speaker_name_map = None
    run_args.reference_characters = None
    run_args.no_reset_output = False
    return run_args


def args_with_single_file_input(args: argparse.Namespace) -> argparse.Namespace:
    if not args.input or not os.path.isdir(os.path.abspath(args.input)):
        return args

    candidates = iter_files_by_extension(
        args.input,
        VIDEO_EXTENSIONS | AUDIO_EXTENSIONS,
    )
    if not candidates:
        return args

    print("Choose one input file for this run:")
    for idx, path in enumerate(candidates, start=1):
        print(f"{idx}. {os.path.basename(path)}")

    while True:
        raw = input("> ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Enter a number from the list.")
            continue
        if 1 <= choice <= len(candidates):
            run_args = argparse.Namespace(**vars(args))
            run_args.input = candidates[choice - 1]
            return run_args
        print("Choice out of range.")


def load_processed_option8_files() -> Set[str]:
    if not os.path.isfile(OPTION8_PROGRESS_FILE):
        return set()
    with open(OPTION8_PROGRESS_FILE, "r", encoding="utf-8-sig") as f:
        return {os.path.abspath(line.strip()) for line in f if line.strip()}


def record_processed_option8_file(path: str) -> None:
    os.makedirs(os.path.dirname(OPTION8_PROGRESS_FILE), exist_ok=True)
    with open(OPTION8_PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(os.path.abspath(path) + "\n")


def looks_instrumental(path: str) -> bool:
    name = os.path.basename(path).lower()
    return any(term in name for term in ("instrumental", "instrumenta", "instrument", "karaoke", "bgm"))


def option8_sort_key(path: str) -> tuple:
    name = os.path.basename(path).lower()
    return (0 if "vocal" in name else 1, name)


def args_with_next_option8_file(args: argparse.Namespace) -> argparse.Namespace:
    if not args.input or not os.path.isdir(os.path.abspath(args.input)):
        return args

    candidates = iter_files_by_extension(args.input, VIDEO_EXTENSIONS | AUDIO_EXTENSIONS)
    candidates = sorted(
        [path for path in candidates if not looks_instrumental(path)],
        key=option8_sort_key,
    )
    if not candidates:
        return args

    processed = load_processed_option8_files()
    selected = next((path for path in candidates if os.path.abspath(path) not in processed), None)
    if selected is None:
        print("All option 8 candidate files in this folder were already processed. Restarting from the first vocals file.")
        selected = candidates[0]

    print(f"Option 8 auto-selected one file: {selected}")
    run_args = argparse.Namespace(**vars(args))
    run_args.input = selected
    return run_args


def option8_candidate_files(args: argparse.Namespace) -> List[str]:
    if not args.input:
        return []

    input_path = os.path.abspath(args.input)
    if os.path.isfile(input_path):
        return [input_path] if not looks_instrumental(input_path) else []
    if not os.path.isdir(input_path):
        return []

    candidates = iter_files_by_extension(input_path, VIDEO_EXTENSIONS | AUDIO_EXTENSIONS)
    return sorted(
        [path for path in candidates if not looks_instrumental(path)],
        key=option8_sort_key,
    )


def option8_files_to_process(args: argparse.Namespace) -> List[str]:
    candidates = option8_candidate_files(args)
    if not candidates:
        return []

    if len(candidates) == 1:
        return candidates

    processed = load_processed_option8_files()
    remaining = [path for path in candidates if os.path.abspath(path) not in processed]
    if remaining:
        return remaining

    print("All option 8 candidate files in this folder were already processed. Restarting from the first vocals file.")
    return candidates


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def load_hf_token_from_file() -> str:
    for path in (HF_TOKEN_FILE, HF_TOKEN_FILE_TXT):
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8-sig") as f:
                content = f.read().strip()
            match = re.search(r"hf_[A-Za-z0-9_\-]+", content)
            return match.group(0) if match else content
    return ""


def load_openai_key_from_file() -> tuple[str, Optional[str]]:
    for path in OPENAI_KEY_FILES:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                token = f.read().strip()
        except OSError:
            continue
        if token:
            return token, path
    return "", None


def args_with_openai_relabel_settings(args: argparse.Namespace) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(args))
    mode = input("Relabel judge mode: 1=normal/local, 2=OpenAI GPT (blank = normal): ").strip()
    if mode not in {"2", "gpt", "openai"}:
        run_args.openai_api_key = None
        run_args.openai_judge_limit = 0
        print("Using normal local emotion tournament.", flush=True)
        return run_args

    if getattr(run_args, "openai_api_key", None) or os.environ.get("OPENAI_API_KEY"):
        pass
    else:
        file_key, key_path = load_openai_key_from_file()
        if file_key:
            run_args.openai_api_key = file_key
            print(f"Using OpenAI API key from {key_path}", flush=True)
        else:
            key = input("OpenAI API key for option 2 LLM judge (blank = skip): ").strip()
            run_args.openai_api_key = key or None

    if getattr(run_args, "openai_api_key", None) or os.environ.get("OPENAI_API_KEY"):
        model = input(f"OpenAI model for emotion judge (blank = {run_args.openai_model}): ").strip()
        if model.startswith("sk-"):
            run_args.openai_api_key = model
            print("That looks like an API key, not a model name. Saved it as the key.", flush=True)
            model = input(f"OpenAI model for emotion judge (blank = {run_args.openai_model}): ").strip()
        if model:
            run_args.openai_model = model
        limit = input(
            f"Max OpenAI judge calls for this relabel run "
            f"(blank = {run_args.openai_judge_limit}, 0 = disable): "
        ).strip()
        if limit:
            try:
                run_args.openai_judge_limit = max(0, int(limit))
            except ValueError:
                print(f"Invalid judge call limit; keeping {run_args.openai_judge_limit}.", flush=True)
        print(
            f"OpenAI LLM judge enabled: model={run_args.openai_model}, "
            f"max_calls={run_args.openai_judge_limit}",
            flush=True,
        )
    else:
        print("OpenAI LLM judge skipped; using local emotion tournament only.", flush=True)
    return run_args


def args_with_openai_classifier_settings(args: argparse.Namespace) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(args))
    policy = input(
        "Auto-label policy? 1=balanced, 2=strict, 3=volume/weak labels (blank = balanced): "
    ).strip().lower()
    if policy in {"2", "strict"}:
        run_args.auto_label_policy = "strict"
    elif policy in {"3", "volume", "weak"}:
        run_args.auto_label_policy = "volume"
    else:
        run_args.auto_label_policy = "balanced"
    print(f"Auto-label policy: {run_args.auto_label_policy}", flush=True)

    mode = input("Classifier GPT judge for uncertain clips? 1=no, 2=yes (blank = no): ").strip()
    if mode not in {"2", "yes", "y", "gpt", "openai"}:
        run_args.openai_classifier_judge = False
        print("Using normal classifier: Whisper + audio model + prosody + manual review fallback.", flush=True)
        return run_args

    run_args.openai_classifier_judge = True
    if not (getattr(run_args, "openai_api_key", None) or os.environ.get("OPENAI_API_KEY")):
        file_key, key_path = load_openai_key_from_file()
        if file_key:
            run_args.openai_api_key = file_key
            print(f"Using OpenAI API key from {key_path}", flush=True)
        else:
            key = input("OpenAI API key for classifier GPT judge (blank = skip GPT): ").strip()
            run_args.openai_api_key = key or None

    if getattr(run_args, "openai_api_key", None) or os.environ.get("OPENAI_API_KEY"):
        model = input(f"OpenAI model for classifier judge (blank = {run_args.openai_model}): ").strip()
        if model.startswith("sk-"):
            run_args.openai_api_key = model
            print("That looks like an API key, not a model name. Saved it as the key.", flush=True)
            model = input(f"OpenAI model for classifier judge (blank = {run_args.openai_model}): ").strip()
        if model:
            run_args.openai_model = model
        limit = input(
            f"Max OpenAI judge calls for this classification run "
            f"(blank = {run_args.openai_judge_limit}, 0 = disable): "
        ).strip()
        if limit:
            try:
                run_args.openai_judge_limit = max(0, int(limit))
            except ValueError:
                print(f"Invalid judge call limit; keeping {run_args.openai_judge_limit}.", flush=True)
        print(
            f"Classifier GPT judge enabled for uncertain clips: model={run_args.openai_model}, "
            f"max_calls={run_args.openai_judge_limit}",
            flush=True,
        )
    else:
        run_args.openai_classifier_judge = False
        print("OpenAI key missing; using normal classifier only.", flush=True)
    return run_args


def print_pyannote_install_help() -> None:
    print("pyannote.audio is not installed in this Python environment.", flush=True)
    print("Install it in the same environment used by the workflow:", flush=True)
    print(f'"{sys.executable}" -m pip install pyannote.audio', flush=True)
    print("", flush=True)
    print("After installing, run option 8 again with a Hugging Face token.", flush=True)


def iter_segmented_dialogue_wavs() -> List[str]:
    if not os.path.isdir(SEGMENTED_DIALOGUE_DIR):
        return []
    results: List[str] = []
    for root, _, files in os.walk(SEGMENTED_DIALOGUE_DIR):
        for name in sorted(files):
            if name.lower().endswith(".wav"):
                results.append(os.path.abspath(os.path.join(root, name)))
    return results


def load_manifest_source_files() -> Set[str]:
    manifest_csv = os.path.join(CLASSIFIED_ROOT, "_multimodal_emotion_manifest.csv")
    if not os.path.isfile(manifest_csv):
        return set()

    import csv

    seen: Set[str] = set()
    with open(manifest_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            source_file = os.path.abspath(row.get("source_file", "").strip()) if row.get("source_file") else ""
            if source_file:
                seen.add(source_file)
    return seen


def find_unclassified_segmented_wavs() -> List[str]:
    segmented = iter_segmented_dialogue_wavs()
    classified = load_manifest_source_files()
    return [path for path in segmented if os.path.abspath(path) not in classified]


def classification_manifest_needs_rebuild(manifest_csv: str) -> bool:
    if not os.path.isfile(manifest_csv):
        return True

    import csv

    try:
        with open(manifest_csv, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return True

    if not rows:
        return True

    bad_source_rejects = sum(
        1
        for row in rows
        if (row.get("reason") or "").startswith("source_folder_rejected:")
    )
    if bad_source_rejects == len(rows):
        return True

    return False


def reset_auto_classification_outputs() -> None:
    keep_dirs = {"_approved", "manual_review_sorted"}
    keep_files = {"_manual_review_progress.csv"}

    if not os.path.isdir(CLASSIFIED_ROOT):
        os.makedirs(CLASSIFIED_ROOT, exist_ok=True)
        return

    for name in os.listdir(CLASSIFIED_ROOT):
        full_path = os.path.join(CLASSIFIED_ROOT, name)
        if os.path.isdir(full_path):
            if name in keep_dirs:
                continue
            shutil.rmtree(full_path, ignore_errors=True)
            continue
        if name in keep_files:
            continue
        os.remove(full_path)


def reset_speaker_emotion_outputs(speaker_dir: str) -> None:
    emotion_dirs = {
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Neutral",
        "Pleasant_Surprise",
        "Sad",
        "Needs_Review",
        "Rejected_LowQuality",
    }
    manifest_files = {
        "_multimodal_emotion_manifest.csv",
        "_multimodal_emotion_manifest_details.jsonl",
        "_high_confidence_manifest.csv",
        "_manual_review_queue.csv",
    }

    for name in os.listdir(speaker_dir):
        full_path = os.path.join(speaker_dir, name)
        if os.path.isdir(full_path) and name in emotion_dirs:
            shutil.rmtree(full_path, ignore_errors=True)
        elif os.path.isfile(full_path) and name in manifest_files:
            os.remove(full_path)


def rerun_classification_from_segmented(args: argparse.Namespace) -> int:
    if not os.path.isdir(SEGMENTED_DIALOGUE_DIR):
        print(f"No segmented dialogue folder found: {SEGMENTED_DIALOGUE_DIR}", flush=True)
        return 1

    print("Rebuilding auto-classification outputs from segmented dialogue clips...", flush=True)
    reset_auto_classification_outputs()

    import multimodal_emotion_classifier as mec

    mec.INPUT_BASES = [os.path.join(PIPELINE_WORKSPACE, "segmented")]
    mec.OUTPUT_BASE = CLASSIFIED_ROOT
    mec.CLASSIFY_SOURCE_FOLDERS = {"japanese_dialogue"}
    mec.REJECT_SOURCE_FOLDERS = set()
    mec.COPY_FILES = True
    mec.PRESERVE_ORIGINAL_FILENAME = True
    mec.CFG.setdefault("classification", {})["emotion_set"] = str(getattr(args, "emotion_set", "7") or "7")
    os.environ["EMOTION_SET"] = str(getattr(args, "emotion_set", "7") or "7")
    mec.CFG["models"]["asr"]["device"] = "cuda"
    mec.CFG["models"]["asr"]["compute_type"] = "float16"
    mec.CFG["models"]["audio_emotion"]["device"] = "cuda"

    original_collect = mec.collect_dialogue_segments
    limit = args.classify_limit

    def filtered_collect():
        items = [
            item
            for item in original_collect()
            if os.path.abspath(item["segment_dir"]) == os.path.abspath(SEGMENTED_DIALOGUE_DIR)
        ]
        if limit is not None:
            return items[:limit]
        return items

    mec.collect_dialogue_segments = filtered_collect
    try:
        mec.main()
        if getattr(args, "speaker_diarize", False):
            import extract_segment_multimodal_pipeline as pipeline

            counts = pipeline.group_classified_by_character(
                CLASSIFIED_ROOT,
                speaker_name_map_path=getattr(args, "speaker_name_map", None),
            )
            print(f"Character/emotion output: {os.path.join(CLASSIFIED_ROOT, '_by_character_emotion')}", flush=True)
            for key in sorted(counts):
                print(f"  {key}: {counts[key]}", flush=True)
        return 0
    finally:
        mec.collect_dialogue_segments = original_collect


def classify_one_speaker_folder(speaker_dir: str, args: argparse.Namespace) -> int:
    import multimodal_emotion_classifier as mec

    wavs = [
        name
        for name in sorted(os.listdir(speaker_dir))
        if name.lower().endswith(".wav") and os.path.isfile(os.path.join(speaker_dir, name))
    ]
    limit = getattr(args, "classify_limit", None)
    if limit is not None:
        wavs = wavs[: int(limit)]
    if not wavs:
        return 0

    reset_speaker_emotion_outputs(speaker_dir)

    original_input_bases = mec.INPUT_BASES
    original_output_base = mec.OUTPUT_BASE
    original_classify_sources = mec.CLASSIFY_SOURCE_FOLDERS
    original_reject_sources = mec.REJECT_SOURCE_FOLDERS
    original_copy_files = mec.COPY_FILES
    original_preserve_filename = mec.PRESERVE_ORIGINAL_FILENAME
    original_collect = mec.collect_dialogue_segments

    def speaker_collect():
        manifest_path = os.path.join(SEGMENTED_DIALOGUE_DIR, "_segments_manifest.csv")
        return [
            {
                "file_path": os.path.join(speaker_dir, file_name),
                "file_name": file_name,
                "segment_dir": speaker_dir,
                "manifest_path": manifest_path,
                "source_group": "japanese_dialogue",
            }
            for file_name in wavs
        ]

    try:
        mec.INPUT_BASES = [speaker_dir]
        mec.OUTPUT_BASE = speaker_dir
        mec.CLASSIFY_SOURCE_FOLDERS = {"japanese_dialogue"}
        mec.REJECT_SOURCE_FOLDERS = set()
        mec.COPY_FILES = True
        mec.PRESERVE_ORIGINAL_FILENAME = True
        mec.CFG.setdefault("classification", {})["emotion_set"] = str(getattr(args, "emotion_set", "7") or "7")
        os.environ["EMOTION_SET"] = str(getattr(args, "emotion_set", "7") or "7")
        mec.CFG["models"]["asr"]["device"] = "cuda"
        mec.CFG["models"]["asr"]["compute_type"] = "float16"
        mec.CFG["models"]["audio_emotion"]["device"] = "cuda"
        mec.collect_dialogue_segments = speaker_collect
        mec.main()
        return 0
    except Exception as e:
        print(f"Speaker classification failed for {speaker_dir}: {type(e).__name__}: {e}", flush=True)
        return 1
    finally:
        mec.INPUT_BASES = original_input_bases
        mec.OUTPUT_BASE = original_output_base
        mec.CLASSIFY_SOURCE_FOLDERS = original_classify_sources
        mec.REJECT_SOURCE_FOLDERS = original_reject_sources
        mec.COPY_FILES = original_copy_files
        mec.PRESERVE_ORIGINAL_FILENAME = original_preserve_filename
        mec.collect_dialogue_segments = original_collect


def classify_speaker_folders(args: argparse.Namespace) -> int:
    if not os.path.isdir(SEGMENTED_BY_SPEAKER_DIR):
        print(f"No speaker folder found: {SEGMENTED_BY_SPEAKER_DIR}", flush=True)
        return 1

    speakers = [
        os.path.join(SEGMENTED_BY_SPEAKER_DIR, name)
        for name in sorted(os.listdir(SEGMENTED_BY_SPEAKER_DIR))
        if os.path.isdir(os.path.join(SEGMENTED_BY_SPEAKER_DIR, name))
    ]
    if not speakers:
        print(f"No speaker subfolders found: {SEGMENTED_BY_SPEAKER_DIR}", flush=True)
        return 1

    failed = 0
    print(f"Classifying emotions inside {len(speakers)} speaker folder(s)...", flush=True)
    for idx, speaker_dir in enumerate(speakers, start=1):
        wav_count = sum(
            1
            for name in os.listdir(speaker_dir)
            if name.lower().endswith(".wav") and os.path.isfile(os.path.join(speaker_dir, name))
        )
        print(f"\nSpeaker {idx}/{len(speakers)}: {os.path.basename(speaker_dir)} ({wav_count} wavs)", flush=True)
        if wav_count == 0:
            continue
        code = classify_one_speaker_folder(speaker_dir, args)
        if code != 0:
            failed += 1

    return 1 if failed else 0


def sync_classification_if_needed(args: argparse.Namespace) -> int:
    manifest_csv = os.path.join(CLASSIFIED_ROOT, "_multimodal_emotion_manifest.csv")
    if classification_manifest_needs_rebuild(manifest_csv):
        print("Classification manifest is missing or invalid. Rebuilding auto-classification outputs...", flush=True)
        return rerun_classification_from_segmented(args)

    missing = find_unclassified_segmented_wavs()
    if missing:
        print(f"Unclassified segmented clips detected: {len(missing)}", flush=True)
        return rerun_classification_from_segmented(args)

    print("No unclassified segmented clips detected. Using existing classification manifest.", flush=True)
    return 0


def run_relabel(args: argparse.Namespace) -> int:
    exit_code = sync_classification_if_needed(args)
    if exit_code != 0:
        return exit_code

    os.environ["EMOTION_SET"] = str(getattr(args, "emotion_set", "7") or "7")
    extra_args = [
        "--target-per-emotion",
        str(args.target_per_emotion),
        "--candidate-batch",
        str(args.candidate_batch),
        "--compare-rounds",
        str(args.compare_rounds),
        "--candidate-min-score",
        str(args.candidate_min_score),
    ]
    openai_key = getattr(args, "openai_api_key", None)
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if getattr(args, "openai_model", None):
        extra_args.extend(["--openai-model", args.openai_model])
    if getattr(args, "openai_judge_limit", None) is not None:
        extra_args.extend(["--openai-judge-limit", str(args.openai_judge_limit)])
    if getattr(args, "openai_timeout_sec", None) is not None:
        extra_args.extend(["--openai-timeout-sec", str(args.openai_timeout_sec)])
    return run_python_script(RELABEL_SCRIPT, extra_args)


def run_review(args: argparse.Namespace) -> int:
    extra_args: List[str] = []
    if args.review_queue:
        extra_args.extend(["--queue", args.review_queue])
    if args.no_autoplay:
        extra_args.append("--no-autoplay")
    return run_python_script(REVIEW_SCRIPT, extra_args)


def run_cluster_plot(args: argparse.Namespace) -> int:
    output_root = str(args.output_root or PIPELINE_WORKSPACE)
    output_root_path = Path(output_root)
    print(f"Using output root: {Path(output_root).resolve()}", flush=True)
    character_dir = output_root_path / getattr(args, "classification_output_name", "character_emotion_classified")
    basic_dir = output_root_path / "multimodal_emotion_classified"
    print(f"Using basic emotion folder: {basic_dir}", flush=True)
    print(f"Using character emotion folder: {character_dir}", flush=True)
    if getattr(args, "dry_run", False):
        roots = [path for path in [basic_dir, character_dir] if path.is_dir()]
        print("Dry run: cluster analysis would inspect:", flush=True)
        for root in roots:
            print(f"  {root}", flush=True)
            for path in sorted(p for p in root.iterdir() if p.is_dir()):
                if path.name.startswith("_") and path.name != "_global":
                    continue
                print(f"    {path}", flush=True)
        return 0

    exit_code = 0
    if basic_dir.is_dir():
        try:
            from labeling_quality_control import normalize_rows, write_cluster_analysis

            rows = normalize_rows(basic_dir, "basic")
            if rows:
                write_cluster_analysis(basic_dir, rows)
                print(f"Saved basic cluster analysis: {basic_dir / 'cluster_analysis'}", flush=True)
            else:
                print(f"No basic rows found for cluster analysis: {basic_dir}", flush=True)
                exit_code = max(exit_code, 1)
        except Exception as exc:
            print(f"Basic cluster analysis failed: {exc}", flush=True)
            exit_code = max(exit_code, 1)

    if character_dir.is_dir():
        extra_args = [
            "--output-root",
            output_root,
            "--per-emotion-limit",
            str(args.cluster_plot_limit),
            "--cluster-feature",
            str(getattr(args, "cluster_feature", "acoustic") or "acoustic"),
        ]
        if getattr(args, "skip_tsne", False):
            extra_args.append("--skip-tsne")
        if getattr(args, "skip_umap", False):
            extra_args.append("--skip-umap")
        if getattr(args, "plot_global_only", False):
            extra_args.append("--plot-global-only")
        if getattr(args, "plot_shortlist_only", False):
            extra_args.append("--plot-shortlist-only")
        exit_code = max(exit_code, run_python_script(CHARACTER_CLUSTER_ANALYSIS_SCRIPT, extra_args))
    elif not basic_dir.is_dir():
        print(f"No analysis input found: {basic_dir} or {character_dir}", flush=True)
        return 1
    return exit_code


def run_speaker_emotion_cluster_plots(args: argparse.Namespace) -> int:
    by_speaker_root = SEGMENTED_BY_SPEAKER_DIR
    extra_args = [
        "--by-speaker-root",
        by_speaker_root,
        "--per-emotion-limit",
        str(args.cluster_plot_limit),
        "--embedding-backend",
        args.cluster_embedding_backend,
        "--embedding-model",
        args.cluster_embedding_model,
        "--embedding-device",
        args.cluster_embedding_device,
        "--reducer",
        args.cluster_reducer,
    ]
    return run_python_script(CLUSTER_PLOT_SCRIPT, extra_args)


def has_japanese_dialogue_clips(input_root: str = SEGMENTED_DIALOGUE_DIR) -> bool:
    if not os.path.isdir(input_root):
        return False
    for root, _, files in os.walk(input_root):
        if any(name.lower().endswith(".wav") for name in files):
            return True
    return False


def resolve_raw_input(raw_arg: Optional[str]) -> Path:
    if raw_arg:
        return Path(raw_arg).expanduser().resolve()
    if DEFAULT_RAW_INPUT_DIR.exists():
        return DEFAULT_RAW_INPUT_DIR.resolve()
    return Path(INPUT_DIR).resolve()


def supported_raw_media_files(raw_input: Path) -> List[Path]:
    if raw_input.is_file():
        return [raw_input] if raw_input.suffix.lower() in RAW_MEDIA_EXTENSIONS else []
    if not raw_input.is_dir():
        return []
    return sorted(
        path
        for path in raw_input.rglob("*")
        if path.is_file() and path.suffix.lower() in RAW_MEDIA_EXTENSIONS
    )


def count_japanese_dialogue_clips() -> int:
    dialogue_dir = Path(SEGMENTED_DIALOGUE_DIR)
    if not dialogue_dir.is_dir():
        return 0
    return sum(
        1
        for path in dialogue_dir.rglob("*.wav")
        if path.is_file() and not path.name.startswith("_")
    )


def clear_japanese_dialogue_clips() -> None:
    dialogue_dir = Path(SEGMENTED_DIALOGUE_DIR)
    if dialogue_dir.exists():
        shutil.rmtree(dialogue_dir)
    dialogue_dir.mkdir(parents=True, exist_ok=True)


def mirror_dialogue_to_sentence_clips() -> None:
    src_dir = Path(SEGMENTED_DIALOGUE_DIR)
    dst_dir = Path(PIPELINE_WORKSPACE) / "segmented" / "whisperx_sentence_clips"
    if not src_dir.is_dir():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for wav in sorted(src_dir.glob("*.wav")):
        dst = dst_dir / wav.name
        if not dst.exists():
            shutil.copy2(wav, dst)
    manifest = src_dir / "_segments_manifest.csv"
    if manifest.is_file():
        shutil.copy2(manifest, dst_dir / "_sentence_clip_manifest.csv")


def count_speaker_cluster_dirs(output_root: str) -> int:
    cluster_root = Path(output_root) / "speaker_clusters"
    if not cluster_root.is_dir():
        return 0
    return sum(
        1
        for path in cluster_root.iterdir()
        if path.is_dir() and re.fullmatch(r"SPEAKER_\d+", path.name)
    )


def print_emotion_folder_counts(root: Path) -> None:
    print(f"PCA source selected: {root}", flush=True)
    class_count = 0
    for emotion in ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant_Surprise", "Sad"]:
        count = len(list((root / emotion).glob("*.wav"))) if (root / emotion).is_dir() else 0
        if count:
            class_count += 1
        print(f"  {emotion}: {count}", flush=True)
    if class_count <= 1:
        print("PCA cannot show emotion separation because only one class exists.", flush=True)


def run_dialogue_segmentation_only(args: argparse.Namespace, raw_input: Path) -> int:
    prep_args = argparse.Namespace(**vars(args))
    prep_args.input = str(raw_input)
    prep_args.skip_classify = True
    prep_args.speaker_diarize = False
    prep_args.speaker_name_map = None
    prep_args.reference_characters = None
    prep_args.no_reset_output = True
    return run_pipeline(prep_args)


def run_character_cluster(args: argparse.Namespace) -> int:
    raw_input = resolve_raw_input(args.input or args.input_root)
    print(f"Using output root: {Path(args.output_root or PIPELINE_WORKSPACE).resolve()}", flush=True)
    print(f"Character clustering raw input: {raw_input}", flush=True)

    if not raw_input.exists():
        print(f"Raw input path not found: {raw_input}", flush=True)
        return 1

    raw_files = supported_raw_media_files(raw_input)
    print(f"Found {len(raw_files)} raw audio/video files for segmentation.", flush=True)
    if not raw_files:
        print(f"No supported audio/video files found in: {raw_input}", flush=True)
        return 1

    existing_dialogue_count = count_japanese_dialogue_clips()
    segmentation_ran = False
    if args.force_resplit or existing_dialogue_count == 0:
        if args.force_resplit:
            clear_japanese_dialogue_clips()
        code = run_dialogue_segmentation_only(args, raw_input)
        segmentation_ran = True
        if code != 0:
            print(f"Dialogue preparation failed with exit code {code}; speaker clustering stopped.", flush=True)
            return code
    else:
        print(
            f"Reusing existing Japanese dialogue clips in: {SEGMENTED_DIALOGUE_DIR}",
            flush=True,
        )

    dialogue_count = count_japanese_dialogue_clips()
    print(f"Japanese dialogue clips created: {dialogue_count if segmentation_ran else 0}", flush=True)
    if not segmentation_ran:
        print(f"Japanese dialogue clips reused: {dialogue_count}", flush=True)
    if dialogue_count == 0:
        print("No Japanese dialogue clips found after segmentation.", flush=True)
        return 1
    mirror_dialogue_to_sentence_clips()

    extra_args = [
        "--input-root",
        SEGMENTED_DIALOGUE_DIR,
        "--output-root",
        args.output_root or PIPELINE_WORKSPACE,
        "--cluster-method",
        args.cluster_method,
    ]
    if args.expected_speakers:
        extra_args.extend(["--expected-speakers", str(args.expected_speakers)])
    if args.copy_files:
        extra_args.append("--copy-files")
    if args.force_character_map_template:
        extra_args.append("--force-character-map-template")
    code = run_python_script(SPEAKER_CLUSTER_SCRIPT, extra_args)
    if code != 0:
        return code

    output_root = args.output_root or PIPELINE_WORKSPACE
    print("Character/speaker clustering complete.", flush=True)
    print(f"Speaker clusters created: {count_speaker_cluster_dirs(output_root)}", flush=True)
    print(
        f"Optional: edit {Path(output_root) / 'speaker_clusters' / 'character_map_template.json'} before running option 7 for real character names.",
        flush=True,
    )
    print("If you do not edit it, option 7 will use SPEAKER_XX names automatically.", flush=True)
    return 0


def build_auto_speaker_character_map(path: str, output_root: str) -> Optional[str]:
    import csv
    import json

    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Character map could not be read: {type(e).__name__}: {e}", flush=True)
        return None

    speaker_manifest = Path(output_root) / "speaker_clusters" / "_speaker_cluster_manifest.csv"
    current_clusters = set()
    if speaker_manifest.is_file():
        with speaker_manifest.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                cluster = str(row.get("speaker_cluster", "")).strip()
                if re.fullmatch(r"SPEAKER_\d+", cluster):
                    current_clusters.add(cluster)
    current_clusters.update(str(key).strip() for key in data if re.fullmatch(r"SPEAKER_\d+", str(key).strip()))

    has_auto_changes = False
    auto_map = {}
    for speaker in sorted(current_clusters):
        text = str(data.get(speaker, "")).strip()
        if re.fullmatch(r"Character_\d+", text):
            auto_map[speaker] = speaker
            has_auto_changes = True
        elif not text:
            auto_map[speaker] = speaker
            has_auto_changes = True
        else:
            auto_map[speaker] = text or "Unknown"

    if not has_auto_changes:
        return path

    auto_path = Path(output_root) / "speaker_clusters" / "_character_map_auto_speaker.json"
    with auto_path.open("w", encoding="utf-8") as f:
        json.dump(auto_map, f, ensure_ascii=False, indent=2)
    print(
        "Character map contains placeholder names; using automatic speaker IDs as character names for this run.",
        flush=True,
    )
    print(f"Automatic speaker map saved: {auto_path}", flush=True)
    print("Edit character_map_template.json later if you want real character names instead of SPEAKER_XX.", flush=True)
    return str(auto_path)


def character_map_has_placeholders(path: str) -> bool:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    for value in data.values():
        text = str(value).strip()
        if re.fullmatch(r"Character_\d+", text):
            return True
    return False


def validate_character_map(path: str, output_root: str, print_details: bool = True) -> bool:
    report_path = Path(output_root) / "speaker_clusters" / "_character_map_validation_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception as exc:
        rows.append({"speaker_cluster": "", "mapped_name": "", "status": "invalid", "reason": f"{type(exc).__name__}: {exc}"})
        with report_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["speaker_cluster", "mapped_name", "status", "reason"])
            writer.writeheader()
            writer.writerows(rows)
        if print_details:
            print(f"Character map could not be read: {type(exc).__name__}: {exc}", flush=True)
        return False

    placeholders = []
    for speaker, mapped in sorted(data.items()):
        speaker_text = str(speaker).strip()
        mapped_text = str(mapped).strip()
        if re.fullmatch(r"Character_\d+", mapped_text):
            status, reason = "placeholder", "edit this generated Character_XX placeholder"
            placeholders.append(speaker_text)
        elif not mapped_text:
            status, reason = "invalid", "empty mapping"
        else:
            status, reason = "ok", "accepted"
        rows.append({"speaker_cluster": speaker_text, "mapped_name": mapped_text, "status": status, "reason": reason})

    with report_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker_cluster", "mapped_name", "status", "reason"])
        writer.writeheader()
        writer.writerows(rows)

    if placeholders:
        if print_details:
            print("Character map still contains placeholder names. Please edit these keys before running option 7:", flush=True)
            for key in placeholders:
                print(f"  {key}", flush=True)
            print(f"Validation report saved: {report_path}", flush=True)
            print("Temporary names like Speaker00_Unchecked, Bocchi_candidate, Kita_candidate, Nijika_candidate, Ryo_candidate, and Unknown are allowed.", flush=True)
        return False
    if print_details:
        print(f"Character map validation report saved: {report_path}", flush=True)
    return True


def run_character_map(args: argparse.Namespace) -> int:
    extra_args = [
        "--output-root",
        args.output_root or PIPELINE_WORKSPACE,
    ]
    if args.character_map:
        extra_args.extend(["--character-map", args.character_map])
    if args.copy_files:
        extra_args.append("--copy-files")
    return run_python_script(CHARACTER_MAP_SCRIPT, extra_args)


def run_character_emotion_labeling(args: argparse.Namespace) -> int:
    output_root = args.output_root or PIPELINE_WORKSPACE
    default_map = os.path.join(output_root, "speaker_clusters", "character_map_template.json")
    map_path = args.character_map or default_map
    print(f"Using output root: {Path(output_root).resolve()}", flush=True)
    print(f"Using speaker cluster folder: {Path(output_root) / 'speaker_clusters'}", flush=True)
    print(f"Using character map: {Path(map_path).resolve()}", flush=True)
    print(f"Using character emotion folder: {Path(output_root) / getattr(args, 'classification_output_name', 'character_emotion_classified')}", flush=True)
    if not os.path.isfile(map_path):
        print("Character map not found. Run option 6 first.", flush=True)
        return 1
    if not validate_character_map(map_path, output_root, print_details=True):
        return 1

    effective_map = map_path

    if getattr(args, "dry_run", False):
        manifest = Path(output_root) / "speaker_clusters" / "_speaker_cluster_manifest.csv"
        char_manifest = Path(output_root) / "character_dialogue" / "_character_dialogue_manifest.csv"
        speaker_rows = 0
        if manifest.is_file():
            with manifest.open("r", encoding="utf-8-sig", newline="") as f:
                speaker_rows = sum(1 for _ in csv.DictReader(f))
        print("Dry run: option 7 would run character mapping, pseudo-labeling, shortlist generation, cluster analysis, and final dataset build.", flush=True)
        print(f"Speaker manifest rows: {speaker_rows}", flush=True)
        print(f"Existing character dialogue manifest: {char_manifest if char_manifest.is_file() else 'missing'}", flush=True)
        return 0

    map_args = argparse.Namespace(**vars(args))
    map_args.character_map = effective_map
    code = run_character_map(map_args)
    if code != 0:
        return code

    extra_args = [
        "--input-root",
        args.input_root or os.path.join(output_root, "character_dialogue"),
        "--output-root",
        output_root,
        "--target-per-emotion",
        str(args.target_per_emotion),
        "--review-sample-per-class",
        str(args.review_sample_per_class),
        "--max-review-total",
        str(args.max_review_total),
        "--embedding-model",
        args.cluster_embedding_model,
        "--cluster-embedding-device",
        args.cluster_embedding_device,
    ]
    if args.classify_limit is not None:
        extra_args.extend(["--classify-limit", str(args.classify_limit)])
    if getattr(args, "classification_output_name", None):
        extra_args.extend(["--classification-output-name", args.classification_output_name])
    if getattr(args, "trained_model_weights", False):
        extra_args.append("--trained-model-weights")
    if args.skip_emotion_labeling:
        extra_args.append("--skip-emotion-labeling")
    if getattr(args, "include_rejected_review", False):
        extra_args.append("--include-rejected-review")
    if args.copy_files:
        extra_args.append("--copy-files")
    code = run_python_script(CHARACTER_EMOTION_SCRIPT, extra_args)
    if code != 0:
        return code
    analysis_code = run_cluster_plot(args)
    if analysis_code != 0:
        print(f"Warning: cluster analysis finished with exit code {analysis_code}", flush=True)
    return run_build_final_dataset(args, source_output_name=getattr(args, "classification_output_name", "character_emotion_classified"))


def run_text_guided_emotion_labeling(args: argparse.Namespace) -> int:
    return run_python_script(TEXT_GUIDED_EMOTION_SCRIPT, [])


def run_cluster_clean_target(args: argparse.Namespace) -> int:
    output_root = str(args.output_root or PIPELINE_WORKSPACE)
    print(f"Using output root: {Path(output_root).resolve()}", flush=True)
    extra_args = [
        "--output-root",
        output_root,
        "--target-per-emotion",
        str(args.target_per_emotion),
        "--cluster-clean-feature",
        str(getattr(args, "cluster_clean_feature", "multi_feature") or "multi_feature"),
    ]
    if getattr(args, "allow_overlap_fill", False):
        extra_args.append("--allow-overlap-fill")
    if getattr(args, "allow_usable_fill", False):
        extra_args.append("--allow-usable-fill")
    return run_python_script(CLUSTER_CLEAN_TARGET_SCRIPT, extra_args)


def run_quality_control_for(args: argparse.Namespace, workflow_type: str) -> int:
    from labeling_quality_control import run_post_label_quality_control

    output_root = Path(args.output_root or PIPELINE_WORKSPACE)
    input_dir = output_root / ("character_emotion_classified" if workflow_type == "character" else "multimodal_emotion_classified")
    return int(
        run_post_label_quality_control(
            output_root=output_root,
            input_emotion_dir=input_dir,
            workflow_type=workflow_type,
            target_per_emotion=int(args.target_per_emotion),
            max_review_total=int(args.max_review_total),
            review_sample_per_class=int(args.review_sample_per_class),
            cluster_clean=True,
            cluster_clean_feature=str(getattr(args, "cluster_clean_feature", "multi_feature") or "multi_feature"),
        )
        or 0
    )


def run_refresh_quality_control(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root or PIPELINE_WORKSPACE)
    code = 0
    if (output_root / "multimodal_emotion_classified").is_dir():
        code = max(code, run_quality_control_for(args, "basic"))
    if (output_root / "character_emotion_classified").is_dir():
        code = max(code, run_quality_control_for(args, "character"))
    if code == 0 and not (output_root / "multimodal_emotion_classified").is_dir() and not (output_root / "character_emotion_classified").is_dir():
        print(f"No labeling outputs found under: {output_root}", flush=True)
    return code


def refresh_cluster_clean_target_if_available(args: argparse.Namespace, reason: str = "") -> int:
    output_root = Path(args.output_root or PIPELINE_WORKSPACE)
    manifest = output_root / "character_emotion_classified" / "_character_emotion_manifest.csv"
    if not manifest.is_file():
        print(
            "Cluster-clean target skipped: character-aware Step 7 manifest not found. "
            "Run Step 6 -> Step 7 first if you want overlap filtering.",
            flush=True,
        )
        return 0
    if reason:
        print(f"Refreshing cluster-clean 100 target after {reason}.", flush=True)
    return run_cluster_clean_target(args)


def audit_status(path: Path) -> str:
    return "ok" if path.exists() else "missing"


def write_function_audit_report(output_root: Path) -> Path:
    audit_dir = output_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    report = audit_dir / "_function_audit_report.md"
    rows = [
        ("emotion_workflow_runner.py", "menu/main/run_*", "CLI/user", "--mode, --output-root", "subprocess workflow outputs", "ok", "Routing needed audit mode/logging", "Added audit-workflow/dry-run/logging"),
        ("speaker_embedding_cluster.py", "speaker clustering", "option 6", "segmented/japanese_dialogue", "speaker_clusters manifests/reports/map template", "ok", "S3/Speaker folder quality depends on embeddings", "Reviewed routing/output-root"),
        ("character_cluster_mapper.py", "run_character_mapping", "option 7", "speaker_clusters + character_map_template.json", "character_dialogue + rejected manifest", "ok", "Map placeholders need clear validation", "Added validation in runner"),
        ("character_emotion_pseudolabel.py", "run_character_emotion", "option 7", "character_dialogue", "character_emotion_classified reports/shortlist/debug", "ok", "Needs_Review too large; rejected count bug", "Added shortlist/count consistency/rejected count fix"),
        ("character_cluster_analysis.py", "run/collect_points", "option 7/8", "character_emotion_classified", "cluster_analysis png/csv", "ok", "Candidate folders not plotted", "Candidate folders map to canonical labels"),
        ("manual_review_helper.py", "manual review", "option 3", "review queue", "approved/rejected review outputs", "not_changed", "Basic workflow only", "No change"),
        ("iterative_emotion_relabel.py", "relabel loop", "option 2/5", "classified/review folders", "refreshed candidates", "not_changed", "Research methods untouched", "No change"),
        ("build_final_emotion_dataset.py", "build final dataset", "option 7 after labeling", "auto_accept manifest rows", "final_character_emotion_dataset", "ok", "Final remains small due strict accept", "No accept threshold change"),
        ("plot_emotion_clusters.py", "basic cluster plots", "legacy/basic", "multimodal classified folders", "PCA plots", "not_changed", "Basic workflow only", "No change"),
        ("extract_segment_multimodal_pipeline.py", "segmentation/basic pipeline", "option 1/6 prep", "raw input", "segmented clips/basic classified output", "not_changed", "Heavy pipeline not run during audit", "No change"),
    ]
    with report.open("w", encoding="utf-8") as f:
        f.write("# Function Audit Report\n\n")
        f.write("| Script | Function | Called by | Inputs | Outputs | Status | Issue | Fix Applied |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for row in rows:
            f.write("| " + " | ".join(str(x).replace("|", "/") for x in row) + " |\n")
    return report


def run_audit_workflow(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root or PIPELINE_WORKSPACE).resolve()
    audit_dir = output_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    character_root = output_root / getattr(args, "classification_output_name", "character_emotion_classified")
    speaker_root = output_root / "speaker_clusters"
    map_path = Path(args.character_map).resolve() if args.character_map else speaker_root / "character_map_template.json"
    checks = []

    def add(check_name: str, ok: bool, details: str, fix: str = "") -> None:
        checks.append({"check_name": check_name, "status": "ok" if ok else "warning", "details": details, "fix_suggestion": fix})

    print(f"Using output root: {output_root}", flush=True)
    print(f"Using speaker cluster folder: {speaker_root}", flush=True)
    print(f"Using character map: {map_path}", flush=True)
    print(f"Using character emotion folder: {character_root}", flush=True)

    add("speaker_clusters_folder", speaker_root.is_dir(), str(speaker_root), "Run option 6.")
    add("character_emotion_folder", character_root.is_dir(), str(character_root), "Run option 7.")
    add("character_map_exists", map_path.is_file(), str(map_path), "Run option 6 then edit character_map_template.json.")
    if map_path.is_file():
        valid_map = validate_character_map(str(map_path), str(output_root), print_details=False)
        add("character_map_valid", valid_map, str(speaker_root / "_character_map_validation_report.csv"), "Replace Character_XX placeholders.")

    expected_reports = [
        "_character_emotion_manifest.csv",
        "_review_shortlist_manifest.csv",
        "_review_workload_report.csv",
        "_count_consistency_report.csv",
        "_candidate_summary_report.csv",
        "_branch_bias_report.csv",
        "_model_status_report.csv",
    ]
    for name in expected_reports:
        add(f"report_{name}", (character_root / name).is_file(), str(character_root / name), "Rerun option 7 or refresh shortlist with --skip-emotion-labeling.")

    shortlist = character_root / "_review_shortlist"
    shortlist_count = sum(1 for p in shortlist.rglob("*.wav")) if shortlist.is_dir() else 0
    add("review_shortlist", shortlist.is_dir() and shortlist_count <= int(args.max_review_total), f"{shortlist_count} clips at {shortlist}", "Run option 7; shortlist must be <= --max-review-total.")

    for folder in ["Candidate_Fear", "Candidate_Disgust", "Candidate_Pleasant_Surprise"]:
        add(f"global_{folder}", (character_root / "_global" / folder).is_dir(), str(character_root / "_global" / folder), "Rerun option 7.")

    graph_root = character_root / "_global" / "cluster_analysis"
    for name in ["pca_7emotion.png", "tsne_7emotion.png", "distance_histogram.png", "class_count_bar.png", "cluster_points.csv", "cluster_summary.csv"]:
        add(f"global_graph_{name}", (graph_root / name).is_file(), str(graph_root / name), "Run option 8.")

    consistency = character_root / "_count_consistency_report.csv"
    if consistency.is_file():
        with consistency.open("r", encoding="utf-8-sig", newline="") as f:
            mismatches = [row for row in csv.DictReader(f) if row.get("status") != "ok"]
        add("count_consistency", not mismatches, f"{len(mismatches)} mismatch rows", "Refresh reports from option 7.")

    function_report = write_function_audit_report(output_root)
    add("function_audit_report", function_report.is_file(), str(function_report), "")

    summary = audit_dir / "_workflow_audit_summary.csv"
    with summary.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check_name", "status", "details", "fix_suggestion"])
        writer.writeheader()
        writer.writerows(checks)
    print(f"Workflow audit summary saved: {summary}", flush=True)
    print(f"Function audit report saved: {function_report}", flush=True)
    warnings = sum(1 for row in checks if row["status"] != "ok")
    print(f"Audit warnings: {warnings}", flush=True)
    return 0


def run_train_embedding_classifier(args: argparse.Namespace) -> int:
    extra_args = []
    if args.input_root:
        extra_args.extend(["--input-root", args.input_root])
    return run_python_script(TRAIN_EMBEDDING_SCRIPT, extra_args)


def run_self_train_emotion(args: argparse.Namespace) -> int:
    extra_args = [
        "--output-root",
        args.output_root or PIPELINE_WORKSPACE,
        "--final-output",
        args.final_output,
    ]
    if args.copy_files:
        extra_args.append("--copy-files")
    return run_python_script(SELF_TRAIN_SCRIPT, extra_args)


def run_relabel_with_trained_model(args: argparse.Namespace) -> int:
    relabel_args = argparse.Namespace(**vars(args))
    relabel_args.classification_output_name = "character_emotion_classified_second_pass"
    relabel_args.trained_model_weights = True
    return run_character_emotion_labeling(relabel_args)


def run_build_final_dataset(args: argparse.Namespace, source_output_name: str = "character_emotion_classified") -> int:
    default_final_output = os.path.join(BASE_DIR, "output", "final_character_emotion_dataset")
    final_output = args.final_output
    if args.output_root and os.path.abspath(str(args.final_output)) == os.path.abspath(default_final_output):
        final_output = str(Path(args.output_root) / "final_character_emotion_dataset")
    extra_args = [
        "--output-root",
        args.output_root or PIPELINE_WORKSPACE,
        "--final-output",
        final_output,
        "--source-output-name",
        source_output_name,
    ]
    if args.copy_files:
        extra_args.append("--copy-files")
    return run_python_script(FINAL_DATASET_SCRIPT, extra_args)


def run_full_loop(args: argparse.Namespace) -> int:
    exit_code = run_pipeline(args)
    if exit_code != 0:
        return exit_code

    exit_code = run_relabel(args)
    if exit_code != 0:
        return exit_code

    if args.skip_review:
        return 0

    exit_code = run_review(args)
    if exit_code != 0:
        return exit_code

    return run_relabel(args)


def menu(args: argparse.Namespace) -> int:
    while True:
        print("")
        print("Emotion Workflow")
        print("")
        print("[BASIC LABELING WORKFLOW]")
        print("1. Run full pipeline")
        print("2. Relabel: refresh candidates and reduce Needs_Review")
        print("3. Start manual review")
        print("4. Full basic loop: pipeline -> relabel -> review -> relabel")
        print("5. Promote approved + regroup only")
        print("")
        print("[CHARACTER/SPEAKER-AWARE LABELING WORKFLOW]")
        print("6. Character/speaker clustering")
        print("7. Character-aware 7-emotion labeling")
        print("8. Plot speaker + emotion cluster analysis")
        print("q. Quit")
        choice = input("> ").strip().lower()

        if choice == "1":
            run_args = args_for_normal_emotion_pipeline(args_with_menu_input(args))
            run_args = args_with_openai_classifier_settings(run_args)
            code = run_pipeline(run_args)
            if code == 0:
                run_quality_control_for(run_args, "basic")
            print(f"Pipeline finished with exit code {code}")
            continue
        if choice == "2":
            run_args = args_with_openai_relabel_settings(args_for_normal_emotion_pipeline(args))
            code = run_relabel(run_args)
            if code == 0:
                run_quality_control_for(run_args, "basic")
            print(f"Relabel refresh finished with exit code {code}")
            continue
        if choice == "3":
            code = run_review(args)
            if code == 0:
                run_refresh_quality_control(args)
            print(f"Manual review finished with exit code {code}")
            continue
        if choice == "4":
            run_args = args_for_normal_emotion_pipeline(args_with_menu_input(args))
            code = run_full_loop(run_args)
            if code == 0:
                run_quality_control_for(run_args, "basic")
            print(f"Full loop finished with exit code {code}")
            continue
        if choice == "5":
            os.environ["EMOTION_SET"] = str(getattr(args, "emotion_set", "7") or "7")
            code = run_python_script(
                RELABEL_SCRIPT,
                [
                    "--target-per-emotion",
                    str(args.target_per_emotion),
                    "--candidate-batch",
                    str(args.candidate_batch),
                    "--no-seed",
                ],
            )
            if code == 0:
                run_quality_control_for(args, "basic")
            print(f"Promote/regroup finished with exit code {code}")
            continue
        if choice == "6":
            code = run_character_cluster(args)
            print(f"Character/speaker clustering finished with exit code {code}")
            continue
        if choice == "7":
            code = run_character_emotion_labeling(args)
            if code == 0:
                run_quality_control_for(args, "character")
            print(f"Character-aware 7-emotion labeling finished with exit code {code}")
            continue
        if choice == "8":
            code = run_cluster_plot(args)
            print(f"Speaker + emotion cluster analysis finished with exit code {code}")
            continue
        if choice == "q":
            return 0
        print("Unknown choice.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single runner for the emotion dataset workflow.",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "menu",
            "pipeline",
            "relabel",
            "review",
            "full",
            "promote-regroup",
            "character-cluster",
            "character-emotion",
            "cluster-analysis",
            "audit-workflow",
            "select-cluster-clean-target",
            "quality-control-basic",
            "quality-control-character",
            "refresh-quality-control",
            "select-cluster-clean-target-basic",
            "select-cluster-clean-target-character",
            "select-balanced-review-target-basic",
            "select-balanced-review-target-character",
        ],
        default="menu",
        help="Workflow mode. Default: %(default)s",
    )
    parser.add_argument(
        "--input",
        help="Input file or folder for pipeline steps; raw media source for character/speaker clustering.",
    )
    parser.add_argument(
        "--input-root",
        help="Raw input root for character/speaker clustering, or character dialogue root for character emotion labeling.",
    )
    parser.add_argument(
        "--output-root",
        help="Output root for character-aware workflow. Default: output/pipeline_workspace.",
    )
    parser.add_argument(
        "--expected-speakers",
        type=int,
        help="Expected speaker count for KMeans/fallback clustering.",
    )
    parser.add_argument(
        "--cluster-method",
        choices=["agglomerative", "hdbscan", "kmeans"],
        default="agglomerative",
        help="Speaker embedding clustering method. Default: %(default)s",
    )
    parser.add_argument(
        "--force-resplit",
        action="store_true",
        help="For character-cluster, clear existing japanese_dialogue clips and segment raw input again.",
    )
    parser.add_argument(
        "--force-character-map-template",
        action="store_true",
        help="For character-cluster, overwrite an existing character_map_template.json.",
    )
    parser.add_argument(
        "--skip-speaker-clustering",
        action="store_true",
        help="Compatibility argument; character-emotion expects existing speaker_clusters.",
    )
    parser.add_argument(
        "--skip-emotion-labeling",
        action="store_true",
        help="Compatibility argument for character-aware emotion pseudo-labeling.",
    )
    parser.add_argument(
        "--classification-output-name",
        default="character_emotion_classified",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--trained-model-weights",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--character-map",
        help="Path to edited character_map_template.json.",
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        default=True,
        help="Copy files into generated output folders. Default: enabled.",
    )
    parser.add_argument(
        "--final-output",
        default=os.path.join(BASE_DIR, "output", "final_character_emotion_dataset"),
        help="Final dataset output folder. Default: %(default)s",
    )
    parser.add_argument(
        "--skip-classify",
        action="store_true",
        help="Pass through to the pipeline step.",
    )
    parser.add_argument(
        "--no-reset-output",
        action="store_true",
        help="Pass through to the pipeline step.",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        help="Pass through to the pipeline step.",
    )
    parser.add_argument(
        "--classify-limit",
        type=int,
        help="Pass through to the pipeline step.",
    )
    parser.add_argument(
        "--transcript-backend",
        choices=["faster-whisper", "whisperx"],
        default="faster-whisper",
        help="Pass through to the segmentation/pipeline step. Default: %(default)s",
    )
    parser.add_argument(
        "--whisperx-model",
        default="base",
        help="Pass through to the pipeline step. Default: %(default)s",
    )
    parser.add_argument(
        "--speaker-diarize",
        action="store_true",
        help="Pass through to the pipeline step: cluster clips by speaker before emotion classification.",
    )
    parser.add_argument(
        "--diarization-model",
        default="pyannote/speaker-diarization-community-1",
        help="Pass through to the pipeline step. Default: %(default)s",
    )
    parser.add_argument(
        "--hf-token",
        help="Hugging Face token for --speaker-diarize. If omitted, HF_TOKEN or HUGGINGFACE_TOKEN is used.",
    )
    parser.add_argument(
        "--cut-mode",
        choices=["sentence", "silence", "fixed"],
        default="sentence",
        help="Pass through to the pipeline step. Default: %(default)s",
    )
    parser.add_argument(
        "--min-emotion-clip-sec",
        type=float,
        default=0.65,
        help="Pass through to the pipeline step. Default: %(default)s",
    )
    parser.add_argument(
        "--emotion-set",
        choices=["5", "7"],
        default="7",
        help="Use primary 5 emotions or the full 7-emotion set. Default: %(default)s",
    )
    parser.add_argument(
        "--auto-label-policy",
        choices=["strict", "balanced", "volume"],
        default="balanced",
        help="Emotion auto-accept policy: strict is cleanest, balanced is default, volume produces more weak labels. Default: %(default)s",
    )
    parser.add_argument(
        "--speaker-name-map",
        help='JSON map from diarization speaker IDs to character names, for example {"SPEAKER_00": "Nanoha"}.',
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
        help="Minimum dominant-speaker overlap ratio for option 8 speaker folders. Default: %(default)s",
    )
    parser.add_argument(
        "--speaker-max-other-overlap",
        type=float,
        default=0.20,
        help="Maximum non-dominant speaker overlap ratio for option 8 speaker folders. Default: %(default)s",
    )
    parser.add_argument(
        "--target-per-emotion",
        type=int,
        default=100,
        help="Target minimum approved clips per emotion for the relabel loop. Default: %(default)s",
    )
    parser.add_argument(
        "--allow-overlap-fill",
        action="store_true",
        help="For select-cluster-clean-target, allow overlap_review clips to fill shortages.",
    )
    parser.add_argument(
        "--allow-usable-fill",
        action="store_true",
        help="For select-cluster-clean-target, allow usable_review clips to fill shortages.",
    )
    parser.add_argument(
        "--cluster-clean-feature",
        choices=["multi_feature", "mfcc", "mfcc_paper9", "mfcc_separable", "mfcc_rich", "acoustic"],
        default="multi_feature",
        help=(
            "Feature mode for cluster-clean filtering. "
            "Use multi_feature for one clean dataset shared by all training methods, "
            "Use mfcc for notebook-style KMeans outlier filtering, "
            "or mfcc_separable for MFCC-compatible low-overlap sample selection."
        ),
    )
    parser.add_argument(
        "--candidate-batch",
        type=int,
        default=30,
        help="How many candidate review clips to refresh per emotion per relabel run. Default: %(default)s",
    )
    parser.add_argument(
        "--compare-rounds",
        type=int,
        default=3,
        help="Tournament passes across configured emotion labels when refreshing relabel candidates. Default: %(default)s",
    )
    parser.add_argument(
        "--candidate-min-score",
        type=float,
        default=0.05,
        help="Minimum score for a clip to enter the relabel tournament. Default: %(default)s",
    )
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key for optional LLM tie-breaking in the relabel step. If omitted, OPENAI_API_KEY can be used.",
    )
    parser.add_argument(
        "--openai-classifier-judge",
        action="store_true",
        help="Use OpenAI GPT during option 1 classification for uncertain clips.",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4.1",
        help="OpenAI model for optional relabel tie-breaking. Default: %(default)s",
    )
    parser.add_argument(
        "--openai-judge-limit",
        type=int,
        default=40,
        help="Maximum OpenAI judge calls per relabel run to control cost. Default: %(default)s",
    )
    parser.add_argument(
        "--openai-timeout-sec",
        type=int,
        default=120,
        help="Timeout per OpenAI judge call in seconds. Default: %(default)s",
    )
    parser.add_argument(
        "--review-queue",
        help="Specific _manual_review_queue.csv for manual review.",
    )
    parser.add_argument(
        "--no-autoplay",
        action="store_true",
        help="Pass through to manual review.",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="For --mode full, skip the manual review step.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check planned option 7/8 actions without model inference, copying, or plotting.",
    )
    parser.add_argument(
        "--extract-source",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--extract-output",
        default=EXTRACTED_AUDIO_DIR,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--audio-source",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cluster-plot-limit",
        type=int,
        default=500,
        help="Max clips per emotion for PCA cluster plot. 0 = no limit. Default: %(default)s",
    )
    parser.add_argument(
        "--review-sample-per-class",
        type=int,
        default=50,
        help="How many clips per class to copy into _balanced_review_sample. Default: %(default)s",
    )
    parser.add_argument(
        "--max-review-total",
        type=int,
        default=500,
        help="Maximum total clips to copy into _review_shortlist. Default: %(default)s",
    )
    parser.add_argument(
        "--include-rejected-review",
        action="store_true",
        help="Allow rejected clips into the review shortlist. Default: disabled.",
    )
    parser.add_argument(
        "--cluster-plot-title",
        default="Anime emotion clusters",
        help="Title for PCA emotion cluster plot. Default: %(default)s",
    )
    parser.add_argument(
        "--cluster-embedding-backend",
        choices=["model", "acoustic"],
        default="acoustic",
        help="Embedding backend for cluster plots. Use model for Hugging Face embeddings. Default: %(default)s",
    )
    parser.add_argument(
        "--cluster-embedding-model",
        default="facebook/wav2vec2-base",
        help="Hugging Face audio model used for cluster plot embeddings. Default: %(default)s",
    )
    parser.add_argument(
        "--cluster-embedding-device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for model cluster embeddings. Default: %(default)s",
    )
    parser.add_argument(
        "--cluster-feature",
        choices=["acoustic", "japanese_ssl_embedding"],
        default="acoustic",
        help="Feature method for option 8 cluster analysis. Default: %(default)s",
    )
    parser.add_argument(
        "--skip-tsne",
        action="store_true",
        help="Skip t-SNE during option 8 cluster analysis.",
    )
    parser.add_argument(
        "--skip-umap",
        action="store_true",
        help="Skip UMAP during option 8 cluster analysis.",
    )
    parser.add_argument(
        "--plot-global-only",
        action="store_true",
        help="Only regenerate global cluster analysis.",
    )
    parser.add_argument(
        "--plot-shortlist-only",
        action="store_true",
        help="Only regenerate review-shortlist cluster analysis.",
    )
    parser.add_argument(
        "--cluster-reducer",
        choices=["pca", "umap", "tsne"],
        default="pca",
        help="2D projection for cluster plots. Try umap for clearer clusters. Default: %(default)s",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.mode == "pipeline":
        code = run_pipeline(args)
        if code == 0:
            run_quality_control_for(args, "basic")
        return code
    if args.mode == "relabel":
        code = run_relabel(args)
        if code == 0:
            run_quality_control_for(args, "basic")
        return code
    if args.mode == "review":
        code = run_review(args)
        if code == 0:
            run_refresh_quality_control(args)
        return code
    if args.mode == "full":
        code = run_full_loop(args)
        if code == 0:
            run_quality_control_for(args, "basic")
        return code
    if args.mode == "promote-regroup":
        os.environ["EMOTION_SET"] = str(getattr(args, "emotion_set", "7") or "7")
        code = run_python_script(
            RELABEL_SCRIPT,
            [
                "--target-per-emotion",
                str(args.target_per_emotion),
                "--candidate-batch",
                str(args.candidate_batch),
                "--no-seed",
            ],
        )
        if code == 0:
            run_quality_control_for(args, "basic")
        return code
    if args.mode == "character-cluster":
        return run_character_cluster(args)
    if args.mode == "character-emotion":
        code = run_character_emotion_labeling(args)
        if code == 0:
            run_quality_control_for(args, "character")
        return code
    if args.mode == "cluster-analysis":
        return run_cluster_plot(args)
    if args.mode == "audit-workflow":
        return run_audit_workflow(args)
    if args.mode == "select-cluster-clean-target":
        return run_cluster_clean_target(args)
    if args.mode == "quality-control-basic":
        return run_quality_control_for(args, "basic")
    if args.mode == "quality-control-character":
        return run_quality_control_for(args, "character")
    if args.mode == "refresh-quality-control":
        return run_refresh_quality_control(args)
    if args.mode == "select-cluster-clean-target-basic":
        return run_quality_control_for(args, "basic")
    if args.mode == "select-cluster-clean-target-character":
        return run_quality_control_for(args, "character")
    if args.mode == "select-balanced-review-target-basic":
        return run_quality_control_for(args, "basic")
    if args.mode == "select-balanced-review-target-character":
        return run_quality_control_for(args, "character")
    return menu(args)


if __name__ == "__main__":
    raise SystemExit(main())
