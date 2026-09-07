import argparse
import csv
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = BASE_DIR / "output" / "pipeline_workspace"
SELF_TRAINING_ROOT = BASE_DIR / "output" / "self_training_dataset"
DEFAULT_FINAL_OUTPUT = BASE_DIR / "output" / "final_character_emotion_dataset"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant_Surprise", "Sad"]
CSV_ENCODING = "utf-8-sig"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_unique(src: Path, dst_dir: Path) -> Path:
    ensure_dir(dst_dir)
    candidate = dst_dir / src.name
    if not candidate.exists():
        shutil.copy2(src, candidate)
        return candidate
    idx = 2
    while True:
        candidate = dst_dir / f"{src.stem}__dup{idx}{src.suffix}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
        idx += 1


def read_rows(path: Path) -> List[dict]:
    with path.open("r", encoding=CSV_ENCODING, newline="") as f:
        return list(csv.DictReader(f))


def approved_manifest_candidates(workspace: Path) -> List[dict]:
    candidates = []
    for path in [
        workspace / "character_emotion_classified" / "_manual_approved.csv",
        workspace / "character_emotion_classified" / "_approved_manifest.csv",
        workspace / "multimodal_emotion_classified" / "_approved_manifest.csv",
    ]:
        if path.is_file():
            candidates.extend(read_rows(path))
    return candidates


def source_path(row: dict) -> Path:
    for key in ("output_file", "global_output_file", "file_path", "source_file"):
        value = str(row.get(key, "")).strip()
        if value:
            path = Path(value)
            if path.is_file():
                return path
    return Path("")


def row_label(row: dict) -> str:
    return str(row.get("top_label") or row.get("emotion_label") or row.get("label") or "").strip()


def build_self_training_dataset(workspace: Path, output_root: Path) -> Dict[str, int]:
    manifest = workspace / "character_emotion_classified" / "_character_emotion_manifest.csv"
    if not manifest.is_file():
        print(f"First-pass character emotion manifest not found: {manifest}")
        return {}
    if output_root.exists():
        shutil.rmtree(output_root)
    for emotion in EMOTIONS:
        ensure_dir(output_root / emotion)

    rows = [
        row
        for row in read_rows(manifest)
        if row.get("confidence_status") == "auto_accept" and row_label(row) in EMOTIONS
    ]
    rows.extend(row for row in approved_manifest_candidates(workspace) if row_label(row) in EMOTIONS)

    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        label = row_label(row)
        src = source_path(row)
        if label not in EMOTIONS or not src.is_file():
            continue
        copy_unique(src, output_root / label)
        counts[label] += 1

    present = [label for label, count in counts.items() if count > 0]
    if len(present) < 2:
        print("Not enough class diversity for self-training. Need at least 2 emotion classes with enough clips.")
        return {}
    for label in EMOTIONS:
        if counts[label] < 10:
            print(f"Class {label} has low sample count. Model may be unstable.")
    print(f"Self-training dataset saved: {output_root}")
    for label in EMOTIONS:
        print(f"  {label}: {counts[label]}")
    return dict(counts)


def run_command(args: List[str]) -> int:
    print("Running: " + " ".join(args), flush=True)
    return subprocess.call([sys.executable] + args, cwd=BASE_DIR)


def main() -> int:
    parser = argparse.ArgumentParser(description="Iterative anime-domain emotion self-training.")
    parser.add_argument("--output-root", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--final-output", default=str(DEFAULT_FINAL_OUTPUT))
    parser.add_argument("--copy-files", action="store_true", default=True)
    args = parser.parse_args()

    workspace = Path(args.output_root).resolve()
    counts = build_self_training_dataset(workspace, SELF_TRAINING_ROOT)
    if not counts:
        return 1

    code = run_command(["train_mel_cnn_7emotion.py", "--input-root", str(SELF_TRAINING_ROOT)])
    if code != 0:
        return code
    code = run_command(["train_emotion_embedding_classifier.py", "--input-root", str(SELF_TRAINING_ROOT)])
    if code != 0:
        return code

    code = run_command(
        [
            "character_emotion_pseudolabel.py",
            "--input-root",
            str(workspace / "character_dialogue"),
            "--output-root",
            str(workspace),
            "--classification-output-name",
            "character_emotion_classified_second_pass",
            "--trained-model-weights",
        ]
    )
    if code != 0:
        return code

    code = run_command(
        [
            "build_final_emotion_dataset.py",
            "--output-root",
            str(workspace),
            "--source-output-name",
            "character_emotion_classified_second_pass",
            "--final-output",
            str(Path(args.final_output).resolve()),
        ]
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
