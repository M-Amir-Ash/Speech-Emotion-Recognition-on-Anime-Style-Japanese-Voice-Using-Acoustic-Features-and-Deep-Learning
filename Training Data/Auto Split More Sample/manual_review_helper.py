import argparse
import csv
import os
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import winsound
except Exception:
    winsound = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_WORKSPACE_ROOT = os.path.join(BASE_DIR, "output", "pipeline_workspace")

CANONICAL_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Pleasant_Surprise",
    "Sad",
]
REJECT_LABEL = "Rejected"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compact_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").replace("\t", " ").split())


def safe_copy_path(folder: str, file_name: str) -> str:
    ensure_dir(folder)
    candidate = os.path.join(folder, file_name)
    if not os.path.exists(candidate):
        return candidate

    stem, ext = os.path.splitext(file_name)
    copy_idx = 1
    while True:
        candidate = os.path.join(folder, f"{stem}_copy_{copy_idx:02d}{ext}")
        if not os.path.exists(candidate):
            return candidate
        copy_idx += 1


def find_latest_manual_queue(root_dir: str) -> str:
    candidates = []
    for root, _, files in os.walk(root_dir):
        if "_manual_review_queue.csv" in files:
            path = os.path.join(root, "_manual_review_queue.csv")
            candidates.append((os.path.getmtime(path), path))

    if not candidates:
        raise FileNotFoundError(f"No _manual_review_queue.csv found under {root_dir}")

    candidates.sort(reverse=True)
    return candidates[0][1]


def load_queue(queue_csv: str) -> List[Dict[str, str]]:
    with open(queue_csv, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_progress(progress_csv: str) -> Dict[str, Dict[str, str]]:
    if not os.path.isfile(progress_csv):
        return {}
    with open(progress_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["source_file"]: row for row in rows if row.get("source_file")}


def save_progress(progress_csv: str, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "source_file",
        "pred_folder",
        "top_label",
        "top_score",
        "top_gap",
        "transcript",
        "review_decision",
        "review_note",
        "reviewed_at",
        "review_output_file",
    ]
    with open(progress_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_play(path: str) -> None:
    if winsound is None:
        print(f"Playback unavailable in this environment: {path}")
        return
    winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)


def stop_playback() -> None:
    if winsound is None:
        return
    winsound.PlaySound(None, winsound.SND_PURGE)


def resolve_review_source(row: Dict[str, str]) -> str:
    for key in ("source_file", "output_file"):
        path = row.get(key, "").strip()
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No existing audio file found for row: {row.get('source_file', '')}")


def copy_reviewed_file(src_path: str, review_root: str, decision: str) -> str:
    dst_folder = os.path.join(review_root, decision)
    dst_path = safe_copy_path(dst_folder, os.path.basename(src_path))
    shutil.copy2(src_path, dst_path)
    return dst_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive manual review helper for _manual_review_queue.csv",
    )
    parser.add_argument(
        "--queue",
        help="Path to a specific _manual_review_queue.csv. Defaults to the latest one under output/pipeline_workspace.",
    )
    parser.add_argument(
        "--workspace-root",
        default=PIPELINE_WORKSPACE_ROOT,
        help="Root folder to search when --queue is omitted.",
    )
    parser.add_argument(
        "--no-autoplay",
        action="store_true",
        help="Do not auto-play each clip when it is shown.",
    )
    return parser.parse_args()


def prompt_command() -> str:
    print("Commands: 1 Angry | 2 Disgust | 3 Fear | 4 Happy | 5 Neutral | 6 Pleasant_Surprise | 7 Sad")
    print("          r Rejected | p Play again | o Open file | n Note | s Skip | q Quit")
    return input("> ").strip()


def main() -> None:
    args = parse_args()
    queue_csv = os.path.abspath(args.queue) if args.queue else find_latest_manual_queue(args.workspace_root)
    queue_rows = load_queue(queue_csv)
    queue_dir = os.path.dirname(queue_csv)
    review_root = os.path.join(queue_dir, "manual_review_sorted")
    progress_csv = os.path.join(queue_dir, "_manual_review_progress.csv")

    progress_map = load_progress(progress_csv)
    progress_rows = list(progress_map.values())

    pending_rows = [row for row in queue_rows if row.get("source_file") not in progress_map]
    print(f"Queue: {queue_csv}")
    print(f"Pending clips: {len(pending_rows)} / {len(queue_rows)}")
    print(f"Review output: {review_root}")
    print(f"Progress file: {progress_csv}")

    label_map = {
        "1": "Angry",
        "2": "Disgust",
        "3": "Fear",
        "4": "Happy",
        "5": "Neutral",
        "6": "Pleasant_Surprise",
        "7": "Sad",
        "r": REJECT_LABEL,
    }

    for idx, row in enumerate(pending_rows, start=1):
        source_file = row.get("source_file", "")
        transcript = compact_text(row.get("transcript", ""))
        top_label = row.get("top_label", "")
        top_score = row.get("top_score", "")
        top_gap = row.get("top_gap", "")
        review_reason = row.get("review_reason", "")
        src_path = resolve_review_source(row)

        print("")
        print(f"[{idx}/{len(pending_rows)}] {os.path.basename(src_path)}")
        print(f"Predicted: {top_label} | score={top_score} | gap={top_gap} | reason={review_reason}")
        print(f"Transcript: {transcript}")
        print(f"Source: {src_path}")

        if not args.no_autoplay:
            maybe_play(src_path)

        note = ""
        while True:
            cmd = prompt_command().lower()
            if cmd in label_map:
                stop_playback()
                decision = label_map[cmd]
                reviewed_path = copy_reviewed_file(src_path, review_root, decision)
                progress_row = {
                    "source_file": source_file,
                    "pred_folder": row.get("pred_folder", ""),
                    "top_label": top_label,
                    "top_score": top_score,
                    "top_gap": top_gap,
                    "transcript": transcript,
                    "review_decision": decision,
                    "review_note": note,
                    "reviewed_at": datetime.now().isoformat(timespec="seconds"),
                    "review_output_file": reviewed_path,
                }
                progress_map[source_file] = progress_row
                progress_rows = list(progress_map.values())
                save_progress(progress_csv, progress_rows)
                print(f"Saved: {decision} -> {reviewed_path}")
                break
            if cmd == "p":
                maybe_play(src_path)
                continue
            if cmd == "o":
                try:
                    os.startfile(src_path)
                except Exception as e:
                    print(f"Open failed: {e}")
                continue
            if cmd == "n":
                note = input("Note: ").strip()
                continue
            if cmd == "s":
                stop_playback()
                print("Skipped.")
                break
            if cmd == "q":
                stop_playback()
                save_progress(progress_csv, list(progress_map.values()))
                print("Stopped. Progress saved.")
                return
            print("Unknown command.")

    save_progress(progress_csv, list(progress_map.values()))
    print("Review queue completed.")


if __name__ == "__main__":
    main()
