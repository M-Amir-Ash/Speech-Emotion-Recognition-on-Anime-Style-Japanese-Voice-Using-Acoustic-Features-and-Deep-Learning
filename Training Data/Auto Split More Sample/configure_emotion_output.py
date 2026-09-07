import argparse
import csv
import os
import shutil
from typing import Dict, List


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "emotion_classifier_config.json")
DEFAULT_CLASSIFIED_ROOT = os.path.join(
    BASE_DIR,
    "output",
    "pipeline_workspace",
    "multimodal_emotion_classified",
)

PRIMARY_5_EMOTIONS = [
    "Angry",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
]

EXTENDED_7_EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Pleasant_Surprise",
    "Sad",
]


def configured_emotion_set() -> str:
    env_value = os.environ.get("EMOTION_SET", "").strip()
    if env_value:
        return "7" if env_value == "7" else "5"
    try:
        import json

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        value = str(cfg.get("classification", {}).get("emotion_set", "5")).strip()
        return "7" if value == "7" else "5"
    except Exception:
        return "5"


EMOTIONS = EXTENDED_7_EMOTIONS if configured_emotion_set() == "7" else PRIMARY_5_EMOTIONS


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def wav_count(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    return sum(1 for name in os.listdir(path) if name.lower().endswith(".wav"))


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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


def clear_directory_files(folder: str) -> None:
    ensure_dir(folder)
    for name in os.listdir(folder):
        full_path = os.path.join(folder, name)
        if os.path.isfile(full_path):
            os.remove(full_path)


def ensure_emotion_folders(classified_root: str) -> None:
    ensure_dir(classified_root)
    for emotion in EMOTIONS:
        ensure_dir(os.path.join(classified_root, emotion))
    ensure_dir(os.path.join(classified_root, "Needs_Review"))
    ensure_dir(os.path.join(classified_root, "Rejected_LowQuality"))


def current_counts(classified_root: str) -> Dict[str, int]:
    return {emotion: wav_count(os.path.join(classified_root, emotion)) for emotion in EMOTIONS}


def review_counts(classified_root: str) -> Dict[str, int]:
    return {
        "Needs_Review": wav_count(os.path.join(classified_root, "Needs_Review")),
        "Rejected_LowQuality": wav_count(os.path.join(classified_root, "Rejected_LowQuality")),
    }


def clear_review_seed_folder(folder: str) -> None:
    ensure_dir(folder)
    for name in os.listdir(folder):
        full_path = os.path.join(folder, name)
        if os.path.isfile(full_path):
            os.remove(full_path)


def build_candidate_pools(classified_root: str) -> Dict[str, List[Dict[str, str]]]:
    queue_csv = os.path.join(classified_root, "_manual_review_queue.csv")
    manifest_csv = os.path.join(classified_root, "_multimodal_emotion_manifest.csv")

    queue_rows = read_csv_rows(queue_csv)
    manifest_rows = read_csv_rows(manifest_csv)
    queue_source_files = {row.get("source_file", "") for row in queue_rows}

    all_rows = queue_rows + [
        row for row in manifest_rows
        if row.get("source_file", "") and row.get("source_file", "") not in queue_source_files
    ]

    pools: Dict[str, List[Dict[str, str]]] = {emotion: [] for emotion in EMOTIONS}
    for row in all_rows:
        top_label = row.get("top_label", "")
        second_label = row.get("second_label", "")
        if top_label in pools:
            pools[top_label].append(row)
        if second_label in pools and second_label != top_label:
            pools[second_label].append(row)

    def row_rank(row: Dict[str, str], emotion: str) -> tuple:
        top_label = row.get("top_label", "")
        top_score = float(row.get("top_score", 0.0) or 0.0)
        second_score = float(row.get("second_score", 0.0) or 0.0)
        top_gap = float(row.get("top_gap", 0.0) or 0.0)
        transcript_rel = float(row.get("transcript_reliability", 0.0) or 0.0)
        is_top = 1 if top_label == emotion else 0
        score = top_score if is_top else second_score
        return (is_top, score, top_gap, transcript_rel)

    for emotion in EMOTIONS:
        dedup = {}
        for row in pools[emotion]:
            source = row.get("source_file", "")
            if source and source not in dedup:
                dedup[source] = row
        pools[emotion] = sorted(
            dedup.values(),
            key=lambda row: row_rank(row, emotion),
            reverse=True,
        )

    return pools


def resolve_existing_audio_path(row: Dict[str, str]) -> str:
    for key in ("output_file", "source_file"):
        path = row.get(key, "").strip()
        if path and os.path.isfile(path):
            return path
    return ""


def regroup_all_predictions(classified_root: str) -> Dict[str, int]:
    manifest_csv = os.path.join(classified_root, "_multimodal_emotion_manifest.csv")
    regroup_root = os.path.join(classified_root, "_regrouped_top_label")
    rows = read_csv_rows(manifest_csv)

    ensure_dir(regroup_root)
    for emotion in EMOTIONS:
        clear_directory_files(os.path.join(regroup_root, emotion))

    report_rows: List[Dict[str, str]] = []
    counts = {emotion: 0 for emotion in EMOTIONS}

    for row in rows:
        top_label = row.get("top_label", "")
        if top_label not in counts:
            continue

        src_path = resolve_existing_audio_path(row)
        if not src_path:
            continue

        dst_dir = os.path.join(regroup_root, top_label)
        dst_path = safe_copy_path(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        counts[top_label] += 1

        report_rows.append({
            "emotion": top_label,
            "source_file": src_path,
            "copied_to": dst_path,
            "pred_folder": row.get("pred_folder", ""),
            "top_label": row.get("top_label", ""),
            "top_score": row.get("top_score", ""),
            "second_label": row.get("second_label", ""),
            "second_score": row.get("second_score", ""),
            "top_gap": row.get("top_gap", ""),
            "review_status": row.get("review_status", ""),
            "transcript": row.get("transcript", ""),
        })

    report_csv = os.path.join(regroup_root, "_regrouped_summary.csv")
    if report_rows:
        with open(report_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(report_rows)

    return counts


def seed_empty_emotion_candidates(classified_root: str, per_emotion: int) -> Dict[str, int]:
    counts = current_counts(classified_root)
    pools = build_candidate_pools(classified_root)
    seeded_counts: Dict[str, int] = {}

    for emotion, current in counts.items():
        if current > 0:
            seeded_counts[emotion] = 0
            continue

        review_dir = os.path.join(classified_root, "_candidate_review", emotion)
        clear_review_seed_folder(review_dir)
        created = 0
        report_rows: List[Dict[str, str]] = []

        for row in pools.get(emotion, []):
            src_path = row.get("output_file", "").strip()
            if not src_path or not os.path.isfile(src_path):
                src_path = row.get("source_file", "").strip()
            if not src_path or not os.path.isfile(src_path):
                continue

            dst_path = safe_copy_path(review_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            report_rows.append({
                "emotion": emotion,
                "source_file": src_path,
                "copied_to": dst_path,
                "top_label": row.get("top_label", ""),
                "top_score": row.get("top_score", ""),
                "second_label": row.get("second_label", ""),
                "second_score": row.get("second_score", ""),
                "top_gap": row.get("top_gap", ""),
                "transcript": row.get("transcript", ""),
            })
            created += 1
            if created >= per_emotion:
                break

        if report_rows:
            report_csv = os.path.join(review_dir, "_candidate_report.csv")
            with open(report_csv, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
                writer.writeheader()
                writer.writerows(report_rows)

        seeded_counts[emotion] = created

    return seeded_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configure and inspect the multimodal_emotion_classified folder.",
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_CLASSIFIED_ROOT,
        help="Path to multimodal_emotion_classified. Default: %(default)s",
    )
    parser.add_argument(
        "--seed-empty",
        action="store_true",
        help="Explicitly seed empty emotion folders with review candidates.",
    )
    parser.add_argument(
        "--no-seed-empty",
        action="store_true",
        help="Do not create candidate review clips for empty emotion folders.",
    )
    parser.add_argument(
        "--per-emotion",
        type=int,
        default=25,
        help="Number of candidate clips to copy for each empty emotion when --seed-empty is used.",
    )
    parser.add_argument(
        "--no-regroup",
        action="store_true",
        help="Do not create the _regrouped_top_label folder from all manifest predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classified_root = os.path.abspath(args.root)

    ensure_emotion_folders(classified_root)
    counts = current_counts(classified_root)
    review = review_counts(classified_root)
    should_seed = args.seed_empty or not args.no_seed_empty
    should_regroup = not args.no_regroup

    print(f"Configured root: {classified_root}")
    print(f"\nCurrent {configured_emotion_set()}-emotion counts:")
    for emotion in EMOTIONS:
        print(f"{emotion}: {counts[emotion]}")

    empty = [emotion for emotion, count in counts.items() if count == 0]
    print(f"\nEmpty emotion folders: {empty if empty else 'none'}")
    print("\nReview backlog:")
    print(f"Needs_Review: {review['Needs_Review']}")
    print(f"Rejected_LowQuality: {review['Rejected_LowQuality']}")

    if should_regroup:
        regrouped = regroup_all_predictions(classified_root)
        print("\nRegrouped all predictions into:")
        print(f"{os.path.join(classified_root, '_regrouped_top_label')}")
        for emotion in EMOTIONS:
            print(f"{emotion}: {regrouped.get(emotion, 0)}")

    if empty and should_seed:
        seeded = seed_empty_emotion_candidates(classified_root, args.per_emotion)
        print("\nCandidate review copies created:")
        for emotion in EMOTIONS:
            print(f"{emotion}: {seeded.get(emotion, 0)}")
        print(
            "\nReview folders are under: "
            f"{os.path.join(classified_root, '_candidate_review')}"
        )
    elif empty:
        print("\nCandidate review copy step skipped.")


if __name__ == "__main__":
    main()
