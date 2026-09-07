import argparse
import csv
import json
import os
import shutil
import socket
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

import configure_emotion_output as ceo


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CLASSIFIED_ROOT = ceo.DEFAULT_CLASSIFIED_ROOT
EMOTIONS = ceo.EMOTIONS


def ensure_loop_folders(classified_root: str) -> None:
    for parent in ("_approved", "_candidate_review", "manual_review_sorted"):
        for emotion in EMOTIONS:
            ceo.ensure_dir(os.path.join(classified_root, parent, emotion))


def read_existing_names(folder: str) -> set[str]:
    if not os.path.isdir(folder):
        return set()
    return {name.lower() for name in os.listdir(folder) if name.lower().endswith(".wav")}


def promote_from_folder(src_dir: str, dst_dir: str) -> List[Dict[str, str]]:
    ceo.ensure_dir(src_dir)
    ceo.ensure_dir(dst_dir)
    promoted: List[Dict[str, str]] = []
    existing = read_existing_names(dst_dir)

    for name in sorted(os.listdir(src_dir)):
        if not name.lower().endswith(".wav"):
            continue
        src_path = os.path.join(src_dir, name)
        if not os.path.isfile(src_path):
            continue
        if name.lower() in existing:
            continue

        dst_path = os.path.join(dst_dir, name)
        shutil.copy2(src_path, dst_path)
        promoted.append({
            "source_file": src_path,
            "promoted_to": dst_path,
        })
        existing.add(name.lower())

    return promoted


def promote_staged_labels(classified_root: str) -> Dict[str, int]:
    approved_root = os.path.join(classified_root, "_approved")
    manual_root = os.path.join(classified_root, "manual_review_sorted")
    promotion_log: List[Dict[str, str]] = []
    counts: Dict[str, int] = {emotion: 0 for emotion in EMOTIONS}

    for emotion in EMOTIONS:
        dst_dir = os.path.join(classified_root, emotion)
        for source_root, source_tag in (
            (approved_root, "approved"),
            (manual_root, "manual_review_sorted"),
        ):
            src_dir = os.path.join(source_root, emotion)
            if not os.path.isdir(src_dir):
                continue
            promoted = promote_from_folder(src_dir, dst_dir)
            counts[emotion] += len(promoted)
            for row in promoted:
                row["emotion"] = emotion
                row["source_tag"] = source_tag
                promotion_log.append(row)

    if promotion_log:
        log_path = os.path.join(classified_root, "_promotion_log.csv")
        with open(log_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["emotion", "source_tag", "source_file", "promoted_to"],
            )
            writer.writeheader()
            writer.writerows(promotion_log)

    return counts


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def score_column(emotion: str) -> str:
    return f"final_score_{emotion.lower()}"


def emotion_scores(row: Dict[str, str]) -> Dict[str, float]:
    scores = {
        emotion: parse_float(row.get(score_column(emotion)), 0.0)
        for emotion in EMOTIONS
    }
    if any(score > 0.0 for score in scores.values()):
        return scores

    top_label = row.get("top_label", "")
    second_label = row.get("second_label", "")
    if top_label in scores:
        scores[top_label] = parse_float(row.get("top_score"), 0.0)
    if second_label in scores:
        scores[second_label] = parse_float(row.get("second_score"), 0.0)
    return scores


def ranked_emotions(row: Dict[str, str]) -> List[Tuple[str, float]]:
    scores = emotion_scores(row)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def load_candidate_rows(classified_root: str) -> List[Dict[str, str]]:
    queue_csv = os.path.join(classified_root, "_manual_review_queue.csv")
    manifest_csv = os.path.join(classified_root, "_multimodal_emotion_manifest.csv")

    queue_rows = ceo.read_csv_rows(queue_csv)
    manifest_rows = ceo.read_csv_rows(manifest_csv)
    queue_source_files = {row.get("source_file", "") for row in queue_rows}

    rows = queue_rows + [
        row for row in manifest_rows
        if row.get("source_file", "") and row.get("source_file", "") not in queue_source_files
    ]

    dedup: Dict[str, Dict[str, str]] = {}
    for row in rows:
        src_path = ceo.resolve_existing_audio_path(row)
        if not src_path:
            continue
        if src_path not in dedup:
            dedup[src_path] = row

    def row_priority(row: Dict[str, str]) -> Tuple[float, float, float]:
        ranked = ranked_emotions(row)
        top_score = ranked[0][1] if ranked else 0.0
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        gap = top_score - second_score
        transcript_rel = parse_float(row.get("transcript_reliability"), 0.0)
        return (top_score, gap, transcript_rel)

    return sorted(dedup.values(), key=row_priority, reverse=True)


def assignment_reason(round_idx: int, emotion: str, ranked: List[Tuple[str, float]]) -> str:
    top_label, top_score = ranked[0] if ranked else ("", 0.0)
    second_label, second_score = ranked[1] if len(ranked) > 1 else ("", 0.0)
    gap = top_score - second_score
    if round_idx == 1:
        return f"round_1_top_winner:{emotion}:score={top_score:.4f}:gap={gap:.4f}"
    if round_idx == 2:
        return f"round_2_top2_compare:{emotion}:top={top_label}:{top_score:.4f}:second={second_label}:{second_score:.4f}"
    return f"round_{round_idx}_all_emotion_compare:{emotion}:top={top_label}:{top_score:.4f}:second={second_label}:{second_score:.4f}"


def extract_response_text(response: Dict[str, object]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: List[str] = []
    for item in response.get("output", []) if isinstance(response.get("output"), list) else []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) if isinstance(item.get("content"), list) else []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


def openai_emotion_judge(
    row: Dict[str, str],
    ranked: List[Tuple[str, float]],
    api_key: str,
    model: str,
    timeout_sec: int = 120,
) -> Dict[str, str]:
    transcript = (row.get("transcript") or "").strip()
    scores = {emotion: round(score, 6) for emotion, score in ranked}
    prompt = {
        "task": "Choose the best emotion label for one Japanese anime dialogue clip.",
        "allowed_labels": EMOTIONS + ["Unknown"],
        "rule": (
            "Use the transcript and classifier scores together. "
            "If the evidence is weak, ambiguous, singing, sound effects, or not emotional, choose Unknown. "
            "Return compact JSON only."
        ),
        "transcript": transcript[:1200],
        "classifier_scores": scores,
        "current_top_label": row.get("top_label", ""),
        "current_second_label": row.get("second_label", ""),
        "output_schema": {"emotion": "one allowed label", "confidence": "0.0-1.0", "reason": "short"},
    }
    request_body = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": "You are a strict emotion-label judge for speech dataset curation. Output JSON only.",
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False),
            },
        ],
        "text": {"format": {"type": "json_object"}},
        "max_output_tokens": 160,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=max(30, int(timeout_sec))) as resp:
            response = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI HTTP {e.code}: {detail[:500]}") from e

    text = extract_response_text(response)
    parsed = json.loads(text)
    emotion = str(parsed.get("emotion", "")).strip()
    confidence = parse_float(parsed.get("confidence"), 0.0)
    reason = str(parsed.get("reason", "")).strip()
    return {
        "llm_emotion": emotion,
        "llm_confidence": f"{confidence:.3f}",
        "llm_reason": reason[:240],
        "llm_model": model,
    }


def should_use_openai_judge(round_idx: int, ranked: List[Tuple[str, float]]) -> bool:
    if len(ranked) < 2:
        return False
    gap = ranked[0][1] - ranked[1][1]
    return round_idx > 1 or gap < 0.08


def seed_candidates_for_deficits(
    classified_root: str,
    target_per_emotion: int,
    candidate_batch: int,
    compare_rounds: int = 3,
    candidate_min_score: float = 0.05,
    openai_api_key: Optional[str] = None,
    openai_model: str = "gpt-4.1",
    openai_judge_limit: int = 0,
    openai_timeout_sec: int = 120,
) -> Dict[str, int]:
    counts = ceo.current_counts(classified_root)
    candidate_rows = load_candidate_rows(classified_root)
    seeded_counts: Dict[str, int] = {emotion: 0 for emotion in EMOTIONS}
    limits: Dict[str, int] = {}
    report_rows_by_emotion: Dict[str, List[Dict[str, str]]] = {emotion: [] for emotion in EMOTIONS}

    for emotion in EMOTIONS:
        review_dir = os.path.join(classified_root, "_candidate_review", emotion)
        ceo.clear_review_seed_folder(review_dir)

        deficit = max(0, int(target_per_emotion) - int(counts.get(emotion, 0)))
        limits[emotion] = min(int(candidate_batch), deficit)

    used_sources = set()
    llm_calls = 0
    llm_disabled = False
    llm_errors = 0
    rounds = max(1, int(compare_rounds))
    for round_idx in range(1, rounds + 1):
        progress = False
        for row in candidate_rows:
            src_path = ceo.resolve_existing_audio_path(row)
            if not src_path or src_path in used_sources:
                continue

            ranked = ranked_emotions(row)
            if not ranked:
                continue
            if ranked[0][1] < float(candidate_min_score):
                continue

            if round_idx == 1:
                candidate_ranked = ranked[:1]
                if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.04:
                    continue
            elif round_idx == 2:
                candidate_ranked = ranked[:2]
            else:
                candidate_ranked = ranked

            llm_result: Dict[str, str] = {}
            llm_chosen = False
            if (
                openai_api_key
                and not llm_disabled
                and llm_calls < max(0, int(openai_judge_limit))
                and should_use_openai_judge(round_idx, ranked)
            ):
                try:
                    llm_calls += 1
                    llm_result = openai_emotion_judge(
                        row,
                        ranked,
                        openai_api_key,
                        openai_model,
                        timeout_sec=openai_timeout_sec,
                    )
                    llm_errors = 0
                except (TimeoutError, socket.timeout) as e:
                    llm_errors += 1
                    print(
                        f"OpenAI LLM judge timed out ({llm_errors}/3); falling back for this clip.",
                        flush=True,
                    )
                    if llm_errors >= 3:
                        llm_disabled = True
                        print("OpenAI LLM judge disabled after repeated timeouts.", flush=True)
                except Exception as e:
                    llm_errors += 1
                    print(
                        f"OpenAI LLM judge error ({llm_errors}/3): {type(e).__name__}: {e}",
                        flush=True,
                    )
                    if llm_errors >= 3:
                        llm_disabled = True
                        print("OpenAI LLM judge disabled after repeated errors.", flush=True)

            chosen_emotion: Optional[str] = None
            chosen_score = 0.0
            llm_emotion = llm_result.get("llm_emotion", "")
            llm_confidence = parse_float(llm_result.get("llm_confidence"), 0.0)
            ranked_labels = [emotion for emotion, _ in ranked[:3]]
            if (
                llm_emotion in limits
                and llm_confidence >= 0.55
                and seeded_counts[llm_emotion] < limits[llm_emotion]
                and (llm_emotion in ranked_labels or emotion_scores(row).get(llm_emotion, 0.0) >= float(candidate_min_score))
            ):
                chosen_emotion = llm_emotion
                chosen_score = emotion_scores(row).get(llm_emotion, 0.0)
                llm_chosen = True

            for emotion, score in candidate_ranked:
                if chosen_emotion:
                    break
                if emotion not in limits or seeded_counts[emotion] >= limits[emotion]:
                    continue
                if score < float(candidate_min_score):
                    continue
                chosen_emotion = emotion
                chosen_score = score
                break

            if not chosen_emotion:
                continue

            review_dir = os.path.join(classified_root, "_candidate_review", chosen_emotion)
            dst_path = ceo.safe_copy_path(review_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            used_sources.add(src_path)
            seeded_counts[chosen_emotion] += 1
            progress = True

            report_row = {
                "emotion": chosen_emotion,
                "compare_round": str(round_idx),
                "assignment_reason": (
                    f"openai_judge:{chosen_emotion}:confidence={llm_confidence:.3f}"
                    if llm_chosen
                    else assignment_reason(round_idx, chosen_emotion, ranked)
                ),
                "source_file": src_path,
                "copied_to": dst_path,
                "assigned_score": f"{chosen_score:.6f}",
                "llm_emotion": llm_result.get("llm_emotion", ""),
                "llm_confidence": llm_result.get("llm_confidence", ""),
                "llm_reason": llm_result.get("llm_reason", ""),
                "llm_model": llm_result.get("llm_model", ""),
                "top_label": row.get("top_label", ""),
                "top_score": row.get("top_score", ""),
                "second_label": row.get("second_label", ""),
                "second_score": row.get("second_score", ""),
                "top_gap": row.get("top_gap", ""),
                "review_status": row.get("review_status", ""),
                "transcript": row.get("transcript", ""),
            }
            for score_emotion, score in ranked:
                report_row[score_column(score_emotion)] = f"{score:.6f}"
            report_rows_by_emotion[chosen_emotion].append(report_row)

            if all(seeded_counts[emotion] >= limits[emotion] for emotion in EMOTIONS):
                break

        if all(seeded_counts[emotion] >= limits[emotion] for emotion in EMOTIONS):
            break
        if not progress:
            break

    for emotion in EMOTIONS:
        report_rows = report_rows_by_emotion[emotion]
        if report_rows:
            review_dir = os.path.join(classified_root, "_candidate_review", emotion)
            report_csv = os.path.join(review_dir, "_candidate_report.csv")
            with open(report_csv, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
                writer.writeheader()
                writer.writerows(report_rows)

    return seeded_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative relabel loop: promote approved labels, regroup predictions, and refill candidate review folders.",
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_CLASSIFIED_ROOT,
        help="Path to multimodal_emotion_classified. Default: %(default)s",
    )
    parser.add_argument(
        "--target-per-emotion",
        type=int,
        default=100,
        help="Keep generating candidate review clips for emotions below this count. Default: %(default)s",
    )
    parser.add_argument(
        "--candidate-batch",
        type=int,
        default=30,
        help="Maximum number of candidate clips to refresh per emotion per run. Default: %(default)s",
    )
    parser.add_argument(
        "--compare-rounds",
        type=int,
        default=3,
        help="How many tournament passes to use when comparing candidates across all emotions. Default: %(default)s",
    )
    parser.add_argument(
        "--candidate-min-score",
        type=float,
        default=0.05,
        help="Minimum per-emotion score required before a clip can be seeded as a candidate. Default: %(default)s",
    )
    parser.add_argument(
        "--openai-api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key for optional LLM tie-breaking. If omitted, OPENAI_API_KEY is used.",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4.1",
        help="OpenAI model for optional LLM tie-breaking. Default: %(default)s",
    )
    parser.add_argument(
        "--openai-judge-limit",
        type=int,
        default=0,
        help="Maximum OpenAI judge calls for this run. 0 disables LLM judging. Default: %(default)s",
    )
    parser.add_argument(
        "--openai-timeout-sec",
        type=int,
        default=120,
        help="Timeout per OpenAI judge call in seconds. Default: %(default)s",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Do not import files from _approved/<Emotion> or manual_review_sorted/<Emotion> into the final 7 folders.",
    )
    parser.add_argument(
        "--no-regroup",
        action="store_true",
        help="Do not rebuild _regrouped_top_label.",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Do not refresh _candidate_review for low-count emotions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classified_root = os.path.abspath(args.root)

    ceo.ensure_emotion_folders(classified_root)
    ensure_loop_folders(classified_root)

    if not args.no_promote:
        promoted = promote_staged_labels(classified_root)
        print("Promoted into final 7 folders:")
        for emotion in EMOTIONS:
            print(f"{emotion}: {promoted.get(emotion, 0)}")
        print("")

    counts = ceo.current_counts(classified_root)
    review = ceo.review_counts(classified_root)

    print(f"Current final counts: {classified_root}")
    for emotion in EMOTIONS:
        print(f"{emotion}: {counts[emotion]}")
    print(f"Needs_Review: {review['Needs_Review']}")
    print(f"Rejected_LowQuality: {review['Rejected_LowQuality']}")

    if not args.no_regroup:
        regrouped = ceo.regroup_all_predictions(classified_root)
        print("\nRegrouped predictions:")
        print(os.path.join(classified_root, "_regrouped_top_label"))
        for emotion in EMOTIONS:
            print(f"{emotion}: {regrouped.get(emotion, 0)}")

    if not args.no_seed:
        seeded = seed_candidates_for_deficits(
            classified_root,
            target_per_emotion=args.target_per_emotion,
            candidate_batch=args.candidate_batch,
            compare_rounds=args.compare_rounds,
            candidate_min_score=args.candidate_min_score,
            openai_api_key=args.openai_api_key,
            openai_model=args.openai_model,
            openai_judge_limit=args.openai_judge_limit,
            openai_timeout_sec=args.openai_timeout_sec,
        )
        print("\nCandidate review refresh:")
        print(
            "Selection method: iterative emotion tournament "
            f"(rounds={args.compare_rounds}, min_score={args.candidate_min_score})"
        )
        if args.openai_api_key and args.openai_judge_limit > 0:
            print(f"OpenAI LLM tie-breaker: enabled model={args.openai_model} max_calls={args.openai_judge_limit}")
        else:
            print("OpenAI LLM tie-breaker: disabled")
        print(os.path.join(classified_root, "_candidate_review"))
        for emotion in EMOTIONS:
            deficit = max(0, args.target_per_emotion - counts.get(emotion, 0))
            print(f"{emotion}: current={counts[emotion]} deficit={deficit} candidates={seeded.get(emotion, 0)}")

    print("\nLoop usage:")
    print(f"1. Review files in {os.path.join(classified_root, '_candidate_review')}")
    print(f"2. Put approved files into {os.path.join(classified_root, '_approved')}")
    print(f"3. Or use {os.path.join(classified_root, 'manual_review_sorted')}")
    print("4. Run this script again")


if __name__ == "__main__":
    main()
