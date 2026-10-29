"""Independently verify the pinned audio and Japanese text model branches."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_jvnv_results_v2 import OUTPUT_LABELS
from run_historical_multimodal_validation import (
    HERE,
    LABELER_MODEL_ID,
    TEXT_MODEL_ID,
    configure_offline_runtime,
    create_run,
    load_historical_classifier,
)

SAMPLE_PREDICTIONS = (
    HERE
    / "results"
    / "runs"
    / "20261023_historical_multimodal_smoke_v003"
    / "raw_predictions"
    / "jvnv_predictions.csv"
)
EXPECTED_AUDIO_MAPPING = {
    "neu": "Neutral",
    "ang": "Angry",
    "hap": "Happy",
    "sad": "Sad",
}
JAPANESE_SMOKE_TEXT = "感情をこめて話している日本語の短い発話です。"


def scores_are_valid(scores: dict[str, float]) -> bool:
    return (
        set(scores) == set(OUTPUT_LABELS)
        and all(math.isfinite(float(score)) and float(score) >= 0 for score in scores.values())
        and math.isclose(sum(map(float, scores.values())), 1.0, rel_tol=1e-4, abs_tol=1e-4)
    )


def run_worker(branch: str, run_dir: Path, audio_path: Path, text: str) -> dict[str, Any]:
    configure_offline_runtime(run_dir)
    module = load_historical_classifier(run_dir)
    result: dict[str, Any] = {
        "branch": branch,
        "sample_audio": str(audio_path),
        "text_sample": text,
        "passed": False,
        "audio_model_id": LABELER_MODEL_ID,
        "text_model_id": TEXT_MODEL_ID,
    }
    try:
        if branch in ("audio", "combined"):
            audio_model = module.load_audio_model()
            if audio_model is None:
                raise RuntimeError(module.AUDIO_MODEL_ERROR or "Pinned audio model did not load")
            raw_audio_scores = module.audio_emotion_scores(str(audio_path))
            if not scores_are_valid(raw_audio_scores):
                raise RuntimeError(
                    f"Historical audio scoring returned unexpected labels or probabilities: "
                    f"{raw_audio_scores}"
                )
            result["audio"] = {
                "loaded": module.AUDIO_MODEL is not None,
                "runtime": module.AUDIO_RUNTIME,
                "raw_label_order": list(module.AUDIO_MODEL.hparams.label_encoder.ind2lab.values()),
                "historical_audio_mapping": module.CFG["label_maps"]["audio_to_canonical"],
                "expected_four_label_mapping": EXPECTED_AUDIO_MAPPING,
                "scores": raw_audio_scores,
                "score_labels_match_seven_outputs": set(raw_audio_scores) == set(OUTPUT_LABELS),
                "probabilities_sum_to_one": math.isclose(
                    sum(raw_audio_scores.values()), 1.0, rel_tol=1e-4, abs_tol=1e-4
                ),
                "fidelity": vars(module.AUDIO_MODEL.fidelity),
                "waveform_preprocessing": "mono float32, resampled to 16 kHz, whole-waveform layer normalization",
                "encoder_output_normalization": "single layer normalization across time and feature axes",
                "pooling": "SpeechBrain StatisticsPooling masked mean with original epsilon Gaussian perturbation",
                "classifier_head": "bias-free Linear(768, 4)",
                "probability_function": "softmax over neu, ang, hap, sad logits",
            }

        if branch in ("text", "combined"):
            text_model = module.load_text_model()
            if text_model is None:
                raise RuntimeError(module.TEXT_MODEL_ERROR or "Japanese text model did not load")
            text_scores = module.text_emotion_scores(text)
            if not scores_are_valid(text_scores):
                raise RuntimeError(
                    f"Historical text scoring returned unexpected labels or probabilities: "
                    f"{text_scores}"
                )
            result["text"] = {
                "loaded": module.TEXT_MODEL is not None,
                "runtime": module.TEXT_RUNTIME,
                "model_name": module.TEXT_MODEL.get("model_name"),
                "id2label": module.TEXT_MODEL.get("id2label"),
                "removed_unsupported_tokenizer_inputs": sorted(
                    getattr(module.TEXT_MODEL["tokenizer"], "removed_inputs", set())
                ),
                "scores": text_scores,
                "score_labels_match_seven_outputs": set(text_scores) == set(OUTPUT_LABELS),
                "probabilities_sum_to_one": math.isclose(
                    sum(text_scores.values()), 1.0, rel_tol=1e-4, abs_tol=1e-4
                ),
            }
        result["passed"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["audio_model_error"] = getattr(module, "AUDIO_MODEL_ERROR", "")
        result["text_model_error"] = getattr(module, "TEXT_MODEL_ERROR", "")
    return result


def load_sample() -> tuple[Path, str]:
    if not SAMPLE_PREDICTIONS.is_file():
        raise FileNotFoundError(f"Existing one-record sample not found: {SAMPLE_PREDICTIONS}")
    with SAMPLE_PREDICTIONS.open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream), None)
    if row is None:
        raise ValueError(f"No sample row in {SAMPLE_PREDICTIONS}")
    audio_path = Path(row["audio_path"])
    if not audio_path.is_file():
        raise FileNotFoundError(f"Existing JVNV sample audio not found: {audio_path}")
    text = str(row.get("transcript") or JAPANESE_SMOKE_TEXT).strip()
    if not text:
        text = JAPANESE_SMOKE_TEXT
    return audio_path, text


def invoke_worker(branch: str, run: Path, audio_path: Path, text: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        branch,
        "--run-dir",
        str(run),
        "--audio",
        str(audio_path),
        "--text",
        text,
    ]
    completed = subprocess.run(
        command,
        cwd=HERE,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=os.environ.copy(),
    )
    log_path = run / "logs" / f"model_compat_{branch}.log"
    log_path.write_text(
        f"$ {' '.join(command)}\n\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}",
        encoding="utf-8",
    )
    marker = "HISTORICAL_MODEL_COMPAT_RESULT="
    payload = next(
        (line[len(marker) :] for line in reversed(completed.stdout.splitlines()) if line.startswith(marker)),
        None,
    )
    if payload is None:
        return {
            "branch": branch,
            "passed": False,
            "error": f"Worker exited {completed.returncode} without a result record",
            "log": str(log_path),
        }
    result = json.loads(payload)
    result["worker_exit_code"] = completed.returncode
    result["log"] = str(log_path)
    return result


def write_report(run: Path, results: list[dict[str, Any]], audio: Path, text: str) -> bool:
    passed = all(item.get("passed") for item in results)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "historical_model_compatibility_only",
        "full_validation_run": False,
        "twelve_clip_smoke_run": False,
        "audio_sample": str(audio),
        "japanese_text_sample": text,
        "expected_historical_audio_label_order": list(EXPECTED_AUDIO_MAPPING),
        "expected_historical_audio_mapping": EXPECTED_AUDIO_MAPPING,
        "seven_output_labels": OUTPUT_LABELS,
        "results": results,
        "all_independent_and_combined_checks_passed": passed,
        "combined_failure_diagnosis": (
            "The earlier combined attempt failed while the SpeechBrain YAML loader "
            "discovered an unrelated optional k2 integration. With that import path "
            "bypassed, text-model prediction exposed a second incompatibility: the "
            "tokenizer emitted token_type_ids, but this DistilBert forward signature "
            "does not accept them. The compatibility tokenizer now filters only inputs "
            "unsupported by the loaded model; the combined one-sample test validates both."
        ),
        "next_command": (
            "python run_historical_multimodal_validation.py --smoke-test "
            "--run-id 20261024_historical_multimodal_smoke_v004"
            if passed
            else None
        ),
    }
    report_path = run / "reports" / "historical_model_compatibility.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    status = "PASS" if passed else "NOT READY"
    next_command = report["next_command"] or "No integrated smoke command: resolve the reported model error first."
    summary = [
        "# Historical model compatibility check",
        "",
        f"- Overall: **{status}**",
        "- Twelve-clip smoke run: **not run**",
        "- Full validation: **not run**",
        f"- Audio sample: `{audio}`",
        f"- Audio label order: `{', '.join(EXPECTED_AUDIO_MAPPING)}`",
        f"- Japanese sample text: `{text}`",
        "",
        "## Branch results",
        "",
    ]
    for item in results:
        summary.append(
            f"- **{item.get('branch')}**: {'PASS' if item.get('passed') else 'FAIL'}"
            + (f" — {item['error']}" if item.get("error") else "")
        )
    summary.extend(["", "## Next command", "", f"`{next_command}`", ""])
    (run / "reports" / "historical_model_compatibility.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="20261024_historical_model_compat_v001")
    parser.add_argument("--worker", choices=("audio", "text", "combined"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--text", default="")
    args = parser.parse_args()

    if args.worker:
        if args.run_dir is None or args.audio is None:
            parser.error("--worker requires --run-dir and --audio")
        result = run_worker(args.worker, args.run_dir, args.audio, args.text)
        print("HISTORICAL_MODEL_COMPAT_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if result["passed"] else 1

    audio_path, text = load_sample()
    run_id, run = create_run(args.run_id)
    configure_offline_runtime(run)
    results = [
        invoke_worker("audio", run, audio_path, text),
        invoke_worker("text", run, audio_path, text),
    ]
    if all(item.get("passed") for item in results):
        results.append(invoke_worker("combined", run, audio_path, text))
    return 0 if write_report(run, results, audio_path, text) else 1


if __name__ == "__main__":
    raise SystemExit(main())
