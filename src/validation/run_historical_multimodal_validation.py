"""Isolated JVNV validation using the original multimodal classifier functions."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from evaluate_jvnv_results_v2 import OUTPUT_LABELS, evaluate
from historical_speechbrain_compat import (
    ModelInputFilteringTokenizer,
    SPEECHBRAIN_LABEL_ORDER,
    load_pinned_speechbrain_adapter,
)
from validate_jvnv_labels import extract_and_manifest

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
VALIDATION_CFG = json.loads((HERE / "jvnv_validation_config.json").read_text(encoding="utf-8"))
ORIGINAL = (PROJECT / VALIDATION_CFG["historical_pipeline_relative"]).resolve()
HISTORICAL_SCRIPT = ORIGINAL / "multimodal_emotion_classifier.py"
HISTORICAL_CONFIG = ORIGINAL / "emotion_classifier_config.json"
REVISION_PATH = HERE / "model_revisions.json"
MODEL_REVISIONS = json.loads(REVISION_PATH.read_text(encoding="utf-8"))
LABELER_MODEL_ID = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
LABELER_FALLBACK_ID = "superb/wav2vec2-base-superb-er"
TEXT_MODEL_ID = "koshin2001/Japanese-to-emotions"
ASR_MODEL_ID = "Systran/faster-whisper-small"
FACEBOOK_WAV2VEC2_REVISION = "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
FACEBOOK_WAV2VEC2_METADATA = ORIGINAL / "wav2vec2_checkpoints" / "models--facebook--wav2vec2-base" / "snapshots" / FACEBOOK_WAV2VEC2_REVISION


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_path(model_id: str) -> Path:
    namespace, repo = model_id.split("/", 1)
    revision = MODEL_REVISIONS.get(model_id)
    if not revision:
        raise ValueError(f"No pinned revision recorded for {model_id}")
    return HERE / "model_cache" / "huggingface" / "hub" / f"models--{namespace}--{repo}" / "snapshots" / revision


def model_cache_facts() -> dict[str, Any]:
    speechbrain = snapshot_path(LABELER_MODEL_ID)
    speechbrain_files = ("hyperparams.yaml", "wav2vec2.ckpt", "model.ckpt", "label_encoder.txt")
    fallback = snapshot_path(LABELER_FALLBACK_ID)
    text = snapshot_path(TEXT_MODEL_ID)
    asr = snapshot_path(ASR_MODEL_ID)
    return {
        "speechbrain_primary": {
            "model_id": LABELER_MODEL_ID,
            "revision": MODEL_REVISIONS[LABELER_MODEL_ID],
            "snapshot": str(speechbrain),
            "required_files_present": all((speechbrain / name).is_file() for name in speechbrain_files),
            "files": {name: (speechbrain / name).is_file() for name in speechbrain_files},
        },
        "facebook_wav2vec2_feature_extractor": {
            "model_id": "facebook/wav2vec2-base",
            "revision": FACEBOOK_WAV2VEC2_REVISION,
            "snapshot": str(FACEBOOK_WAV2VEC2_METADATA),
            "config_present": (FACEBOOK_WAV2VEC2_METADATA / "config.json").is_file(),
            "preprocessor_present": (FACEBOOK_WAV2VEC2_METADATA / "preprocessor_config.json").is_file(),
            "source": "existing historical read-only cache",
        },
        "audio_fallback": {
            "model_id": LABELER_FALLBACK_ID,
            "revision": MODEL_REVISIONS[LABELER_FALLBACK_ID],
            "snapshot": str(fallback),
            "snapshot_present": fallback.is_dir(),
        },
        "text": {
            "model_id": TEXT_MODEL_ID,
            "revision": MODEL_REVISIONS[TEXT_MODEL_ID],
            "snapshot": str(text),
            "snapshot_present": text.is_dir(),
            "weights_present": any((text / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")),
        },
        "asr": {
            "model_id": ASR_MODEL_ID,
            "revision": MODEL_REVISIONS[ASR_MODEL_ID],
            "snapshot": str(asr),
            "snapshot_present": asr.is_dir(),
            "model_bin_present": (asr / "model.bin").is_file(),
        },
    }


def configure_offline_runtime(run: Path) -> None:
    """Resolve only runtime caches/devices; no model, label, or fusion settings change."""
    runtime = run / "runtime"
    temp = runtime / "tmp"
    pycache = runtime / "pycache"
    numba_cache = runtime / "numba_cache"
    for directory in (runtime, temp, pycache, numba_cache):
        directory.mkdir(parents=True, exist_ok=True)
    isolated_hub = runtime / "huggingface" / "hub"
    cached_models = (
        LABELER_MODEL_ID,
        LABELER_FALLBACK_ID,
        TEXT_MODEL_ID,
        ASR_MODEL_ID,
    )
    for model_id in cached_models:
        source_snapshot = snapshot_path(model_id)
        if not source_snapshot.is_dir():
            continue
        namespace, repo = model_id.split("/", 1)
        repo_cache = isolated_hub / f"models--{namespace}--{repo}"
        target_snapshot = repo_cache / "snapshots" / MODEL_REVISIONS[model_id]
        target_snapshot.mkdir(parents=True, exist_ok=True)
        for source_file in source_snapshot.rglob("*"):
            if not source_file.is_file():
                continue
            destination = target_snapshot / source_file.relative_to(source_snapshot)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                continue
            try:
                os.link(source_file, destination)
            except OSError:
                import shutil

                shutil.copy2(source_file, destination)
        refs = repo_cache / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(MODEL_REVISIONS[model_id], encoding="utf-8")
    local_hub = str(isolated_hub)
    os.environ.update(
        {
            "HF_HOME": str(runtime / "huggingface"),
            "HF_HUB_CACHE": local_hub,
            "HUGGINGFACE_HUB_CACHE": local_hub,
            "TORCH_HOME": str(runtime / "torch"),
            "XDG_CACHE_HOME": str(runtime / "cache"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "OPENAI_EMOTION_JUDGE": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(pycache),
            "NUMBA_DISABLE_INTEL_SVML": "1",
            "NUMBA_CACHE_DIR": str(numba_cache),
            "TMP": str(temp),
            "TEMP": str(temp),
        }
    )
    os.environ.pop("TRANSFORMERS_CACHE", None)
    import numba

    if numba.config.USING_SVML:
        raise RuntimeError("Numba unexpectedly enabled Intel SVML in the isolated validation runtime")
    if Path(numba.config.CACHE_DIR).resolve() != numba_cache.resolve():
        raise RuntimeError(
            f"Numba cache is not isolated to this run: {numba.config.CACHE_DIR}"
        )


def load_historical_classifier(run: Path):
    if not HISTORICAL_SCRIPT.is_file() or not HISTORICAL_CONFIG.is_file():
        raise FileNotFoundError(f"Historical classifier/config not found under {ORIGINAL}")
    spec = importlib.util.spec_from_file_location("original_multimodal_emotion_classifier", HISTORICAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load historical classifier from {HISTORICAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Keep runtime compatibility writes inside this validation run.
    module.BASE_DIR = str(run / "runtime")
    original_load_whisper = module.load_whisper
    module.load_whisper = lambda force_cpu=False: original_load_whisper(force_cpu=True)
    module.AUDIO_MODEL = None
    module.AUDIO_MODEL_DISABLED = False
    module.AUDIO_MODEL_ERROR = ""
    module.AUDIO_RUNTIME = {}

    def load_pinned_audio_model():
        if module.AUDIO_MODEL is not None:
            return module.AUDIO_MODEL
        try:
            model, loading = load_pinned_speechbrain_adapter(
                snapshot_path(LABELER_MODEL_ID),
                sample_rate=int(module.TARGET_SR),
                requested_device=str(
                    module.CFG["models"]["audio_emotion"].get("device", "cpu")
                ),
                feature_extractor_snapshot=FACEBOOK_WAV2VEC2_METADATA,
            )
            module.AUDIO_MODEL = model
            module.AUDIO_RUNTIME = {
                "backend": "speechbrain_pinned_local_compat",
                "device": model.device,
                "model_name": LABELER_MODEL_ID,
                "is_fallback": False,
                "checkpoint_loading": loading,
                "label_order": list(SPEECHBRAIN_LABEL_ORDER),
                "fidelity": vars(model.fidelity),
            }
            return model
        except Exception as exc:
            module.AUDIO_MODEL_ERROR = f"{type(exc).__name__}: {exc}"
            module.AUDIO_MODEL_DISABLED = True
            print(
                "Warning: pinned SpeechBrain audio model unavailable; continuing with "
                f"independent branches. Reason: {module.AUDIO_MODEL_ERROR}",
                flush=True,
            )
            return None

    module.load_audio_model = load_pinned_audio_model
    original_load_text_model = module.load_text_model

    def load_compatible_text_model():
        model = original_load_text_model()
        if model is not None and not isinstance(
            model.get("tokenizer"), ModelInputFilteringTokenizer
        ):
            import inspect

            forward_parameters = inspect.signature(model["model"].forward).parameters
            accepts_arbitrary_inputs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in forward_parameters.values()
            )
            accepted_inputs = None if accepts_arbitrary_inputs else set(forward_parameters)
            model["tokenizer"] = ModelInputFilteringTokenizer(
                model["tokenizer"], accepted_inputs
            )
        return model

    module.load_text_model = load_compatible_text_model
    original_text_emotion_scores = module.text_emotion_scores

    def score_with_compatible_text_model(text):
        scores = original_text_emotion_scores(text)
        if module.TEXT_MODEL is not None:
            tokenizer = module.TEXT_MODEL["tokenizer"]
            module.TEXT_RUNTIME["compat_removed_tokenizer_inputs"] = sorted(
                getattr(tokenizer, "removed_inputs", set())
            )
        return scores

    module.text_emotion_scores = score_with_compatible_text_model
    return module


def write_fidelity_audit(run: Path, module_path: Path, config_path: Path, caches: dict[str, Any]) -> Path:
    """Write the audit before importing or invoking any model inference function."""
    report = run / "reports" / "historical_multimodal_fidelity_audit.md"
    revision = MODEL_REVISIONS[LABELER_MODEL_ID]
    audit = f"""# Historical multimodal fidelity audit

Created before inference: {datetime.now(timezone.utc).isoformat()}

## Verified matches

- The implementation loaded for this run is the original project file
  `{module_path}` (SHA-256 `{sha256(module_path)}`), not
  `frozen_source/character_emotion_pseudolabel.py`.
- Its adjacent frozen configuration is `{config_path}`
  (SHA-256 `{sha256(config_path)}`). Runtime fusion weights, decision thresholds,
  labels, and mappings are not changed.
- The executable classifier path uses `extract_features`,
  `transcribe_if_needed`, `audio_emotion_scores`, `text_emotion_scores`,
  `prosody_emotion_scores`, `weighted_fusion`, and `auto_label_decision`.
- The original classifier's `audio_emotion_scores` remains the scoring path.
  For validation only, its loader is replaced with a local adapter that loads
  `{LABELER_MODEL_ID}` from the pinned snapshot `{revision}`:
  `{caches["speechbrain_primary"]["required_files_present"]}`. The adapter
  uses the declared Wav2Vec2 encoder, waveform normalization, full time-by-feature
  output normalization, StatisticsPooling masked mean/noise, 4-way linear head,
  softmax, sample rate, and label order. The `facebook/wav2vec2-base` feature
  extractor metadata was recovered from the existing read-only historical cache
  at `{caches["facebook_wav2vec2_feature_extractor"]["snapshot"]}`; it specifies
  16 kHz, `do_normalize=true`, and no returned attention mask. This bypasses SpeechBrain's YAML
  importer and unrelated optional `k2` import; trained weights are unchanged.
- The executable code does not call the mel-CNN or embedding-emotion model
  loaders. The configuration's `feature_extractors` list is therefore not
  treated as evidence that those branches ran. No Bagus or Dpng model is loaded.
- Text model cache: `{caches["text"]["snapshot_present"]}`; faster-whisper
  small cache: `{caches["asr"]["snapshot_present"]}`. Offline mode prevents
  downloads. The configured fallback cache is present:
  `{caches["audio_fallback"]["snapshot_present"]}`.

## Differences for this JVNV validation

- JVNV provides already-segmented WAV utterances. The historical corpus first
  used its anime-media segmentation and Japanese/dialogue/quality filters;
  that upstream segmentation is not repeated on JVNV.
- This process is offline to avoid downloads. Cached snapshots are hard-linked
  (or copied where necessary) into this run's private cache; the existing cache
  is not used as a write target. Faster-whisper is forced to CPU; audio and text
  follow their configured device preference when available.
- Numba's Intel SVML vector library is disabled and its compiled cache is
  isolated to this run. JIT remains enabled; disabling it breaks librosa's
  gufunc initialization in the selected Python environment.
- The local audio adapter does not import SpeechBrain or activate its YAML lazy
  modules, so its optional `k2` import failure cannot interfere with
  initialization of the independent Japanese text branch.
- JVNV has no historical source transcript. `transcribe_if_needed` generates
  Japanese transcripts with the historical classifier's faster-whisper path
  (language `ja`, configured beam size, VAD, and transcript reliability inputs).
- The classifier fits its feature scaler to the unlabeled JVNV validation pool
  (only the smoke candidate pool during smoke). The original scaler was fit to
  the source corpus. No JVNV reference labels enter feature extraction, scoring,
  fusion, confidence decisions, or the scaler.
- The primary evaluation retains all seven model outputs. JVNV `surprise` is
  mapped to `Pleasant_Surprise`; this is an approximate semantic mapping, not
  equivalence. JVNV has no Neutral reference class.
- Clean clustering is not called in this mode.

## Unknowns

- Historical per-run model status/revisions are absent. Although the pinned
  SpeechBrain model exists now, its exact activation in the conference run and
  whether that run used the Transformers fallback remain unknown.
- The original run's ASR outputs, runtime device/fallback records, and exact
  shell/menu invocation were not preserved.
- The configured Transformers audio fallback is not cached at the expected
  pinned path. It will not be downloaded; if the pinned local primary checkpoint
  fails to load, the audio branch is reported unavailable.

This report is generated before inference; runtime model and branch evidence is
recorded separately in `historical_multimodal_smoke_status.json`.
"""
    report.write_text(audit, encoding="utf-8")
    return report


def instrument(module) -> Counter:
    calls: Counter = Counter()
    names = (
        "extract_features",
        "sfx_rejection_gate",
        "low_quality_flag",
        "transcribe_if_needed",
        "load_whisper",
        "audio_emotion_scores",
        "load_audio_model",
        "text_emotion_scores",
        "load_text_model",
        "prosody_emotion_scores",
        "weighted_fusion",
        "auto_label_decision",
    )
    for name in names:
        original = getattr(module, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        setattr(module, name, counted)
    return calls


def extract_features(module, manifest: pd.DataFrame) -> list[dict[str, Any]]:
    results = []
    for item in manifest.to_dict("records"):
        try:
            feats = module.extract_features(item["audio_path"])
            results.append({"clip_id": item["clip_id"], "audio_path": item["audio_path"], "features": feats})
        except Exception as exc:
            results.append(
                {
                    "clip_id": item["clip_id"],
                    "audio_path": item["audio_path"],
                    "feature_error": f"{type(exc).__name__}: {exc}",
                }
            )
    return results


def quality_result(module, features: dict[str, float], scaler: dict[str, Any]) -> tuple[bool, str]:
    is_sfx, sfx_reasons = module.sfx_rejection_gate(features, scaler)
    if is_sfx:
        return False, "SFX_rejection | " + " | ".join(sfx_reasons)
    is_low, low_reasons = module.low_quality_flag(features, scaler)
    if is_low:
        return False, "Low_quality | " + " | ".join(low_reasons)
    return True, ""


def branch_top(module, scores: dict[str, float]) -> str:
    return module.top2(scores)[0][0] if scores else ""


def predict_one(module, record: dict[str, Any], features: dict[str, float], scaler: dict[str, Any]) -> dict[str, Any]:
    clip_id = str(record["clip_id"])
    audio_path = str(record["audio_path"])
    try:
        transcript_result = module.transcribe_if_needed(audio_path)
        transcript = str(transcript_result.get("transcript") or "")
        transcript_rel = module.transcript_reliability(
            transcript_result.get("avg_logprob"),
            transcript_result.get("no_speech_prob"),
        )
        audio_scores = module.audio_emotion_scores(audio_path)
        text_scores = module.text_emotion_scores(transcript)
        prosody_scores = module.prosody_emotion_scores(features, scaler)
        final_scores = module.weighted_fusion(audio_scores, text_scores, features, scaler, transcript_rel)
        if not final_scores:
            raise RuntimeError("No active audio, text, or prosody branch returned usable scores")

        (top_label, top_score), (second_label, second_score) = module.top2(final_scores)
        gap = float(top_score - second_score)
        auto_ok, decision_reason = module.auto_label_decision(
            top_label=top_label,
            top_score=float(top_score),
            gap=gap,
            text_scores=module.normalize_to_active_scores(text_scores),
            prosody=prosody_scores,
            transcript_rel=transcript_rel,
        )
        min_conf = float(module.CFG["fusion"]["min_final_confidence"])
        min_gap = float(module.CFG["fusion"]["min_final_gap"])
        max_duration = float(module.CFG.get("review", {}).get("max_auto_duration_sec", 10.0))
        if float(features.get("duration_sec", 0.0)) > max_duration:
            status, reason = "needs_review", "duration_too_long"
        elif top_score < min_conf or gap < min_gap or not auto_ok:
            status, reason = "needs_review", decision_reason if not auto_ok else "uncertain_fusion"
        else:
            status, reason = "auto_accept", decision_reason

        label = top_label if top_label in OUTPUT_LABELS else ""
        if not label:
            raise RuntimeError(f"Historical fusion returned invalid output label: {top_label!r}")
        return {
            "clip_id": clip_id,
            "predicted_emotion": label,
            "second_prediction": second_label,
            "confidence": float(top_score),
            "confidence_gap": gap,
            "confidence_status": status,
            "audio_quality_status": "pass",
            "transcript": transcript,
            "transcript_reliability": transcript_rel,
            "audio_prediction": branch_top(module, audio_scores),
            "text_prediction": branch_top(module, text_scores),
            "prosody_prediction": branch_top(module, prosody_scores),
            "model_agreement_count": module.branch_top_count(top_label, audio_scores, text_scores, prosody_scores),
            "decision_reason": reason,
            "audio_scores": json.dumps(audio_scores, ensure_ascii=False, sort_keys=True),
            "text_scores": json.dumps(text_scores, ensure_ascii=False, sort_keys=True),
            "prosody_scores": json.dumps(prosody_scores, ensure_ascii=False, sort_keys=True),
            "final_scores": json.dumps(final_scores, ensure_ascii=False, sort_keys=True),
            "error_message": "",
            "duration_sec": float(features.get("duration_sec", 0.0)),
            "historical_duration_eligible": float(features.get("duration_sec", 0.0)) <= max_duration,
        }
    except Exception as exc:
        return {
            "clip_id": clip_id,
            "predicted_emotion": "",
            "second_prediction": "",
            "confidence": "",
            "confidence_gap": "",
            "confidence_status": "failed_inference",
            "audio_quality_status": "pass",
            "transcript": "",
            "transcript_reliability": "",
            "audio_prediction": "",
            "text_prediction": "",
            "prosody_prediction": "",
            "model_agreement_count": 0,
            "decision_reason": "",
            "audio_scores": "",
            "text_scores": "",
            "prosody_scores": "",
            "final_scores": "",
            "error_message": f"{type(exc).__name__}: {exc}",
            "duration_sec": float(features.get("duration_sec", 0.0)),
            "historical_duration_eligible": float(features.get("duration_sec", 0.0)) <= float(module.CFG.get("review", {}).get("max_auto_duration_sec", 10.0)),
            "_traceback": traceback.format_exc(),
        }


def build_prediction_rows(
    module,
    reference: pd.DataFrame,
    scaler: dict[str, Any],
    feature_rows: list[dict[str, Any]],
    calls: Counter,
    smoke: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features_by_id = {row["clip_id"]: row for row in feature_rows}
    eligible: list[dict[str, Any]] = []
    quality_by_id: dict[str, tuple[bool, str]] = {}
    for row in feature_rows:
        if "feature_error" in row:
            quality_by_id[row["clip_id"]] = (False, f"feature_error: {row['feature_error']}")
            continue
        accepted, reason = quality_result(module, row["features"], scaler)
        quality_by_id[row["clip_id"]] = (accepted, reason)
        if accepted:
            eligible.append(row)

    if smoke:
        # Infer the complete predeclared balanced pool; retain every row so
        # quality rejects and feature failures remain visible in diagnostics.
        chosen = eligible
        output_manifest = reference.copy()
    else:
        chosen = []
        output_manifest = reference.copy()

    output_rows = []
    for ref_row in output_manifest.to_dict("records"):
        clip_id = ref_row["clip_id"]
        entry = features_by_id.get(clip_id, {})
        accepted, quality_reason = quality_by_id.get(clip_id, (False, "feature extraction missing"))
        if "feature_error" in entry:
            result = {
                "clip_id": clip_id,
                "predicted_emotion": "",
                "confidence_status": "failed_inference",
                "audio_quality_status": "unknown",
                "error_message": entry["feature_error"],
            }
        elif not accepted:
            result = {
                "clip_id": clip_id,
                "predicted_emotion": "",
                "confidence_status": "reject",
                "audio_quality_status": "rejected",
                "decision_reason": quality_reason,
                "error_message": "",
            }
        else:
            prediction_input = {
                "clip_id": ref_row["clip_id"],
                "audio_path": ref_row["audio_path"],
            }
            result = predict_one(module, prediction_input, entry["features"], scaler)
        if "duration_sec" not in result:
            duration = float(entry.get("features", {}).get("duration_sec", 0.0))
            max_duration = float(module.CFG.get("review", {}).get("max_auto_duration_sec", 10.0))
            result["duration_sec"] = duration
            result["historical_duration_eligible"] = duration <= max_duration
        output_rows.append(
            {
                **result,
                "audio_path": ref_row["audio_path"],
                "speaker_id": ref_row["speaker_id"],
                "reference_emotion": ref_row["reference_emotion"],
            }
        )
        if result.get("_traceback"):
            calls["failed_inference"] += 1
    return pd.DataFrame(output_rows), {
        "smoke_pool_size": len(reference),
        "eligible_count": len(eligible),
        "inferred_clip_ids": [row["clip_id"] for row in chosen] if smoke else reference.clip_id.tolist(),
        "quality_reasons": {key: value[1] for key, value in quality_by_id.items() if not value[0]},
    }


def create_run(run_id: str) -> tuple[str, Path]:
    base = HERE / "results" / "runs"
    base.mkdir(parents=True, exist_ok=True)
    requested = run_id
    version = 2
    run = base / requested
    while run.exists():
        run_id = f"{requested}_v{version:03d}"
        run = base / run_id
        version += 1
    for directory in ("data", "workspace", "raw_predictions", "figures", "reports", "logs", "runtime"):
        (run / directory).mkdir(parents=True, exist_ok=False)
    return run_id, run


def write_runtime_report(
    run: Path,
    audit_path: Path,
    module,
    calls: Counter,
    pool: dict[str, Any],
    predictions: pd.DataFrame,
    smoke: bool,
    expected_count: int,
) -> tuple[bool, bool]:
    audio_runtime = dict(getattr(module, "AUDIO_RUNTIME", {}) or {})
    text_runtime = dict(getattr(module, "TEXT_RUNTIME", {}) or {})
    whisper_runtime = dict(getattr(module, "WHISPER_RUNTIME", {}) or {})
    audio_model = getattr(module, "AUDIO_MODEL", None)
    text_model = getattr(module, "TEXT_MODEL", None)
    audio_model_name = audio_runtime.get("model_name", "")
    text_model_name = text_model.get("model_name", "") if isinstance(text_model, dict) else ""
    outputs = predictions.get("predicted_emotion", pd.Series(dtype=str)).fillna("").astype(str)
    runtime_failures = predictions.get("error_message", pd.Series(dtype=str)).fillna("").astype(str)
    active_audio = bool(audio_model is not None and audio_runtime and audio_runtime.get("backend") != "")
    active_text = bool(text_model is not None and text_runtime and text_model_name == TEXT_MODEL_ID)
    active_prosody = bool(calls["prosody_emotion_scores"] > 0)
    active_asr = bool(whisper_runtime and calls["transcribe_if_needed"] > 0)
    audio_is_primary = audio_model_name == LABELER_MODEL_ID
    audio_fidelity = dict(audio_runtime.get("fidelity", {}) or {})
    fidelity_ready = bool(
        audio_fidelity.get("sampling_rate") == 16000
        and audio_fidelity.get("do_normalize") is True
        and audio_fidelity.get("output_normalization_axes") == "time_and_feature"
        and audio_fidelity.get("pooling") == "masked_mean_plus_original_statistics_pooling_noise"
        and audio_fidelity.get("label_order") == list(SPEECHBRAIN_LABEL_ORDER)
    )
    smoke_coverage_complete = bool(
        not smoke
        or (
            len(pool["inferred_clip_ids"]) == expected_count
            and len(predictions) == expected_count
            and outputs.isin(OUTPUT_LABELS).all()
            and not runtime_failures.ne("").any()
        )
    )
    smoke_passed = bool(
        smoke
        and audit_path.is_file()
        and active_audio
        and active_text
        and active_prosody
        and active_asr
        and audio_is_primary
        and fidelity_ready
        and calls["audio_emotion_scores"] > 0
        and calls["text_emotion_scores"] > 0
        and calls["weighted_fusion"] > 0
        and calls["auto_label_decision"] > 0
        and smoke_coverage_complete
    )
    full_completed = bool(not smoke and len(predictions) == expected_count)
    report = {
        "mode": "historical_multimodal",
        "smoke_test_only": smoke,
        "smoke_ready": smoke_passed if smoke else None,
        "full_run_performed": not smoke,
        "full_run_completed": full_completed if not smoke else False,
        "fidelity_audit_created_before_inference": audit_path.is_file(),
        "calls": dict(calls),
        "runtime": {
            "audio": audio_runtime,
            "text": text_runtime,
            "text_model_id": text_model_name,
            "asr": whisper_runtime,
            "model_load_errors": {
                "audio": getattr(module, "AUDIO_MODEL_ERROR", ""),
                "text": getattr(module, "TEXT_MODEL_ERROR", ""),
                "asr": getattr(module, "WHISPER_MODEL_ERROR", ""),
            },
            "asr_model_config_id": module.CFG["models"]["asr"]["name"],
            "audio_primary_model_loaded": audio_is_primary,
            "audio_fidelity_ready": fidelity_ready,
            "audio_fidelity": audio_fidelity,
            "audio_branch_active": active_audio,
            "text_branch_active": active_text,
            "prosody_branch_active": active_prosody,
            "asr_active": active_asr,
        },
        "smoke_pool": pool,
        "successful_output_count": int(outputs.isin(OUTPUT_LABELS).sum()),
        "failed_inference_count": int(runtime_failures.ne("").sum()),
        "complete_smoke_pool_inferred": smoke_coverage_complete if smoke else None,
        "model_output_labels": OUTPUT_LABELS,
        "metrics_interpretation": "The smoke run is a compatibility diagnostic; its metrics are not a validation accuracy estimate.",
    }
    status_name = "historical_multimodal_smoke_status" if smoke else "historical_multimodal_run_status"
    (run / "reports" / f"{status_name}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    title = "smoke status" if smoke else "full-run status"
    full_run_line = "no" if smoke else str(full_completed).lower()
    (run / "reports" / f"{status_name}.md").write_text(
        f"# Historical multimodal {title}\n\n"
        f"- Smoke ready: **{smoke_passed}**\n"
        f"- Primary SpeechBrain model loaded: **{audio_is_primary}**\n"
        f"- Audio/text/prosody/ASR active: **{active_audio}/{active_text}/{active_prosody}/{active_asr}**\n"
        f"- Successful seven-label outputs: **{int(outputs.isin(OUTPUT_LABELS).sum())}**\n"
        f"- Failed inferences: **{int(runtime_failures.ne('').sum())}**\n"
        f"- Full inference completed: **{full_run_line}**\n\n"
        "The smoke is a technical compatibility check, not a JVNV accuracy result. See "
        "`historical_multimodal_fidelity_audit.md` for pre-inference assumptions and "
        "limitations; detailed runtime model IDs and instrumented call counts are in "
        f"`{status_name}.json`.\n",
        encoding="utf-8",
    )
    return smoke_passed, full_completed


def write_smoke_diagnostics(predictions: pd.DataFrame, report_dir: Path) -> dict[str, Any]:
    """Report raw predictions separately from confidence-gated operation."""
    labels = predictions["predicted_emotion"].fillna("").astype(str)
    references = predictions["reference_emotion"].fillna("").astype(str)
    statuses = predictions["confidence_status"].fillna("").astype(str)
    returned = labels.isin(OUTPUT_LABELS)
    correct_raw = returned & labels.eq(references)
    accepted = statuses.eq("auto_accept") & returned
    accepted_correct = accepted & labels.eq(references)
    total = len(predictions)

    audio_scores = predictions["audio_scores"].fillna("").astype(str)
    audio_top_labels: list[str] = []
    happy_scores: list[float] = []
    for serialized in audio_scores:
        try:
            scores = json.loads(serialized) if serialized else {}
        except json.JSONDecodeError:
            scores = {}
        audio_top_labels.append(branch_top_from_scores(scores))
        happy_scores.append(float(scores.get("Happy", 0.0)))

    diagnostics = {
        "sample_count": total,
        "successful_raw_predictions": int(returned.sum()),
        "raw_predicted_label_accuracy": float(correct_raw.sum() / total) if total else None,
        "raw_predicted_label_correct_count": int(correct_raw.sum()),
        "auto_accepted_count": int(accepted.sum()),
        "auto_accepted_coverage": float(accepted.sum() / total) if total else None,
        "auto_accepted_accuracy": (
            float(accepted_correct.sum() / accepted.sum()) if accepted.any() else None
        ),
        "operational_accuracy_full_pool": (
            float(accepted_correct.sum() / total) if total else None
        ),
        "needs_review_count": int(statuses.eq("needs_review").sum()),
        "rejected_count": int(statuses.eq("reject").sum()),
        "failed_inference_count": int(statuses.eq("failed_inference").sum()),
        "auto_accept_is_the_only_confidently_accepted_status": True,
        "speechbrain_audio_top_label_counts": pd.Series(audio_top_labels).value_counts().to_dict(),
        "speechbrain_happy_probability_mean": (
            float(sum(happy_scores) / len(happy_scores)) if happy_scores else None
        ),
        "speechbrain_happy_probability_ge_0_99_count": sum(
            score >= 0.99 for score in happy_scores
        ),
        "speechbrain_happy_probability_ge_0_99_by_reference": {
            str(label): int(
                sum(
                    score >= 0.99
                    for score, reference in zip(happy_scores, references)
                    if reference == label
                )
            )
            for label in sorted(references.unique())
        },
        "branch_final_disagreement_counts": {
            branch: int(
                sum(
                    predicted != "" and branch_label != "" and predicted != branch_label
                    for predicted, branch_label in zip(labels, predictions[column].fillna("").astype(str))
                )
            )
            for branch, column in (
                ("audio", "audio_prediction"),
                ("text", "text_prediction"),
                ("prosody", "prosody_prediction"),
            )
        },
    }
    (report_dir / "historical_multimodal_smoke_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return diagnostics


def write_historical_evaluation_strata(predictions: pd.DataFrame, report_dir: Path) -> dict[str, Any]:
    """Report unfiltered primary and duration-eligible secondary results."""
    def summarize(frame: pd.DataFrame, name: str) -> dict[str, Any]:
        labels = frame.predicted_emotion.fillna("").astype(str)
        refs = frame.reference_emotion.fillna("").astype(str)
        status = frame.confidence_status.fillna("").astype(str)
        returned = labels.isin(OUTPUT_LABELS)
        raw_correct = returned & labels.eq(refs)
        accepted = status.eq("auto_accept") & returned
        accepted_correct = accepted & labels.eq(refs)
        return {
            "stratum": name,
            "sample_count": len(frame),
            "raw_prediction_count": int(returned.sum()),
            "raw_prediction_coverage": float(returned.mean()) if len(frame) else None,
            "raw_predicted_label_accuracy_full_stratum": float(raw_correct.mean()) if len(frame) else None,
            "raw_accuracy_among_returned_labels": float(raw_correct.sum() / returned.sum()) if returned.any() else None,
            "auto_accepted_count": int(accepted.sum()),
            "auto_accepted_coverage": float(accepted.mean()) if len(frame) else None,
            "auto_accepted_accuracy": float(accepted_correct.sum() / accepted.sum()) if accepted.any() else None,
            "needs_review_count": int(status.eq("needs_review").sum()),
            "rejected_count": int(status.eq("reject").sum()),
            "failed_inference_count": int(status.eq("failed_inference").sum()),
        }

    eligible = predictions.historical_duration_eligible.fillna(False).astype(bool)
    strata = [
        summarize(predictions, "all_recordings_unfiltered_primary"),
        summarize(predictions[eligible], "historical_duration_eligible_secondary"),
    ]
    reasons = (
        predictions.assign(
            outcome_reason=predictions.get("decision_reason", pd.Series("", index=predictions.index)).fillna("").astype(str),
            outcome_status=predictions.confidence_status.fillna("").astype(str),
        )
        .groupby(["outcome_status", "outcome_reason"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    reasons.to_csv(report_dir / "historical_multimodal_outcome_reasons.csv", index=False)
    report = {
        "primary_analysis": "all 1,615 JVNV recordings without duration or acoustic filtering",
        "secondary_analysis": "recordings at or below the frozen historical 10-second auto-acceptance limit",
        "long_clips_are_retained_in_primary": True,
        "strata": strata,
        "outcome_reason_report": "historical_multimodal_outcome_reasons.csv",
    }
    (report_dir / "historical_multimodal_evaluation_strata.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(strata).to_csv(report_dir / "historical_multimodal_evaluation_strata.csv", index=False)
    return report


def branch_top_from_scores(scores: dict[str, Any]) -> str:
    if not scores:
        return ""
    return max(scores, key=lambda label: float(scores[label]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jvnv-zip", default=str(HERE.parent / "jvnv_ver1.zip"))
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--smoke-test",
        action="store_true",
        help="Infer the complete balanced 12-record smoke pool.",
    )
    modes.add_argument("--full", action="store_true", help="Run all JVNV records; requires a passed historical smoke run.")
    parser.add_argument("--run-id", help="Base run ID; a version suffix is added rather than overwriting.")
    parser.add_argument("--smoke-run-id", help="Passed smoke run required by --full.")
    args = parser.parse_args()

    if HERE.resolve().is_relative_to(ORIGINAL.resolve()):
        raise RuntimeError("Validation workspace is inside the protected historical source tree")
    zip_path = Path(args.jvnv_zip).resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"JVNV archive not found: {zip_path}")
    if args.full:
        if not args.smoke_run_id:
            raise RuntimeError("--full requires --smoke-run-id pointing to a passed historical smoke run")
        smoke_status = HERE / "results" / "runs" / args.smoke_run_id / "reports" / "historical_multimodal_smoke_status.json"
        if not smoke_status.is_file() or not json.loads(smoke_status.read_text(encoding="utf-8")).get("smoke_ready"):
            raise RuntimeError(f"Historical smoke gate is missing or failed: {smoke_status}")

    requested_id = args.run_id or datetime.now().strftime(
        "%Y%m%d_historical_multimodal_" + ("smoke" if args.smoke_test else "full") + "_v001"
    )
    run_id, run = create_run(requested_id)
    smoke_limit = int(VALIDATION_CFG["smoke_clips"]) if args.smoke_test else None
    reference_path = run / "data" / "jvnv_manifest.csv"
    extract_and_manifest(zip_path, run / "data" / "jvnv_extracted", reference_path, smoke_limit)
    reference = pd.read_csv(reference_path)

    caches = model_cache_facts()
    audit_path = write_fidelity_audit(run, HISTORICAL_SCRIPT, HISTORICAL_CONFIG, caches)
    (run / "reports" / "reproducibility_metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "mode": "historical_multimodal",
                "run_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version,
                "platform": platform.platform(),
                "zip": str(zip_path),
                "sample_count": len(reference),
                "smoke_test": args.smoke_test,
                "full_run": False,
                "clean_clustering": False,
                "historical_classifier": str(HISTORICAL_SCRIPT),
                "historical_classifier_sha256": sha256(HISTORICAL_SCRIPT),
                "historical_config": str(HISTORICAL_CONFIG),
                "historical_config_sha256": sha256(HISTORICAL_CONFIG),
                "model_cache_facts": caches,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    configure_offline_runtime(run)
    module = load_historical_classifier(run)
    calls = instrument(module)
    scaler = module.fit_dataset_scaler([{"file_path": row["audio_path"]} for row in reference.to_dict("records")])
    feature_rows = extract_features(module, reference)
    predictions, pool = build_prediction_rows(module, reference, scaler, feature_rows, calls, args.smoke_test)
    prediction_path = run / "raw_predictions" / "jvnv_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    for row in predictions.to_dict("records"):
        if row.get("error_message"):
            with (run / "logs" / "inference_errors.log").open("a", encoding="utf-8") as log:
                log.write(f"{row.get('clip_id')}: {row['error_message']}\n")
                if row.get("_traceback"):
                    log.write(row["_traceback"] + "\n")
    evaluate(prediction_path, run / "reports", run / "figures")
    write_historical_evaluation_strata(predictions, run / "reports")
    if args.smoke_test:
        write_smoke_diagnostics(predictions, run / "reports")

    ready, full_completed = write_runtime_report(
        run,
        audit_path,
        module,
        calls,
        pool,
        predictions,
        args.smoke_test,
        len(reference),
    )
    status_file = (
        "historical_multimodal_smoke_status.json"
        if args.smoke_test
        else "historical_multimodal_run_status.json"
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_dir": str(run),
                "mode": "historical_multimodal",
                "sample_count": len(reference),
                "inferred_count": len(pool["inferred_clip_ids"]),
                "smoke_ready": ready,
                "full_run_performed": args.full,
                "audit": str(audit_path),
                "status": str(run / "reports" / status_file),
                "predictions": str(prediction_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if (ready if args.smoke_test else full_completed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
