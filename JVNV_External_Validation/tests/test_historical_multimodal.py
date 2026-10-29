from evaluate_jvnv_results_v2 import OUTPUT_LABELS
from run_historical_multimodal_validation import (
    HISTORICAL_CONFIG,
    HISTORICAL_SCRIPT,
    ORIGINAL,
    model_cache_facts,
)
import pandas as pd


def test_historical_mode_points_at_original_multimodal_implementation():
    assert HISTORICAL_SCRIPT == ORIGINAL / "multimodal_emotion_classifier.py"
    assert HISTORICAL_CONFIG == ORIGINAL / "emotion_classifier_config.json"
    assert HISTORICAL_SCRIPT.is_file()
    assert "character_emotion_pseudolabel.py" not in str(HISTORICAL_SCRIPT)


def test_historical_mode_keeps_all_seven_outputs():
    assert OUTPUT_LABELS == [
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Neutral",
        "Pleasant_Surprise",
        "Sad",
    ]


def test_pinned_audio_compatibility_preserves_historical_iemocap_order():
    from historical_speechbrain_compat import (
        ModelInputFilteringTokenizer,
        SPEECHBRAIN_LABEL_ORDER,
    )
    from verify_historical_model_compat import EXPECTED_AUDIO_MAPPING

    assert SPEECHBRAIN_LABEL_ORDER == ("neu", "ang", "hap", "sad")
    assert EXPECTED_AUDIO_MAPPING == {
        "neu": "Neutral",
        "ang": "Angry",
        "hap": "Happy",
        "sad": "Sad",
    }
    tokenizer = ModelInputFilteringTokenizer(
        lambda text, **kwargs: {
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
            "token_type_ids": [0, 0],
        },
        {"input_ids", "attention_mask"},
    )
    assert set(tokenizer("日本語").keys()) == {"input_ids", "attention_mask"}
    assert tokenizer.removed_inputs == {"token_type_ids"}


def test_pinned_speechbrain_checkpoint_is_present_and_fallback_is_not_assumed():
    facts = model_cache_facts()
    assert facts["speechbrain_primary"]["required_files_present"]
    assert facts["speechbrain_primary"]["revision"] == "117a9c3dff08be81a3628eecf6a66b547ec1659b"
    assert isinstance(facts["audio_fallback"]["snapshot_present"], bool)


def test_run_artifacts_are_created_under_validation_runs(tmp_path, monkeypatch):
    import run_historical_multimodal_validation as validation

    monkeypatch.setattr(validation, "HERE", tmp_path)
    run_id, run = validation.create_run("historical_multimodal_smoke_v001")
    assert run_id == "historical_multimodal_smoke_v001"
    assert run.parent == tmp_path / "results" / "runs"
    assert (run / "reports").is_dir()
    assert (run / "runtime").is_dir()


def test_numba_runtime_uses_run_isolated_cache_and_disables_svml(tmp_path, monkeypatch):
    import run_historical_multimodal_validation as validation

    monkeypatch.setattr(validation, "HERE", tmp_path)
    _, run = validation.create_run("historical_multimodal_smoke_v002")
    validation.configure_offline_runtime(run)

    import numba

    assert numba.config.USING_SVML is False
    assert numba.config.CACHE_DIR == str(run / "runtime" / "numba_cache")
    assert "TRANSFORMERS_CACHE" not in __import__("os").environ


def test_smoke_infers_all_twelve_balanced_records(monkeypatch):
    import run_historical_multimodal_validation as validation

    labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Pleasant_Surprise"]
    records = [
        {
            "clip_id": f"{label}_{index}",
            "audio_path": f"{label}_{index}.wav",
            "speaker_id": "JVNV_test",
            "reference_emotion": label,
        }
        for label in labels
        for index in range(2)
    ]
    reference = pd.DataFrame(records)
    feature_rows = [
        {"clip_id": row["clip_id"], "audio_path": row["audio_path"], "features": {}}
        for row in records
    ]
    inferred = []

    monkeypatch.setattr(validation, "quality_result", lambda *args: (True, ""))

    def fake_predict(module, record, features, scaler):
        assert set(record) == {"clip_id", "audio_path"}
        inferred.append(record["clip_id"])
        return {
            "clip_id": record["clip_id"],
            "predicted_emotion": record["clip_id"].rsplit("_", 1)[0],
            "confidence_status": "needs_review",
            "error_message": "",
        }

    monkeypatch.setattr(validation, "predict_one", fake_predict)
    predictions, pool = validation.build_prediction_rows(
        module=__import__("types").SimpleNamespace(CFG={"review":{"max_auto_duration_sec":10.0}}),
        reference=reference,
        scaler={},
        feature_rows=feature_rows,
        calls=__import__("collections").Counter(),
        smoke=True,
    )

    assert len(predictions) == 12
    assert len(inferred) == 12
    assert pool["eligible_count"] == 12
    assert set(pool["inferred_clip_ids"]) == set(reference["clip_id"])
    assert reference["reference_emotion"].value_counts().to_dict() == {
        label: 2 for label in labels
    }


def test_facebook_wav2vec2_metadata_is_recovered_from_read_only_historical_cache():
    from run_historical_multimodal_validation import FACEBOOK_WAV2VEC2_METADATA
    import json

    preprocessor=json.loads((FACEBOOK_WAV2VEC2_METADATA/"preprocessor_config.json").read_text(encoding="utf-8"))
    assert preprocessor["sampling_rate"] == 16000
    assert preprocessor["do_normalize"] is True
    assert preprocessor["return_attention_mask"] is False


def test_smoke_diagnostics_exclude_needs_review_from_operational_acceptance(tmp_path):
    import run_historical_multimodal_validation as validation

    predictions = pd.DataFrame(
        [
            {
                "predicted_emotion": "Angry",
                "reference_emotion": "Angry",
                "confidence_status": "auto_accept",
                "audio_scores": '{"Happy": 0.999, "Angry": 0.001}',
                "audio_prediction": "Happy",
                "text_prediction": "Angry",
                "prosody_prediction": "Angry",
            },
            {
                "predicted_emotion": "Sad",
                "reference_emotion": "Sad",
                "confidence_status": "needs_review",
                "audio_scores": '{"Happy": 0.998, "Sad": 0.002}',
                "audio_prediction": "Happy",
                "text_prediction": "Sad",
                "prosody_prediction": "Sad",
            },
        ]
    )

    diagnostics = validation.write_smoke_diagnostics(predictions, tmp_path)

    assert diagnostics["raw_predicted_label_accuracy"] == 1.0
    assert diagnostics["auto_accepted_coverage"] == 0.5
    assert diagnostics["auto_accepted_accuracy"] == 1.0
    assert diagnostics["operational_accuracy_full_pool"] == 0.5
    assert diagnostics["needs_review_count"] == 1
    assert diagnostics["speechbrain_happy_probability_ge_0_99_count"] == 2
