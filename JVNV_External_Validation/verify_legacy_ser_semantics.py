"""Low-memory audit of the recovered legacy Japanese SER head semantics."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from frozen_source.legacy_wav2vec2_ser import LegacyClassificationHead


ROOT = Path(__file__).resolve().parent
REVISION = "26edb2426828afabe63957bafdc9d5de383fbf11"
SNAPSHOT = (
    ROOT
    / "model_cache/huggingface/hub/models--Bagus--wav2vec2-xlsr-japanese-speech-emotion-recognition/snapshots"
    / REVISION
)


def reference_pool(hidden: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "mean":
        return torch.mean(hidden, dim=1)
    if mode == "sum":
        return torch.sum(hidden, dim=1)
    if mode == "max":
        return torch.max(hidden, dim=1)[0]
    raise ValueError(mode)


def main() -> None:
    config = json.loads((SNAPSHOT / "config.json").read_text(encoding="utf-8"))
    tensor_path = SNAPSHOT / "model.safetensors"
    with safe_open(tensor_path, framework="pt", device="cpu") as tensors:
        shapes = {name: list(tensors.get_slice(name).get_shape()) for name in (
            "classifier.dense.weight",
            "classifier.dense.bias",
            "classifier.out_proj.weight",
            "classifier.out_proj.bias",
        )}

    class Config:
        hidden_size = config["hidden_size"]
        final_dropout = config["final_dropout"]
        num_labels = len(config["id2label"])

    torch.manual_seed(20261023)
    head = LegacyClassificationHead(Config()).eval()
    hidden = torch.randn(2, 7, Config.hidden_size)
    pooled = reference_pool(hidden, config["pooling_mode"])
    actual_logits = head(pooled)
    expected_logits = head.out_proj(head.dropout(torch.tanh(head.dense(head.dropout(pooled)))))
    actual_probabilities = torch.softmax(actual_logits, dim=-1)
    expected_probabilities = torch.exp(actual_logits) / torch.exp(actual_logits).sum(dim=-1, keepdim=True)

    label_order = [config["id2label"][str(i)] for i in range(Config.num_labels)]
    canonical_order = [{"ang": "Angry", "joy": "Happy", "neu": "Neutral", "sad": "Sad"}[x] for x in label_order]
    result = {
        "checkpoint_revision": REVISION,
        "original_transformers_version_metadata": config.get("transformers_version"),
        "pooling_mode": config["pooling_mode"],
        "attention_mask_pooling_note": "The original legacy implementation forwards the mask to Wav2Vec2 but pools every returned feature frame without applying a separate masked mean.",
        "hidden_size": Config.hidden_size,
        "num_labels": Config.num_labels,
        "raw_label_order": label_order,
        "canonical_label_order": canonical_order,
        "checkpoint_head_shapes": shapes,
        "synthetic_batch_shape": list(hidden.shape),
        "pooled_shape": list(pooled.shape),
        "logits_shape": list(actual_logits.shape),
        "head_equation_max_abs_difference": float((actual_logits - expected_logits).abs().max()),
        "softmax_max_abs_difference": float((actual_probabilities - expected_probabilities).abs().max()),
        "probability_row_sum_max_abs_error": float((actual_probabilities.sum(dim=-1) - 1).abs().max()),
        "semantic_checks_passed": bool(
            actual_logits.shape == (2, 4)
            and torch.equal(actual_logits, expected_logits)
            and torch.allclose(actual_probabilities, expected_probabilities, atol=1e-7, rtol=1e-7)
            and label_order == ["ang", "joy", "neu", "sad"]
            and shapes["classifier.dense.weight"] == [1024, 1024]
            and shapes["classifier.out_proj.weight"] == [4, 1024]
        ),
        "scope": "Low-memory semantic verification of pooling, classification head, ordering, dimensions, and probability calculation. It does not claim an end-to-end backbone forward pass.",
    }
    out = ROOT / "results/reports/legacy_japanese_ser_semantic_verification.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
