"""Load the pinned historical SpeechBrain checkpoint without SpeechBrain YAML imports."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import json


SPEECHBRAIN_LABEL_ORDER = ("neu", "ang", "hap", "sad")


class ModelInputFilteringTokenizer:
    """Remove tokenizer fields unsupported by the loaded classifier model."""

    def __init__(self, tokenizer: Any, accepted_inputs: set[str] | None):
        self._tokenizer = tokenizer
        self._accepted_inputs = accepted_inputs
        self.removed_inputs: set[str] = set()

    def __call__(self, *args, **kwargs):
        encoded = self._tokenizer(*args, **kwargs)
        for key in tuple(encoded.keys()):
            if self._accepted_inputs is not None and key not in self._accepted_inputs:
                self.removed_inputs.add(key)
                encoded.pop(key)
        return encoded

    def __getattr__(self, name: str):
        return getattr(self._tokenizer, name)


def load_pinned_speechbrain_adapter(
    snapshot: Path,
    sample_rate: int,
    requested_device: str = "cpu",
    feature_extractor_snapshot: Path | None = None,
) -> tuple[Any, str]:
    """Build the checkpoint's declared graph from its existing local weights."""
    import torch
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    if feature_extractor_snapshot is None:
        raise FileNotFoundError("facebook/wav2vec2-base metadata snapshot was not supplied")
    feature_extractor_snapshot = Path(feature_extractor_snapshot)
    preprocessor_path = feature_extractor_snapshot / "preprocessor_config.json"
    config_path = feature_extractor_snapshot / "config.json"
    if not preprocessor_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"Required facebook/wav2vec2-base metadata is incomplete: {feature_extractor_snapshot}"
        )
    preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    encoder_config = Wav2Vec2Config.from_json_file(str(config_path))
    metadata_rate = int(preprocessor["sampling_rate"])
    if int(sample_rate) != metadata_rate:
        raise ValueError(
            f"Historical sample-rate mismatch: classifier={sample_rate}, metadata={metadata_rate}"
        )
    do_normalize = bool(preprocessor["do_normalize"])

    wav2vec2_state = torch.load(
        snapshot / "wav2vec2.ckpt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    wav2vec2 = Wav2Vec2Model(encoder_config)
    wav2vec2_load = wav2vec2.load_state_dict(
        {key.removeprefix("model."): value for key, value in wav2vec2_state.items()},
        strict=True,
    )

    head_state = torch.load(
        snapshot / "model.ckpt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    head = torch.nn.Linear(768, len(SPEECHBRAIN_LABEL_ORDER), bias=False)
    head.load_state_dict({"weight": head_state["0.w.weight"]}, strict=True)

    device = (
        requested_device
        if requested_device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    class EncoderWithOutputNormalization(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, waveform):
            if do_normalize:
                waveform = torch.nn.functional.layer_norm(waveform, waveform.shape[1:])
            hidden = self.model(input_values=waveform).last_hidden_state
            # SpeechBrain Wav2Vec2(output_norm=True) uses out.shape[1:], i.e.
            # one normalization over the complete time x feature plane.
            return torch.nn.functional.layer_norm(hidden, hidden.shape[1:])

    class MeanStatisticsPooling(torch.nn.Module):
        def forward(self, embeddings, relative_length):
            valid_lengths = torch.clamp(
                torch.round(relative_length * embeddings.shape[1]).long(),
                min=1,
                max=embeddings.shape[1],
            )
            pooled = []
            for batch_index, valid_length in enumerate(valid_lengths):
                pooled.append(
                    embeddings[batch_index, : int(valid_length.item())].mean(dim=0)
                )
            mean = torch.stack(pooled, dim=0)
            # Exact StatisticsPooling(return_std=False) perturbation.
            noise = torch.randn(mean.size(), device=mean.device)
            noise -= torch.min(noise)
            noise /= torch.max(noise)
            noise = 1e-5 * ((1 - 9) * noise + 9)
            return (mean + noise).unsqueeze(1)

    class LabelEncoder:
        ind2lab = dict(enumerate(SPEECHBRAIN_LABEL_ORDER))

        @staticmethod
        def decode_torch(indices):
            return [SPEECHBRAIN_LABEL_ORDER[int(index)] for index in indices.reshape(-1)]

    wav2vec2.to(device).eval()
    head.to(device).eval()
    adapter = SimpleNamespace(
        device=device,
        mods=SimpleNamespace(
            wav2vec2=EncoderWithOutputNormalization(wav2vec2),
            avg_pool=MeanStatisticsPooling(),
            output_mlp=head,
        ),
        hparams=SimpleNamespace(
            sample_rate=metadata_rate,
            softmax=torch.nn.Softmax(dim=-1),
            label_encoder=LabelEncoder(),
        ),
        fidelity=SimpleNamespace(
            feature_extractor_snapshot=str(feature_extractor_snapshot),
            sampling_rate=metadata_rate,
            do_normalize=do_normalize,
            return_attention_mask=bool(preprocessor.get("return_attention_mask", False)),
            output_normalization_axes="time_and_feature",
            pooling="masked_mean_plus_original_statistics_pooling_noise",
            head_shape=[len(SPEECHBRAIN_LABEL_ORDER), 768],
            label_order=list(SPEECHBRAIN_LABEL_ORDER),
        ),
    )
    return adapter, str(wav2vec2_load)
