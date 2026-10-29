# Historical multimodal fidelity audit

Created before inference: 2026-10-24T00:18:55.747919+00:00

## Verified matches

- The implementation loaded for this run is the original project file
  `<PROJECT_ROOT>\Speech Emotion Recognition on Anime-Style Japanese Voice Using Acoustic Features and Deep Learning\Training Data\Auto Split More Sample\multimodal_emotion_classifier.py` (SHA-256 `02eec66f6655800615c8225d25d6dec654159849f67c98f4da6c1850283fd3a2`), not
  `frozen_source/character_emotion_pseudolabel.py`.
- Its adjacent frozen configuration is `<PROJECT_ROOT>\Speech Emotion Recognition on Anime-Style Japanese Voice Using Acoustic Features and Deep Learning\Training Data\Auto Split More Sample\emotion_classifier_config.json`
  (SHA-256 `ba4b6de43c19cace3ad709f849600dfdad859c97a5661a02b1c67a84d43d5531`). Runtime fusion weights, decision thresholds,
  labels, and mappings are not changed.
- The executable classifier path uses `extract_features`,
  `transcribe_if_needed`, `audio_emotion_scores`, `text_emotion_scores`,
  `prosody_emotion_scores`, `weighted_fusion`, and `auto_label_decision`.
- The original classifier's `audio_emotion_scores` remains the scoring path.
  For validation only, its loader is replaced with a local adapter that loads
  `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` from the pinned snapshot `117a9c3dff08be81a3628eecf6a66b547ec1659b`:
  `True`. The adapter
  uses the declared Wav2Vec2 encoder, waveform normalization, full time-by-feature
  output normalization, StatisticsPooling masked mean/noise, 4-way linear head,
  softmax, sample rate, and label order. The `facebook/wav2vec2-base` feature
  extractor metadata was recovered from the existing read-only historical cache
  at `<PROJECT_ROOT>\Speech Emotion Recognition on Anime-Style Japanese Voice Using Acoustic Features and Deep Learning\Training Data\Auto Split More Sample\wav2vec2_checkpoints\models--facebook--wav2vec2-base\snapshots\0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8`; it specifies
  16 kHz, `do_normalize=true`, and no returned attention mask. This bypasses SpeechBrain's YAML
  importer and unrelated optional `k2` import; trained weights are unchanged.
- The executable code does not call the mel-CNN or embedding-emotion model
  loaders. The configuration's `feature_extractors` list is therefore not
  treated as evidence that those branches ran. No Bagus or Dpng model is loaded.
- Text model cache: `True`; faster-whisper
  small cache: `True`. Offline mode prevents
  downloads. The configured fallback cache is present:
  `False`.

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
