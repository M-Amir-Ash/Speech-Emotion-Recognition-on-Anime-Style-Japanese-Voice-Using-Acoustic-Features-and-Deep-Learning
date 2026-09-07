# Auto Split More Sample

This directory contains the reproducible code for media extraction, dialogue
segmentation, Japanese multimodal emotion pseudo-labeling, quality control,
anonymous speaker clustering, and multi-feature clean selection.

The local runtime directories `input/`, `input_audio/`, `output/`, `.cache/`, and
`wav2vec2_checkpoints/` are intentionally excluded from Git because together they
contain large media, generated workspaces, and downloaded model files.

Configure external credentials only through environment variables. In
particular, use `OPENAI_API_KEY` and the standard Hugging Face authentication
mechanism. Do not create or commit plaintext token files.

The main interactive entry point is `run_emotion_workflow_gpu311.bat`, which calls
`emotion_workflow_runner.py`. Review the configuration JSON files before running
the pipeline on new media.

The legacy `ser_dual_training_framework_anime_transfer_only.ipynb` is retained as
historical reference and is not the canonical 1,969-sample experiment. The
canonical experiment and reporting evidence are under `Anime_SER_More_Sample/`.
