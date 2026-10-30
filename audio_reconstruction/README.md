# Selected-1969 audio reconstruction

This package reconstructs candidate WAV files from **user-supplied, lawfully
obtained source media**. It contains no source media and performs no download,
DRM circumvention, source separation, labeling, or model inference.

## Provenance coverage

`reconstruction_manifest.csv` records the existing clip ID, output filename,
split, emotion label, source-media identifier, segment bounds, and expected
SHA-256 for all 1,969 selected clips (1,313 train, 261 validation, 395 test).
The segment bounds are relative to the processed source identified by
`source_media_id`.

All 1,969 rows are **conditionally reconstructable from the exact historical
`__Vocals` stem**. None can presently be claimed byte-reconstructable directly
from an original episode file because the saved evidence does not contain the
historical stem SHA-256, source-separation model/settings, or exact tool
versions. These missing facts are recorded per row; they are not inferred.

## Requirements

- Python 3.9+
- `pydub`
- FFmpeg available on `PATH`
- The exact historical processed stems, or lawful media prepared identically

Install only the runtime dependency:

```bash
python -m pip install pydub
```

## Source mapping

Copy `source_media_map_template.csv` to a private working location and fill its
`local_path` column. Do not commit media paths or copyrighted media. Each path
must identify the processed media corresponding to `source_media_id`; matching
is never guessed from directory contents.

## Reconstruct and verify

Dry-run validation (creates no audio):

```bash
python reconstruct_selected_audio.py \
  --source-map /private/source_media_map.csv \
  --output-dir /private/reconstructed \
  --dry-run
```

Reconstruct candidate files and compare each result with its saved SHA-256:

```bash
python reconstruct_selected_audio.py \
  --source-map /private/source_media_map.csv \
  --output-dir /private/reconstructed \
  --report /private/reconstruction_results.csv
```

Outputs are written as `<split>/<emotion>/<filename>`. Existing files are not
overwritten unless `--overwrite` is supplied. A row is verified only when its
computed SHA-256 equals `expected_sha256`; merely producing a playable WAV is
not verification.

The historical evidence indicates normalization to mono, 16,000 Hz, 16-bit
PCM, followed by pydub millisecond slicing and WAV export. The script follows
that path. Encoder/resampler and source-stem differences can still change
bytes, so hash mismatches must remain mismatches rather than being waived.

## Rights boundary

The metadata and reconstruction code do not grant rights to any anime source.
Users must provide media they may lawfully possess and process. The script has
no network downloader and does not handle encrypted or DRM-protected sources.

Historical implementation references retained elsewhere in this repository:

- `Training Data/Auto Split More Sample/extract_segment_multimodal_pipeline.py`
- `Training Data/Auto Split More Sample/segment_audio.py`

