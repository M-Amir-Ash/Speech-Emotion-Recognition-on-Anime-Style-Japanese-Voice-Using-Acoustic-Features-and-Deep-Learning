# Dataset card

The historical pseudo-label manifest contains 68,860 rows. Its on-disk classified audio is 4,637,188,592 bytes: 66,084 quality-valid recordings (4,514,039,280 bytes) and 2,776 rejected recordings (123,149,312 bytes). The final selected cohort contains 1,969 WAV files totaling 127,497,356 bytes, split 1,313/261/395.

No audio is distributed here. Stable clip IDs, hashes, labels, split assignments, quality fields, and scores are retained where available. Transcripts, media names, absolute paths, and source filenames were removed from the release manifests. Speaker separation is unresolved: the selected manifests record `UNKNOWN_SPEAKER`, and no reliable character identifiers are available.
