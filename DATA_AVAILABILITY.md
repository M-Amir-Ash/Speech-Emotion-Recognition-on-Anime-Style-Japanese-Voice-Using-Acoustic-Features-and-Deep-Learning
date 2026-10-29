# Data availability

This repository publishes the implementation, configuration, compact manifests,
saved metrics, figures, and thesis evidence package. It does not publish the raw
or derived anime audio collection.

The following local artifacts are intentionally excluded:

- source anime video and separated vocal/instrumental media;
- segmented and classified audio workspaces;
- the 1,969 selected WAV clips;
- downloaded Hugging Face and SpeechBrain model caches;
- log-Mel and speaker-embedding caches;
- API keys, authentication tokens, and local environment files.

The exclusions are necessary because the local construction workspace is about
90.8 GiB, contains files above GitHub's regular file limit, and may contain media
whose redistribution rights must be evaluated separately.

To reproduce the workflow, provide legally obtained input media locally, install
the dependencies, configure credentials through environment variables, and use
the scripts in `Training Data/Auto Split More Sample/`. Place the final selected
seven-emotion dataset under `Anime_SER_More_Sample/sample/` before running the
training implementation.

The frozen 1,969-sample membership and provenance fields used by the reported
experiment are retained in the train, validation, and test manifests under
`Anime_SER_More_Sample/output/` and in the organized `ready/splits/` package.


## Rights review and verified inventory

No documented permission authorizing redistribution of the anime-derived audio or JVNV audio was located. Repository privacy does not supply that permission. Therefore, zero audio files are included. The 1,969 selected WAVs were located locally (127,497,356 bytes), as was the complete 68,860-record classified collection (4,637,188,592 bytes). This release provides sanitized metadata and saved predictions instead. Access to audio requires the rights holder's authorization or independent lawful acquisition under the relevant dataset/source terms.
