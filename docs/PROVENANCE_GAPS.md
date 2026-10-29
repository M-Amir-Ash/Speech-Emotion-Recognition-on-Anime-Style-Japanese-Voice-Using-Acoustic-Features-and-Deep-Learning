# Provenance gaps

1. **UNVERIFIED historical model ledger**: no preserved per-run `_model_status_report.csv`, package lock, cache revision record, or console log accompanies the 68,860 manifest. SpeechBrain versus fallback activation cannot be proven from configuration alone.
2. **UNVERIFIED human-review completion**: `review_status=manual_review` routes 65,050 rows for review; it is not evidence that all were reviewed or relabelled by a person.
3. **UNVERIFIED source-media redistribution rights**: anime-derived audio, transcripts, filenames, and character/media references require legal/manual review. Git LFS does not solve copyright restrictions.
4. **UNVERIFIED JVNV redistribution rights**: the local data appears third-party. Publish no audio until the dataset's exact license and acquisition source are documented.
5. **UNVERIFIED paper-to-artifact commit**: no immutable Git commit/tag ties the conference submission to the exact notebook/result tree.
6. **CONFLICTING result families**: historical, newer, multiseed, neural-only, text-only, and selection-sensitivity artifacts coexist. Each requires protocol/version labels.
7. **Absolute local paths**: manifests reveal a personal Windows directory and copyrighted source names. Sanitization must retain stable `clip_id`, split, label, source-group hash, and artifact hash while removing local roots.
8. **Environment incompleteness**: historical BAT points to `autogen311`; FFmpeg is absent from PATH; local Mel-CNN and embedding classifier paths are missing; XGBoost/openSMILE are absent in that environment. A generated lockfile would not reconstruct unknown historical revisions.
9. **Notebook mutability**: E notebook writes outputs/caches/checkpoints into original paths when executed. Release should provide a documented read-only historical notebook plus configurable scripts, without pretending the historical notebook is path-portable.
10. **JVNV reconstruction differences**: compatibility adapter, isolated caches, CPU ASR, JVNV-fitted unlabeled scaler, and missing source transcripts differ from the historical anime workflow.

Resolution priorities: document licenses; produce sanitized manifests; pin a publication environment from verified imports; add a release tag and checksums; preserve the historical code unchanged under a clearly labelled `legacy/` area; and document every intentional portability wrapper separately.
