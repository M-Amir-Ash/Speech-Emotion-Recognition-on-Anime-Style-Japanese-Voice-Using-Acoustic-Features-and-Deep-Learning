# Anonymous speaker-cluster metadata analysis

This directory contains an exploratory voice-similarity analysis of the frozen 1,969 clips. It does not identify characters or people, modify emotion labels, copy/reorganize audio, regenerate splits, or train a speech-emotion model.

## Implementation

- SpeechBrain `speechbrain/spkrec-ecapa-voxceleb`
- One normalized 192-dimensional ECAPA embedding per clip
- Global anonymous IDs such as `SPK_0001`
- Conservative clustering within a derived anime/season domain
- Average-linkage agglomerative clustering with cosine distance threshold 0.42, matching the existing project capability
- The number of clusters is produced by the distance threshold; no speaker count is supplied
- Centroid cosine similarity and nearest-other-centroid margin are diagnostic statistics, not calibrated identity probabilities

The script can reuse an existing ECAPA vector only when its source audio hash exactly matches the frozen clip. Missing vectors are computed with the locally installed SpeechBrain model. The old 32-dimensional acoustic fallback is prohibited.

Run from `Anime_SER_More_Sample`:

```powershell
C:\Users\ashsh\anaconda3\envs\autogen311\python.exe speaker_analysis\run_speaker_analysis.py
```

Completed final reports are protected from overwrite. Cached embeddings may be resumed after an interrupted first run.

## Interpretation

These clusters are estimated anonymous voice-similarity groups, not verified speaker identities and not character recognition. Even if estimated clusters do not cross splits, that would not prove speaker independence. The canonical experiment remains described as **source-media/episode group-disjoint**.

The ECAPA model was trained on a different speech domain. Expressive acting, very short clips, background music/effects, and cross-character voice changes can cause both false splits and false merges. Clustering is restricted within conservatively derived anime/season domains, so it also cannot detect the same actor or similar voice across different domains.

## Observed reliability

The completed 0.42-threshold analysis produced 1,526 similarity groups, of which 1,305 (85.52%) were singletons. This fails the project's conservative maximum singleton-cluster share of 25%. Even the looser 0.52 sensitivity threshold left 805 singleton clips.

Accordingly, `SPK_####` values are retained only as traceable exploratory similarity-group IDs. The estimated cluster count must not be interpreted as the number of real speakers. The 92 groups and 353 clips estimated to cross splits indicate possible speaker overlap, but they are not verified identities or a precise overlap rate.
