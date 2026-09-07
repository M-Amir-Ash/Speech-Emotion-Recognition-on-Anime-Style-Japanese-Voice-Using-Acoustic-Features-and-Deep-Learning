# Proposal Duration Comparison

## Exact proposal wording

Proposal PDF page 27 (printed page 22):

> The planned duration of each audio segment is approximately 1 to 6 seconds.

The proposal also states:

> During the implementation stage, the final duration range may be adjusted on the basis of the quality of the collected segments.

## Actual implementation behavior

The upstream segmentation pipeline defines both `MIN_SEGMENT_SEC` and
`MIN_EMOTION_CLIP_SEC` as **0.65 seconds**. The full workflow exposes the same
0.65-second default. A clip can enter Japanese emotion classification when its
duration is at least that configured value and the Japanese-language, speech,
music/noise, and audio-quality gates pass. The final clean-cluster selector
requires passed audio quality and acoustic separability but does not add a
1.00-second exclusion.

Therefore, 1 second was a planning guideline rather than a strict technical
threshold. Clips from 0.65 through 0.99 seconds were intentionally allowed by
the implemented policy.

## Frozen dataset behavior

- 1,821/1,969 clips (92.483%) are within 1.00–6.00 seconds.
- 148/1,969 clips (7.517%) are below 1.00 second.
- No selected clip exceeds 6.00 seconds.
- The actual range is 0.650–5.340 seconds.
- The highest within-class short-clip rate is sad at
  40/154
  (25.974%).

The short clips are not evenly distributed across emotions. Sad (40), Neutral
(34), and Disgust (33) together account for 107/148 (72.297%) of all clips below
one second. Relative to each class size, Sad (25.974%) and Disgust (22.148%) are
the clearest concentrations, compared with the dataset-wide short-clip rate of
7.517%.

## Classification

**Documented implementation refinement**, not an implementation bug.

This classification is supported because the proposal describes 1–6 seconds as
an approximate planned range and explicitly permits the final range to be adjusted
based on segment quality. The implemented 0.65-second minimum is explicit and
consistently passed through the workflow. The deviation should nevertheless be
reported transparently in the thesis rather than describing the realized dataset
as strictly 1–6 seconds.

## Recommended paper wording

> Most selected utterances (92.5%) were within the planned 1–6 s duration interval. A small number of shorter utterances were retained when they satisfied the remaining Japanese-dialogue, audio-quality, and clean-selection criteria.
