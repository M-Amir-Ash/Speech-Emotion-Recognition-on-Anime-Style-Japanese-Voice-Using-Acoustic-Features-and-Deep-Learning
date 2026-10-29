# JVNV primary validation protocol v2

## Research question

How well does the reconstructed automatic Japanese emotion-labeling pipeline agree with independently labeled Japanese emotional speech, without acoustic sample selection?

## Primary analysis

The complete 1,615-recording JVNV population is the primary evaluation set. No clean-clustering status, distance, acoustic-overlap status, confidence-based sample selection, or JVNV reference-label criterion may remove a recording from the primary ledger or denominator.

The model retains seven possible outputs: Angry, Disgust, Fear, Happy, Neutral, Pleasant_Surprise, and Sad. JVNV Surprise is mapped prospectively to Pleasant_Surprise. Because JVNV has no Neutral reference class, a Neutral output is retained and counted as an error.

Primary operational accuracy and macro-F1 use the full population. Rejected outputs are operational abstentions and failed inference is an explicit failure outcome; both contribute zero correct predictions. Prediction coverage and operational coverage are reported separately. Accuracy and macro-F1 among successfully returned labels are conditional diagnostics and cannot replace the primary metrics.

Every recording is written to `jvnv_primary_all_samples_ledger.csv`, including correct and incorrect predictions, Neutral outputs, abstentions, inference failures, and any acoustically overlapping examples.

## Secondary acoustic-separability experiment

Clean clustering is disabled by default and runs only with `--run-clean-clustering`. It is a secondary study of acoustic separability and training-sample selection. Acoustic overlap is not evidence that a reference label or model prediction is incorrect. Secondary subset statistics do not alter the primary dataset, confusion matrix, accuracy, macro-F1, or coverage.

## Frozen methodology and scope

Models, revisions, fusion weights, confidence thresholds, mappings, and labeling rules remain frozen. No model is retrained and no parameter is selected using JVNV labels. A corrected 12-recording smoke test must complete before the full-run guard can be released.

Results are external Japanese acted-speech benchmark validation. They are not direct verification of anime-domain pseudo-label accuracy.
