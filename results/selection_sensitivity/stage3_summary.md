# Phase 3D-B2 Stage 3 final test summary

## Manuscript-ready factual findings

The frozen training-only acoustic selector retained 455 of 13,462 held-out quality-valid clips (3.38%); 13,007 clips were outside the label-conditioned passing subset. On this subset, MFCC-117 + linear SVM achieved macro-F1 0.6917, weighted F1 0.7068, and accuracy 0.7077; eGeMAPS-175 + XGBoost achieved macro-F1 0.7421, weighted F1 0.7523, and accuracy 0.7538. Across all 13,462 held-out clips, the corresponding macro-F1 values were 0.2115 and 0.2470.

The association between selector passage and higher pseudo-label agreement is large. It supports the conclusion that reported performance is strongly conditioned on acoustic sample selection. It does not by itself establish that selection caused the performance difference: passing and nonpassing clips differ systematically, and labels are inherited pseudo-labels. Full-pool results specifically reflect models trained only on the selected 1,549 training clips.

Source-group cluster-bootstrap 95% intervals were 0.6885–0.7849 for the XGBoost selective macro-F1 and 0.6397–0.7311 for the SVM selective macro-F1. All selective class supports were at least 40, so no class triggered the predefined `<30` warning; nevertheless, the smaller classes remain imprecise.

## Material class findings

- Pleasant Surprise was the weakest selective class: F1 0.5432 for XGBoost and 0.4198 for SVM, despite 44 passing cases.
- Angry was also comparatively weak: F1 0.6897 for XGBoost and 0.6346 for SVM on 46 passing cases.
- Disgust had 40 passing cases and high selective F1 (0.8649 XGBoost; 0.7792 SVM), but this is pseudo-label agreement inside a label-conditioned subset, not independent validation of Disgust perception.
- Across the full test population, every class had low F1; the highest class F1 was only 0.3338 for XGBoost and 0.2766 for SVM. No difficult clips were removed after viewing these results.

Selection coverage itself was class-dependent, ranging from 1.28% for Pleasant Surprise to 7.78% for Disgust. This uneven retention further limits comparisons of selective class scores.

## Evidence-versus-claims matrix

| Claim | Evidence | Supported wording | Unsupported wording |
|---|---|---|---|
| Frozen evaluation integrity | Verified hashes, unchanged model parameters, zero source-group overlap | One-time evaluation of frozen artifacts | Independently replicated pipeline |
| Selective performance | Label-conditioned passing-test metrics and denominators | Agreement with inherited pseudo-labels on a label-conditioned subset | Deployable label-free filtering performance |
| Full-pool performance | Predictions for all 13,462 test clips | Out-of-selection-distribution stress test | Model trained on the full candidate population |
| Selection conditioning | Large observed selective/full-pool metric difference | Performance is strongly associated with selector passage | Acoustic selection causally explains the entire difference |
| Emotion validity | No independent human truth in this stage | Evaluation targets are inherited pseudo-labels | Human-validated emotion recognition accuracy |
| Generalization | Train/validation/test source groups are disjoint | Source-group-disjoint evaluation | Speaker-independent generalization |
| Historical comparison | Historical 0.7517/0.7474 under another protocol | Descriptive contextual comparison only | Direct paired improvement or decrease |

## Terminal status

The authorized one-time test evaluation is complete. No model or threshold was changed after viewing test outcomes. Further testing or tuning requires separate authorization.
