# Methodology and limitations

This was the single authorized frozen held-out test evaluation. Input hashes, Stage 1 selector, Stage 2 model bundles, embedded preprocessing, and selected configurations were verified before any test scoring. No fitting, model selection, threshold change, or feature extraction occurred. The frozen linear SVM uses `C=1.0`; the frozen XGBoost model uses 300 trees, depth 3, and learning rate 0.05.

## Label-conditioned selector

The selector is **LABEL-CONDITIONED**. For each test clip, its inherited pseudo-emotion label selects the corresponding class centroid and class-specific distance/margin thresholds. Therefore, the 455-clip passing cohort is not produced by a deployable label-free filter. It is a retrospective, label-conditioned subset used to quantify sensitivity to the historical acoustic-selection procedure. Test labels were not used to alter the selector, models, or method.

The second evaluation includes all 13,462 quality-pass test clips. These predictions come from models trained only on the 1,549 selector-passing training clips, so the full-pool result is an out-of-selection-distribution stress test, not performance from full-pool training.

Uncertainty uses a corrected duplicate-index-safe source-group cluster bootstrap with 2,000 replicates and seed 20261014. Source-group separation is not speaker-disjointness. Targets remain inherited pseudo-labels, not independent human ground truth. The original selector's MFCC-9 statistics were reconstructed code-faithfully but could not be historically numerically verified. Upstream pseudo-label construction may preserve global dependencies across partitions.
