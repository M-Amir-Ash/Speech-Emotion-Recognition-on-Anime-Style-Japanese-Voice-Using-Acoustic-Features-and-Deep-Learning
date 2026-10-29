# Experiment lineage

1. Folder A creates 68,860 pseudo-label records, retains 66,084 quality-valid rows, and applies label-conditioned acoustic selection.
2. Folder E contains the canonical 1,969-record benchmark and its group-disjoint train/validation/test manifests.
3. Folder D is a separate selection-sensitivity study over the 66,084-record pool, with saved predictions for both the selected-test cohort and full test population.
4. Folder F contains the 1,615-row JVNV external-validation prediction archive.

Folder C (`_Anime_SER`) is a junction alias of Folder E and is not duplicated. Folder B is a derivative report folder and is not authoritative.
