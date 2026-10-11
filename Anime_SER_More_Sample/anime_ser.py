from pathlib import Path
import hashlib, json, os, random, re, warnings

# Windows/joblib compatibility: set before importing scikit-learn/joblib.
# This prevents loky from invoking the removed WMIC command to count cores.
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
SAFE_PARALLEL_JOBS = 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import scipy.signal as signal
from IPython.display import display, HTML

from sklearn.model_selection import GroupShuffleSplit, ParameterGrid
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_fscore_support
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]
TARGET_SR = 16_000
DURATION_SECONDS = 4
N_SAMPLES = TARGET_SR * DURATION_SECONDS
N_MELS = 128
TEST_SIZE = 0.20       # Proposal §3.2: held-out testing data
VALIDATION_SIZE = 0.12 # Proposal §3.2: 15% of the remaining 80% = 12% overall
MFCC_CORRELATION_THRESHOLD = 0.95
USE_TRAIN_OVERSAMPLING = False  # Not specified by the proposal; keep off for the primary experiment.
USE_EXPERIMENTAL_AUDIO_ENHANCEMENT = False  # Band-pass/pre-emphasis are outside the proposal.
BATCH_SIZE = 32
EPOCHS = 25
TUNING_EPOCHS = 10
RUN_HYPERPARAMETER_TUNING = True
OUTPUT_DIR = Path(r"D:\Telkom University Kuliah\Publikasi Paper\Speech Emotion Recognition on Anime-Style Japanese Voice Using Acoustic Features and Deep Learning\Speech Emotion Recognition on Anime-Style Japanese Voice Using Acoustic Features and Deep Learning\Anime_SER_More_Sample\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "logmel_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RUN_STATUS_PATH = OUTPUT_DIR / "run_status.json"
RUN_STATUS_PATH.write_text(json.dumps({
    "execution_status": "incomplete",
    "message": "Notebook started but has not reached the final cell. Do not treat partial CSV files as final results.",
    "run_hyperparameter_tuning": RUN_HYPERPARAMETER_TUNING,
    "use_train_oversampling": USE_TRAIN_OVERSAMPLING,
}, indent=2), encoding="utf-8")

print("Device:", DEVICE)

# Primary dataset copied from Training Data/Auto Split More Sample.
# Resolve it explicitly so launching Python from another directory cannot select
# an older dataset by accident.
PROJECT_ROOT = Path(r"D:\Telkom University Kuliah\Publikasi Paper\Speech Emotion Recognition on Anime-Style Japanese Voice Using Acoustic Features and Deep Learning\Speech Emotion Recognition on Anime-Style Japanese Voice Using Acoustic Features and Deep Learning")
PACKAGE_ROOT = PROJECT_ROOT / "Anime_SER_More_Sample"
PROPOSAL_PDF = Path(r"D:\Telkom University Kuliah\Kuliah S1\Tugas S1\Semester 6\Proposal\Speech_Emotion_Recognition_on_Anime_Style_Japanese_Voice_Using_Acoustic_Features_and_Deep_Learning_Proposal_Thesis_FIF_Telkom_University__Tel_U____PHN.pdf")
AUTO_SPLIT_ROOT = PROJECT_ROOT / "Training Data" / "Auto Split More Sample"
LOCAL_ANIME_SAMPLE_ROOT = PACKAGE_ROOT / "sample"
ANIME_SAMPLE_ROOT = LOCAL_ANIME_SAMPLE_ROOT
ANIME_SAMPLE_MANIFEST = ANIME_SAMPLE_ROOT / "_cluster_clean_manifest.csv"
PIPELINE_WORKSPACE = AUTO_SPLIT_ROOT / "output" / "pipeline_workspace"
SEGMENT_MANIFEST = PIPELINE_WORKSPACE / "segmented" / "japanese_dialogue" / "_segments_manifest.csv"
QUALITY_CONTROL_SUMMARY = PIPELINE_WORKSPACE / "multimodal_emotion_classified" / "_quality_control_summary.csv"
WORKFLOW_AUDIT_SUMMARY = PIPELINE_WORKSPACE / "audit" / "_workflow_audit_summary.csv"
CLUSTER_COUNT_REPORT = ANIME_SAMPLE_ROOT / "_cluster_clean_count_report.csv"
RESOURCE_NAME = "anime_sample"

if not ANIME_SAMPLE_ROOT.exists():
    raise FileNotFoundError(f"Missing {RESOURCE_NAME}: {ANIME_SAMPLE_ROOT}")

print("Resource name:", RESOURCE_NAME)
print("Audio root:", ANIME_SAMPLE_ROOT)
print("Provenance manifest:", ANIME_SAMPLE_MANIFEST)
print("Source metadata:", SEGMENT_MANIFEST)
print("Proposal source:", PROPOSAL_PDF)

LABEL_DIRS = {
    "Angry": "angry",
    "Disgust": "disgust",
    "Fear": "fear",
    "Happy": "happy",
    "Neutral": "neutral",
    "Pleasant_Surprise": "pleasant_surprise",
    "Sad": "sad",
}

def infer_source_group(path: Path) -> str:
    # Infer a conservative show/episode group from an Auto Split segment name.
    stem = path.stem
    patterns = [
        r"(?i)(Otakudesu(?:_net)?_.+?--\d{1,3}(?:_720p)?)",
        r"(?i)(Kusonime__.+?__\d{1,3})",
        r"(?i)((?:Otakudesu(?:_net)?|Kusonime).+?[_-]\d{1,3}(?:_720p)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            return match.group(1).lower()
    # Conservative fallback: related unknown clips share a broad prefix rather than
    # being randomly distributed between partitions.
    tokens = stem.split("_")
    return "unknown_" + "_".join(tokens[2:8]).lower()

rows = []
for directory_name, emotion in LABEL_DIRS.items():
    emotion_dir = ANIME_SAMPLE_ROOT / directory_name
    if not emotion_dir.exists():
        raise FileNotFoundError(f"Missing emotion directory: {emotion_dir}")
    for audio_path in emotion_dir.rglob("*.wav"):
        rows.append({
            "file_path": str(audio_path.resolve()),
            "filename": audio_path.name,
            "resource": RESOURCE_NAME,
            "emotion": emotion,
            "source_group": infer_source_group(audio_path),
        })

metadata = pd.DataFrame(rows).sort_values(["emotion", "filename"]).reset_index(drop=True)
label_encoder = LabelEncoder().fit(EMOTIONS)
metadata["label_id"] = label_encoder.transform(metadata["emotion"])

assert set(metadata["emotion"]) == set(EMOTIONS)
assert metadata["file_path"].is_unique, "Duplicate paths found"
print("Anime samples:", len(metadata))
print("Initial filename-derived source groups:", metadata["source_group"].nunique())
display(metadata.groupby("emotion").size().rename("samples").to_frame())
display(metadata[["filename", "emotion", "source_group"]].head(10))

manifest = pd.read_csv(ANIME_SAMPLE_MANIFEST) if ANIME_SAMPLE_MANIFEST.exists() else pd.DataFrame()
segment_manifest = pd.read_csv(SEGMENT_MANIFEST, low_memory=False) if SEGMENT_MANIFEST.exists() else pd.DataFrame()
print("Manifest rows:", len(manifest))
if not manifest.empty:
    # Attach proposal-required transcripts by copied filename. Older absolute
    # manifest paths are intentionally reduced to their filename component.
    manifest["filename"] = manifest["copied_file"].astype(str).map(lambda value: Path(value).name)
    accepted_manifest = manifest[manifest["selection_reason"].eq("selected")].copy()
    if accepted_manifest["filename"].duplicated().any():
        raise ValueError("Accepted cluster manifest contains duplicate filenames; provenance join is ambiguous.")
    accepted_columns = [c for c in [
        "filename", "clip_id", "target_emotion", "character_name", "speaker_cluster",
        "separability_status", "distance_to_own_centroid",
        "distance_to_nearest_other_centroid", "nearest_other_emotion",
        "separation_margin", "evidence_count", "top_score", "top_gap",
        "audio_quality_status", "transcript", "selection_reason"
    ] if c in accepted_manifest.columns]
    metadata = metadata.merge(accepted_manifest[accepted_columns], on="filename", how="left", validate="one_to_one")

    if not segment_manifest.empty:
        if segment_manifest["filename"].duplicated().any():
            raise ValueError("Segment manifest contains duplicate filenames; source metadata join is ambiguous.")
        source_columns = [c for c in [
            "filename", "source_media", "start_sec", "end_sec", "duration_sec",
            "lang", "language_prob", "japanese_char_ratio", "speaker", "character",
            "speaker_purity", "other_speaker_overlap_ratio", "speaker_assignment_pass"
        ] if c in segment_manifest.columns]
        metadata = metadata.merge(segment_manifest[source_columns], on="filename", how="left", validate="one_to_one")
        source_available = metadata["source_media"].fillna("").astype(str).str.strip().ne("")
        metadata.loc[source_available, "source_group"] = metadata.loc[source_available, "source_media"].astype(str)

    metadata["transcript"] = metadata.get("transcript", pd.Series("", index=metadata.index)).fillna("").astype(str)
    if metadata["selection_reason"].isna().any():
        raise ValueError("Some audio files are not traceable to selected rows in the cluster manifest.")
    label_mismatch = metadata["target_emotion"].astype(str).str.lower().str.replace(" ", "_").ne(metadata["emotion"])
    if label_mismatch.any():
        raise ValueError("Folder labels disagree with target_emotion in the accepted cluster manifest.")
    display(manifest[[c for c in [
        "clip_id", "target_emotion", "character_name", "speaker_cluster",
        "separability_status", "audio_quality_status", "selection_reason"
    ] if c in manifest.columns]].head())
else:
    metadata["transcript"] = ""

print("Samples with transcripts:", int(metadata["transcript"].str.strip().ne("").sum()))
print("Source-media groups used for splitting:", metadata["source_group"].nunique())
missing_transcripts = int(metadata["transcript"].str.strip().eq("").sum())
if missing_transcripts:
    raise ValueError(
        f"{missing_transcripts} accepted samples have no transcript. All branches must use "
        "the same sample population, so repair the manifest mapping before training."
    )

quality_control = pd.read_csv(QUALITY_CONTROL_SUMMARY) if QUALITY_CONTROL_SUMMARY.exists() else pd.DataFrame()
workflow_audit = pd.read_csv(WORKFLOW_AUDIT_SUMMARY) if WORKFLOW_AUDIT_SUMMARY.exists() else pd.DataFrame()
cluster_count_report = pd.read_csv(CLUSTER_COUNT_REPORT) if CLUSTER_COUNT_REPORT.exists() else pd.DataFrame()

proposal_source = pd.DataFrame([{
    "source": "Final Project Proposal PDF",
    "path": str(PROPOSAL_PDF),
    "exists": PROPOSAL_PDF.exists(),
    "size_bytes": PROPOSAL_PDF.stat().st_size if PROPOSAL_PDF.exists() else np.nan,
    "modified_time": pd.Timestamp(PROPOSAL_PDF.stat().st_mtime, unit="s").isoformat() if PROPOSAL_PDF.exists() else "",
    "sha256": hashlib.sha256(PROPOSAL_PDF.read_bytes()).hexdigest() if PROPOSAL_PDF.exists() else "",
}])
display(proposal_source)
proposal_source.to_csv(OUTPUT_DIR / "proposal_source_receipt.csv", index=False, encoding="utf-8-sig")

ready_for_final_training = (
    str(quality_control.iloc[0].get("ready_for_final_training", "unknown")).strip().lower()
    if not quality_control.empty else "unknown"
)
unknown_speaker_share = float(
    metadata.get("speaker_cluster", pd.Series("", index=metadata.index))
    .fillna("").astype(str).str.upper().eq("UNKNOWN_SPEAKER").mean()
)
manual_character_share = float((
    ~metadata.get("character_name", pd.Series("", index=metadata.index))
    .fillna("").astype(str).str.upper().isin(["", "GLOBAL_BASIC", "UNKNOWN"])
).mean())

proposal_alignment = pd.DataFrame([
    {"proposal_reference": "§3.2", "requirement": "Effective split near 68% train / 12% validation / 20% test", "implementation": "Group-disjoint source-media split using 0.68 / 0.12 / 0.20 targets", "status": "implemented"},
    {"proposal_reference": "§3.1.4", "requirement": "13 MFCC coefficients × 9 statistical descriptors", "implementation": "117 raw MFCC statistics", "status": "implemented"},
    {"proposal_reference": "§3.1.4", "requirement": "Correlation filtering before classical ML", "implementation": f"Training-only absolute-correlation filter at {MFCC_CORRELATION_THRESHOLD:.2f}", "status": "implemented below"},
    {"proposal_reference": "§3.7", "requirement": "128-band normalized log-Mel", "implementation": "128 bands, dB conversion, per-clip standardization", "status": "implemented"},
    {"proposal_reference": "§3.7–3.8", "requirement": "Six comparison branches and five MFCC classifiers", "implementation": "MFCC ML; CNN; CNN-BiLSTM; TF-IDF LR; 0.40/0.60 fusion; openSMILE-LLD XGBoost", "status": "implemented"},
    {"proposal_reference": "§3.4", "requirement": "Common resampling, mono conversion, and normalization", "implementation": "16 kHz mono peak-normalized signal shared by all acoustic branches", "status": "implemented below"},
    {"proposal_reference": "§3.5", "requirement": "Speaker/character organization for quality control", "implementation": f"Unknown speaker share={unknown_speaker_share:.1%}; identified character share={manual_character_share:.1%}", "status": "BLOCKER — upstream pipeline incomplete"},
    {"proposal_reference": "§3.3 and Appendix D", "requirement": "Pseudo-label QC and manual-review support", "implementation": "Cluster-selected pseudo-labels; not independently verified ground truth", "status": "experimental only"},
    {"proposal_reference": "§3.9", "requirement": "Accuracy, macro precision/recall/F1, confusion matrix, class-wise analysis", "implementation": "Saved for every completed method", "status": "implemented"},
])
display(proposal_alignment)
proposal_alignment.to_csv(OUTPUT_DIR / "proposal_alignment.csv", index=False, encoding="utf-8-sig")

pipeline_readiness = pd.DataFrame([
    {"check": "Auto Split ready_for_final_training", "observed": ready_for_final_training, "status": "pass" if ready_for_final_training == "yes" else "BLOCKER"},
    {"check": "Accepted audio mapped to cluster manifest", "observed": f"{metadata['selection_reason'].eq('selected').sum()}/{len(metadata)}", "status": "pass"},
    {"check": "Transcript coverage", "observed": f"{metadata['transcript'].str.strip().ne('').mean():.1%}", "status": "pass" if metadata['transcript'].str.strip().ne('').all() else "BLOCKER"},
    {"check": "Known speaker clusters", "observed": f"{1 - unknown_speaker_share:.1%}", "status": "BLOCKER" if unknown_speaker_share == 1.0 else "review"},
    {"check": "Identified characters", "observed": f"{manual_character_share:.1%}", "status": "BLOCKER" if manual_character_share == 0 else "review"},
    {"check": "Cluster-selection target reached for all emotions", "observed": "yes" if (not cluster_count_report.empty and cluster_count_report['shortage_count'].eq(0).all()) else "no", "status": "pass" if (not cluster_count_report.empty and cluster_count_report['shortage_count'].eq(0).all()) else "review"},
    {"check": "Planned 1–6 second duration", "observed": f"{metadata['duration_sec'].between(1, 6).mean():.1%} inside range", "status": "review"},
])
display(pipeline_readiness)
pipeline_readiness.to_csv(OUTPUT_DIR / "pipeline_readiness.csv", index=False, encoding="utf-8-sig")
if not cluster_count_report.empty:
    display(cluster_count_report.sort_values("shortage_count", ascending=False))
    cluster_count_report.sort_values("shortage_count", ascending=False).to_csv(
        OUTPUT_DIR / "cluster_selection_shortages.csv", index=False, encoding="utf-8-sig"
    )
if not workflow_audit.empty:
    workflow_warning_table = workflow_audit[workflow_audit["status"].astype(str).str.lower().ne("ok")].copy()
    display(workflow_warning_table)
    workflow_warning_table.to_csv(OUTPUT_DIR / "upstream_workflow_warnings.csv", index=False, encoding="utf-8-sig")

if ready_for_final_training != "yes":
    print("WARNING: The upstream pipeline marks this dataset as not ready for final training.")
    print("Scores may be used for pipeline development, but not described as final thesis evidence yet.")

def sha256_file(path: str, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()

# Full hashing can take time; keep True for the final thesis run.
RUN_FULL_DUPLICATE_AUDIT = True
if RUN_FULL_DUPLICATE_AUDIT:
    metadata["sha256"] = metadata["file_path"].map(sha256_file)
    duplicate_rows = metadata[metadata.duplicated("sha256", keep=False)].sort_values("sha256")
    cross_label_duplicates = duplicate_rows.groupby("sha256")["emotion"].nunique()
    cross_label_duplicates = cross_label_duplicates[cross_label_duplicates > 1]
    print("Duplicate rows:", len(duplicate_rows))
    print("Hashes occurring under multiple labels:", len(cross_label_duplicates))
    if len(cross_label_duplicates):
        raise ValueError("Cross-label duplicate audio found; resolve it before training.")
else:
    metadata["sha256"] = "not_computed"
    print("WARNING: Full duplicate audit disabled.")

def best_group_holdout(df, holdout_size, seed, attempts=300):
    # Choose a leakage-free group split with the closest available class distribution.
    target_distribution = df["emotion"].value_counts(normalize=True).reindex(EMOTIONS, fill_value=0)
    best = None
    for offset in range(attempts):
        splitter = GroupShuffleSplit(n_splits=1, test_size=holdout_size, random_state=seed + offset)
        development_idx, holdout_idx = next(splitter.split(df, groups=df["source_group"]))
        holdout = df.iloc[holdout_idx]
        holdout_counts = holdout["emotion"].value_counts().reindex(EMOTIONS, fill_value=0)
        if (holdout_counts == 0).any():
            continue
        holdout_distribution = holdout_counts / holdout_counts.sum()
        distribution_error = float((target_distribution - holdout_distribution).abs().sum())
        size_error = abs(len(holdout) / len(df) - holdout_size)
        minority_penalty = float((1.0 / holdout_counts.clip(lower=1)).sum()) * 0.01
        score = distribution_error + size_error + minority_penalty
        if best is None or score < best[0]:
            best = (score, development_idx, holdout_idx)
    if best is None:
        raise ValueError("Could not create a group-disjoint split containing all seven emotions.")
    return best[1], best[2]


def group_split(df: pd.DataFrame):
    train_val_idx, test_idx = best_group_holdout(df, TEST_SIZE, RANDOM_STATE)
    train_val = df.iloc[train_val_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    validation_fraction_of_remainder = VALIDATION_SIZE / (1.0 - TEST_SIZE)
    train_idx, validation_idx = best_group_holdout(
        train_val, validation_fraction_of_remainder, RANDOM_STATE + 10_000
    )
    return train_val.iloc[train_idx].reset_index(drop=True), train_val.iloc[validation_idx].reset_index(drop=True), test

train_meta, validation_meta, test_meta = group_split(metadata)

def balance_training_rows(frame):
    target_count = int(frame["emotion"].value_counts().max())
    balanced_parts = []
    for emotion in EMOTIONS:
        emotion_rows = frame[frame["emotion"] == emotion]
        balanced_parts.append(emotion_rows.sample(
            n=target_count, replace=len(emotion_rows) < target_count,
            random_state=RANDOM_STATE,
        ))
    return pd.concat(balanced_parts, ignore_index=True).sample(
        frac=1.0, random_state=RANDOM_STATE
    ).reset_index(drop=True)

train_balanced_meta = balance_training_rows(train_meta)
train_model_meta = train_balanced_meta if USE_TRAIN_OVERSAMPLING else train_meta.copy()
training_population_name = "oversampled training data" if USE_TRAIN_OVERSAMPLING else "raw training data"
print("Model fitting population:", training_population_name)

def assert_disjoint_groups(left, right, left_name, right_name):
    overlap = set(left["source_group"]) & set(right["source_group"])
    assert not overlap, f"Group leakage between {left_name} and {right_name}: {sorted(overlap)[:5]}"

assert_disjoint_groups(train_meta, validation_meta, "train", "validation")
assert_disjoint_groups(train_meta, test_meta, "train", "test")
assert_disjoint_groups(validation_meta, test_meta, "validation", "test")

split_summary = pd.concat([
    frame.groupby("emotion").size().rename(name)
    for name, frame in [("train", train_meta), ("validation", validation_meta), ("test", test_meta)]
], axis=1).fillna(0).astype(int)
display(split_summary)

training_balance_table = pd.concat({
    "raw_train": train_meta["emotion"].value_counts(),
    "balanced_train": train_balanced_meta["emotion"].value_counts(),
    "used_for_model_fitting": train_model_meta["emotion"].value_counts(),
}, axis=1).reindex(EMOTIONS).fillna(0).astype(int)
display(training_balance_table)

missing = {name: sorted(set(EMOTIONS) - set(frame["emotion"])) for name, frame in [
    ("train", train_meta), ("validation", validation_meta), ("test", test_meta)
]}
assert not any(missing.values()), f"One or more splits lack emotion classes: {missing}"

for name, frame in [("train", train_meta), ("validation", validation_meta), ("test", test_meta)]:
    frame.assign(split=name).to_csv(OUTPUT_DIR / f"{name}_manifest.csv", index=False)
train_balanced_meta.assign(split="balanced_train").to_csv(
    OUTPUT_DIR / "balanced_train_manifest.csv", index=False
)
train_model_meta.assign(split="model_train").to_csv(
    OUTPUT_DIR / "model_train_manifest.csv", index=False
)

# Table: complete dataset and split composition.
dataset_table = metadata.groupby("emotion").agg(
    samples=("file_path", "size"),
    source_groups=("source_group", "nunique"),
    transcripts=("transcript", lambda values: values.astype(str).str.strip().ne("").sum()),
).reindex(EMOTIONS)
dataset_table["share_percent"] = 100 * dataset_table["samples"] / dataset_table["samples"].sum()
display(dataset_table.style.format({"share_percent": "{:.2f}%"}))
dataset_table.to_csv(OUTPUT_DIR / "dataset_summary.csv")

split_plot = split_summary.reset_index().melt(id_vars="emotion", var_name="split", value_name="samples")
plt.figure(figsize=(11, 5))
sns.barplot(data=split_plot, x="emotion", y="samples", hue="split", palette="Blues")
plt.title("Anime sample distribution by emotion and data split")
plt.xlabel("Emotion")
plt.ylabel("Audio samples")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sample_distribution.png", dpi=180, bbox_inches="tight")
plt.show()

representative_rows = (
    metadata.sort_values("filename").groupby("emotion", as_index=False).first().set_index("emotion").reindex(EMOTIONS).reset_index()
)
for _, row in representative_rows.iterrows():
    audio, _ = librosa.load(row["file_path"], sr=TARGET_SR, mono=True)
    time_axis = np.arange(len(audio)) / TARGET_SR
    plt.figure(figsize=(11, 4))
    plt.plot(time_axis, audio, color="#245A9A", linewidth=0.7)
    plt.title(f"{row['emotion'].replace('_', ' ').title()} — waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"waveform_{row['emotion']}.png", dpi=180, bbox_inches="tight")
    plt.show()

    frequencies = np.fft.rfftfreq(len(audio), d=1 / TARGET_SR)
    magnitude = np.abs(np.fft.rfft(audio))
    plt.figure(figsize=(11, 4))
    plt.plot(frequencies, magnitude, color="#D28B27", linewidth=0.7)
    plt.xlim(0, TARGET_SR / 2)
    plt.title(f"{row['emotion'].replace('_', ' ').title()} — power spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"spectrum_{row['emotion']}.png", dpi=180, bbox_inches="tight")
    plt.show()

# A bounded stratified sample keeps this diagnostic fast while representing every class.
duration_sample = metadata.groupby("emotion", group_keys=False).sample(
    n=min(150, metadata.groupby("emotion").size().min()), random_state=RANDOM_STATE
).copy()
duration_sample["duration_seconds"] = duration_sample["file_path"].map(lambda path: librosa.get_duration(path=path))
display(duration_sample.groupby("emotion")["duration_seconds"].describe().round(3))

plt.figure(figsize=(11, 5))
sns.boxplot(data=duration_sample, x="emotion", y="duration_seconds", color="#B8D1EA")
plt.title("Audio-duration distribution by emotion (stratified sample)")
plt.xlabel("Emotion")
plt.ylabel("Duration (seconds)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "duration_by_emotion.png", dpi=180, bbox_inches="tight")
plt.show()

def metrics_row(method, y_true, y_pred):
    return {
        "method": method,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

def show_evaluation(method, y_true, y_pred):
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(EMOTIONS)))
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"{method} — Anime-only held-out test set")
    plt.xlabel("Predicted emotion")
    plt.ylabel("True emotion")
    plt.tight_layout()
    plt.show()

results = []
predictions = {}

def expanded_parameter_table(frame, parameter_column="parameters", sort_columns=None):
    expanded = pd.json_normalize(frame[parameter_column]).add_prefix("parameter_")
    tidy = pd.concat([frame.drop(columns=[parameter_column]).reset_index(drop=True), expanded], axis=1)
    if sort_columns:
        tidy = tidy.sort_values(
            [column for column, _ in sort_columns],
            ascending=[ascending for _, ascending in sort_columns],
        )
    return tidy.reset_index(drop=True)

def display_full_table(frame):
    display(HTML(frame.to_html(index=False, escape=False, float_format=lambda value: f"{value:.4f}")))

def load_common_audio(path):
    # Proposal §3.4 common preprocessing for every acoustic branch.
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    peak = np.max(np.abs(audio)) if len(audio) else 0.0
    if peak > 0:
        audio = audio / peak
    if USE_EXPERIMENTAL_AUDIO_ENHANCEMENT and len(audio) > 32:
        audio, _ = librosa.effects.trim(audio, top_db=25)
        nyquist = 0.5 * TARGET_SR
        low = 80.0 / nyquist
        high = min(4000.0 / nyquist, 0.999)
        b, a = signal.butter(4, [low, high], btype="band")
        audio = signal.filtfilt(b, a, audio)
        # Pre-emphasis is part of the optional enhancement, not common preprocessing.
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
    return audio.astype(np.float32)

def load_fixed_audio(path):
    return librosa.util.fix_length(load_common_audio(path), size=N_SAMPLES).astype(np.float32)

def mfcc_statistics(path):
    audio = load_common_audio(path)
    # Proposal Section 3.7: 13 MFCC coefficients summarized by nine statistics.
    mfcc = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=13)
    q1 = np.quantile(mfcc, 0.25, axis=1)
    median = np.quantile(mfcc, 0.50, axis=1)
    q3 = np.quantile(mfcc, 0.75, axis=1)
    return np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        mfcc.max(axis=1),
        mfcc.min(axis=1),
        mfcc.var(axis=1),
        q1,
        median,
        q3,
        q3 - q1,
    ]).astype(np.float32)

def feature_matrix(df, extractor):
    return np.stack([extractor(path) for path in df["file_path"]])

MFCC_STATISTICS = ["mean", "std", "max", "min", "variance", "q1", "median", "q3", "iqr"]
MFCC_FEATURE_NAMES = [f"{statistic}_C{coefficient}" for statistic in MFCC_STATISTICS for coefficient in range(13)]

X_train_mfcc_full = feature_matrix(train_model_meta, mfcc_statistics)
X_validation_mfcc_full = feature_matrix(validation_meta, mfcc_statistics)
X_test_mfcc_full = feature_matrix(test_meta, mfcc_statistics)
y_train = train_model_meta["label_id"].to_numpy()
y_validation = validation_meta["label_id"].to_numpy()
y_test = test_meta["label_id"].to_numpy()

# Proposal §3.1.4: fit correlation filtering on training data only. Validation
# and test data never influence which MFCC features are retained.
mfcc_correlation = np.corrcoef(X_train_mfcc_full, rowvar=False)
mfcc_correlation = np.nan_to_num(mfcc_correlation, nan=0.0, posinf=0.0, neginf=0.0)
mfcc_variance = X_train_mfcc_full.var(axis=0)
retained_mfcc_indices, mfcc_selection_rows = [], []
for feature_index, feature_name in enumerate(MFCC_FEATURE_NAMES):
    status, correlated_with, absolute_correlation = "retained", "", 0.0
    if mfcc_variance[feature_index] <= 1e-12:
        status = "removed_constant"
    elif retained_mfcc_indices:
        candidate_correlations = np.abs(mfcc_correlation[feature_index, retained_mfcc_indices])
        strongest_position = int(np.argmax(candidate_correlations))
        absolute_correlation = float(candidate_correlations[strongest_position])
        if absolute_correlation > MFCC_CORRELATION_THRESHOLD:
            status = "removed_correlated"
            correlated_with = MFCC_FEATURE_NAMES[retained_mfcc_indices[strongest_position]]
    if status == "retained":
        retained_mfcc_indices.append(feature_index)
    mfcc_selection_rows.append({
        "original_order": feature_index + 1,
        "feature": feature_name,
        "variance_in_training_data": float(mfcc_variance[feature_index]),
        "status": status,
        "correlated_with_retained_feature": correlated_with,
        "absolute_correlation": absolute_correlation,
        "threshold": MFCC_CORRELATION_THRESHOLD,
    })

mfcc_feature_selection = pd.DataFrame(mfcc_selection_rows)
display_full_table(mfcc_feature_selection)
mfcc_feature_selection.to_csv(OUTPUT_DIR / "mfcc_feature_selection.csv", index=False, encoding="utf-8-sig")
print(f"MFCC correlation filtering retained {len(retained_mfcc_indices)}/{len(MFCC_FEATURE_NAMES)} features.")

X_train_mfcc = X_train_mfcc_full[:, retained_mfcc_indices]
X_validation_mfcc = X_validation_mfcc_full[:, retained_mfcc_indices]
X_test_mfcc = X_test_mfcc_full[:, retained_mfcc_indices]

mfcc_search = {
    "MFCC + SVM": (
        lambda p: Pipeline([("scale", StandardScaler()), ("model", SVC(
            C=p["C"], gamma=p["gamma"], kernel="rbf", class_weight="balanced"
        ))]),
        {"C": [1, 10, 100], "gamma": ["scale", 0.01, 0.001]},
    ),
    "MFCC + k-NN": (
        lambda p: Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(
            n_neighbors=p["n_neighbors"], weights=p["weights"], p=p["p"]
        ))]),
        {"n_neighbors": [5, 7, 11], "weights": ["uniform", "distance"], "p": [1, 2]},
    ),
    "MFCC + Decision Tree": (
        lambda p: DecisionTreeClassifier(
            max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        {"max_depth": [None, 15, 30], "min_samples_leaf": [1, 2, 4]},
    ),
    "MFCC + Random Forest": (
        lambda p: RandomForestClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            max_features=p["max_features"], class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=SAFE_PARALLEL_JOBS,
        ),
        {"n_estimators": [200, 300], "max_depth": [None, 20, 30], "max_features": ["sqrt", "log2"]},
    ),
    "MFCC + Gaussian Naive Bayes": (
        lambda p: GaussianNB(var_smoothing=p["var_smoothing"]),
        {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]},
    ),
}

mfcc_tuning_rows, tuned_mfcc_models = [], {}
for name, (builder, parameter_grid) in mfcc_search.items():
    candidates = list(ParameterGrid(parameter_grid)) if RUN_HYPERPARAMETER_TUNING else [list(ParameterGrid(parameter_grid))[0]]
    best_candidate = None
    for parameters in candidates:
        candidate = builder(parameters)
        candidate.fit(X_train_mfcc, y_train)
        validation_pred = candidate.predict(X_validation_mfcc)
        row = metrics_row(name, y_validation, validation_pred)
        row["parameters"] = parameters
        mfcc_tuning_rows.append(row)
        if best_candidate is None or row["macro_f1"] > best_candidate[0]:
            best_candidate = (row["macro_f1"], parameters)

    selected_parameters = best_candidate[1]
    final_model = builder(selected_parameters)
    final_model.fit(X_train_mfcc, y_train)
    tuned_mfcc_models[name] = final_model
    test_pred = final_model.predict(X_test_mfcc)
    predictions[name] = test_pred
    results.append(metrics_row(name, y_test, test_pred))

mfcc_tuning_results = pd.DataFrame(mfcc_tuning_rows).sort_values(
    ["method", "macro_f1"], ascending=[True, False]
).reset_index(drop=True)
mfcc_tuning_tidy = expanded_parameter_table(
    mfcc_tuning_results,
    sort_columns=[("method", True), ("macro_f1", False)],
)
mfcc_best_table = mfcc_tuning_tidy.groupby("method", as_index=False, sort=False).first()
display_full_table(mfcc_best_table)
mfcc_tuning_tidy.to_csv(
    OUTPUT_DIR / "mfcc_hyperparameter_tuning.csv", index=False, encoding="utf-8-sig"
)

mfcc_mean_columns = [f"C{index}_mean" for index in range(13)]
mfcc_mean_df = pd.DataFrame(X_train_mfcc_full[:, :13], columns=mfcc_mean_columns)
mfcc_mean_df["emotion"] = train_model_meta["emotion"].to_numpy()

mfcc_configuration = pd.DataFrame([
    {"setting": "Sampling rate", "value": f"{TARGET_SR} Hz"},
    {"setting": "MFCC implementation", "value": "librosa.feature.mfcc"},
    {"setting": "Coefficient indices", "value": "C0-C12 (13 coefficients)"},
    {"setting": "Common audio preprocessing", "value": "16 kHz resampling; mono conversion; peak normalization"},
    {"setting": "Optional enhancement used", "value": str(USE_EXPERIMENTAL_AUDIO_ENHANCEMENT)},
    {"setting": "Optional enhancement contents", "value": "silence trim; 80-4000 Hz band-pass; pre-emphasis 0.97; renormalization"},
    {"setting": "Pre-emphasis applied in primary experiment", "value": str(USE_EXPERIMENTAL_AUDIO_ENHANCEMENT)},
    {"setting": "Per-coefficient summaries used by classical models", "value": "mean, standard deviation, maximum, minimum, variance, Q1, median, Q3, IQR"},
    {"setting": "Raw MFCC feature length", "value": "13 coefficients x 9 statistics = 117 features"},
    {"setting": "Correlation-filtered model input", "value": f"{len(retained_mfcc_indices)} features retained at threshold {MFCC_CORRELATION_THRESHOLD:.2f}"},
])
display_full_table(mfcc_configuration)
mfcc_configuration.to_csv(OUTPUT_DIR / "mfcc_configuration.csv", index=False, encoding="utf-8-sig")

# Coefficient order is an approximate interpretation, not a one-to-one physical label.
mfcc_levels = [
    ("Overall spectral level", ["C0_mean"], "C0: average log-spectrum level; energy-related but not a direct loudness measurement"),
    ("Low-order spectral envelope", [f"C{i}_mean" for i in range(1, 5)], "C1-C4: broad spectral-envelope and vocal-tract shape"),
    ("Mid-order spectral detail", [f"C{i}_mean" for i in range(5, 9)], "C5-C8: intermediate spectral-envelope variation"),
    ("High-order fine detail", [f"C{i}_mean" for i in range(9, 13)], "C9-C12: finer spectral variation; generally more noise-sensitive"),
]

mfcc_level_table = pd.DataFrame([
    {"analysis_level": level, "coefficient_range": description.split(":", 1)[0],
     "approximate_interpretation": description.split(":", 1)[1].strip()}
    for level, _, description in mfcc_levels
])
display_full_table(mfcc_level_table)
mfcc_level_table.to_csv(OUTPUT_DIR / "mfcc_coefficient_levels.csv", index=False, encoding="utf-8-sig")

for level, columns, description in mfcc_levels:
    plot_data = mfcc_mean_df.melt(
        id_vars="emotion", value_vars=columns, var_name="coefficient", value_name="mean_value"
    )
    plot_data["coefficient"] = plot_data["coefficient"].str.replace("_mean", "", regex=False)
    plt.figure(figsize=(11, 5.5))
    sns.boxplot(
        data=plot_data, x="emotion", y="mean_value", hue="coefficient",
        palette="Blues", showfliers=False
    )
    plt.title(f"{level} ({description})")
    plt.xlabel("Emotion")
    plt.ylabel("Mean MFCC value")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Coefficient", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    file_level = level.lower().replace("-", "_").replace(" ", "_")
    plt.savefig(OUTPUT_DIR / f"mfcc_level_{file_level}.png", dpi=180, bbox_inches="tight")
    plt.show()

plt.figure(figsize=(11, 9))
sns.heatmap(mfcc_mean_df[mfcc_mean_columns].corr(), cmap="vlag", center=0, vmin=-1, vmax=1)
plt.title("Correlation of MFCC C0-C12 mean values")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mfcc_correlation.png", dpi=180, bbox_inches="tight")
plt.show()

scaled_mfcc = StandardScaler().fit_transform(X_train_mfcc)
pca_coordinates = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(scaled_mfcc)
pca_df = pd.DataFrame({
    "PC1": pca_coordinates[:, 0], "PC2": pca_coordinates[:, 1],
    "emotion": train_model_meta["emotion"].to_numpy(),
})
plot_pca_df = pca_df.groupby("emotion", group_keys=False).sample(
    n=min(400, pca_df.groupby("emotion").size().min()), random_state=RANDOM_STATE
)
plt.figure(figsize=(10, 7))
sns.scatterplot(data=plot_pca_df, x="PC1", y="PC2", hue="emotion", alpha=0.55, s=22)
plt.title("Pre-training PCA view of MFCC statistical features")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.legend(title="Emotion", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mfcc_pca_clusters.png", dpi=180, bbox_inches="tight")
plt.show()

try:
    import opensmile
    from xgboost import XGBClassifier

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    def opensmile_features(path):
        # Use the same 16 kHz mono normalized waveform as MFCC and log-Mel.
        common_audio = load_common_audio(path)
        frame = smile.process_signal(common_audio, TARGET_SR).select_dtypes(include=[np.number])
        values = frame.to_numpy(dtype=np.float32)
        q1 = np.quantile(values, 0.25, axis=0)
        median = np.quantile(values, 0.50, axis=0)
        q3 = np.quantile(values, 0.75, axis=0)
        return np.concatenate([
            values.mean(axis=0), values.std(axis=0),
            values.min(axis=0), values.max(axis=0),
            q1, median, q3,
        ]).astype(np.float32)

    X_train_smile = feature_matrix(train_model_meta, opensmile_features)
    X_validation_smile = feature_matrix(validation_meta, opensmile_features)
    X_test_smile = feature_matrix(test_meta, opensmile_features)

    opensmile_grid = {
        "n_estimators": [200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.05],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    opensmile_candidates = list(ParameterGrid(opensmile_grid))
    if not RUN_HYPERPARAMETER_TUNING:
        opensmile_candidates = [opensmile_candidates[0]]
    opensmile_tuning_rows = []
    for parameters in opensmile_candidates:
        candidate = XGBClassifier(
            objective="multi:softprob", num_class=len(EMOTIONS),
            eval_metric="mlogloss", random_state=RANDOM_STATE,
            n_jobs=SAFE_PARALLEL_JOBS, **parameters,
        )
        candidate.fit(X_train_smile, y_train)
        validation_pred = candidate.predict(X_validation_smile).astype(int)
        row = metrics_row("openSMILE LLD + XGBoost", y_validation, validation_pred)
        row["parameters"] = parameters
        opensmile_tuning_rows.append(row)

    opensmile_tuning_df = pd.DataFrame(opensmile_tuning_rows).sort_values("macro_f1", ascending=False)
    best_opensmile_parameters = opensmile_tuning_df.iloc[0]["parameters"]
    opensmile_tuning_tidy = expanded_parameter_table(
        opensmile_tuning_df, sort_columns=[("macro_f1", False)]
    )
    display_full_table(opensmile_tuning_tidy.head(10))
    opensmile_tuning_tidy.to_csv(
        OUTPUT_DIR / "opensmile_xgboost_hyperparameter_tuning.csv",
        index=False, encoding="utf-8-sig",
    )

    opensmile_model = XGBClassifier(
        objective="multi:softprob", num_class=len(EMOTIONS),
        eval_metric="mlogloss", random_state=RANDOM_STATE,
        n_jobs=SAFE_PARALLEL_JOBS, **best_opensmile_parameters,
    )
    opensmile_model.fit(X_train_smile, y_train)
    opensmile_pred = opensmile_model.predict(X_test_smile).astype(int)
    predictions["openSMILE eGeMAPS + XGBoost"] = opensmile_pred
    results.append(metrics_row("openSMILE eGeMAPS + XGBoost", y_test, opensmile_pred))
    show_evaluation("openSMILE eGeMAPS + XGBoost", y_test, opensmile_pred)
except ImportError as exc:
    print("Skipped openSMILE/XGBoost:", exc)
    print("Install with: pip install opensmile xgboost")

def log_mel(path):
    audio = load_fixed_audio(path)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=TARGET_SR, n_fft=1024, hop_length=256, n_mels=N_MELS
    )
    db = librosa.power_to_db(mel, ref=np.max)
    return ((db - db.mean()) / (db.std() + 1e-8)).astype(np.float32)

class AnimeMelDataset(Dataset):
    def __init__(self, frame, feature_cache):
        self.frame = frame.reset_index(drop=True)
        self.feature_cache = feature_cache
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        x = self.feature_cache[str(row["file_path"])]
        return x, int(row["label_id"])

def weighted_sampler(frame):
    counts = frame["label_id"].value_counts().to_dict()
    weights = frame["label_id"].map(lambda label: 1.0 / counts[label]).to_numpy()
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)

def cached_log_mel(path):
    path = str(path)
    source_stat = Path(path).stat()
    # Cache identity includes the source path and feature settings. A setting change
    # creates a new file rather than silently reusing an incompatible feature.
    cache_identity = f"v4|{path}|bytes={source_stat.st_size}|mtime={source_stat.st_mtime_ns}|sr={TARGET_SR}|seconds={DURATION_SECONDS}|mels={N_MELS}|fft=1024|hop=256|enhance={USE_EXPERIMENTAL_AUDIO_ENHANCEMENT}"
    cache_name = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest() + ".npy"
    cache_path = CACHE_DIR / cache_name
    try:
        feature = np.load(cache_path, allow_pickle=False) if cache_path.exists() else None
    except (OSError, ValueError):
        # Recover automatically if an earlier interrupted run left a partial cache file.
        cache_path.unlink(missing_ok=True)
        feature = None
    if feature is None:
        feature = log_mel(path)
        np.save(cache_path, feature, allow_pickle=False)
    return torch.from_numpy(feature).unsqueeze(0)

unique_neural_paths = metadata["file_path"].astype(str).drop_duplicates().tolist()
print(f"Preparing log-Mel cache for {len(unique_neural_paths)} unique audio files ...")
mel_feature_cache = {}
for cache_index, audio_path in enumerate(unique_neural_paths, start=1):
    mel_feature_cache[audio_path] = cached_log_mel(audio_path)
    if cache_index == 1 or cache_index % 50 == 0 or cache_index == len(unique_neural_paths):
        print(f"  cached {cache_index}/{len(unique_neural_paths)}")
print("Log-Mel features are now reused for every epoch and tuning candidate.")

loader_options = {
    "batch_size": BATCH_SIZE,
    "num_workers": 0,  # Stable on Windows/Jupyter; features are already cached in memory.
    "pin_memory": torch.cuda.is_available(),
}
train_loader = DataLoader(AnimeMelDataset(train_model_meta, mel_feature_cache), shuffle=True, **loader_options)
validation_loader = DataLoader(AnimeMelDataset(validation_meta, mel_feature_cache), shuffle=False, **loader_options)
test_loader = DataLoader(AnimeMelDataset(test_meta, mel_feature_cache), shuffle=False, **loader_options)

class MelCNN(nn.Module):
    def __init__(self, n_classes=7, dropout=0.35, hidden_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class CNNBiLSTM(nn.Module):
    def __init__(self, n_classes=7, dropout=0.35, hidden_size=96):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 1)),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size * 2, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, n_classes),
        )
    def forward(self, x):
        x = self.cnn(x)                 # batch, channels, mel, time
        x = x.mean(dim=2).permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.classifier(x.mean(dim=1))

def class_weight_tensor(frame):
    counts = frame["label_id"].value_counts().reindex(range(len(EMOTIONS)), fill_value=1).to_numpy()
    weights = len(frame) / (len(EMOTIONS) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

@torch.no_grad()
def predict_loader(model, loader):
    model.eval()
    truth, pred, probabilities = [], [], []
    for x, y in loader:
        logits = model(x.to(DEVICE))
        truth.extend(y.numpy().tolist())
        pred.extend(logits.argmax(1).cpu().numpy().tolist())
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.asarray(truth), np.asarray(pred), np.vstack(probabilities)

def train_neural_model(model, method_name, learning_rate=1e-3, patience=5, evaluate_test=True, max_epochs=None):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_state, best_f1, stale = None, -1.0, 0
    history = []

    max_epochs = EPOCHS if max_epochs is None else int(max_epochs)
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)

        val_true, val_pred, _ = predict_loader(model, validation_loader)
        val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": running_loss / len(train_model_meta), "validation_macro_f1": val_f1})
        print(f"{method_name} epoch={epoch:02d} loss={history[-1]['train_loss']:.4f} val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    if not evaluate_test:
        return model, best_f1, history
    test_true, test_pred, test_probabilities = predict_loader(model, test_loader)
    return model, test_true, test_pred, test_probabilities, best_f1, history

trained_neural_models = {}
neural_test_probabilities = {}
neural_search_spaces = {
    "Log-mel CNN": [
        {"dropout": 0.30, "learning_rate": 1e-3},
        {"dropout": 0.40, "learning_rate": 5e-4},
        {"dropout": 0.50, "learning_rate": 3e-4},
    ],
    "Log-mel CNN-BiLSTM": [
        {"dropout": 0.30, "hidden_size": 64, "learning_rate": 1e-3},
        {"dropout": 0.40, "hidden_size": 96, "learning_rate": 5e-4},
        {"dropout": 0.40, "hidden_size": 128, "learning_rate": 3e-4},
    ],
}
neural_tuning_rows = []
for method_name, configurations in neural_search_spaces.items():
    active_configurations = configurations if RUN_HYPERPARAMETER_TUNING else configurations[:1]
    best_configuration = None
    for configuration in active_configurations:
        print(f"\nTuning {method_name}: {configuration} (maximum {TUNING_EPOCHS} epochs)")
        model_parameters = {key: value for key, value in configuration.items() if key != "learning_rate"}
        candidate_model = MelCNN(**model_parameters) if method_name == "Log-mel CNN" else CNNBiLSTM(**model_parameters)
        _, validation_macro_f1, _ = train_neural_model(
            candidate_model, method_name,
            learning_rate=configuration["learning_rate"], evaluate_test=False,
            max_epochs=TUNING_EPOCHS,
        )
        neural_tuning_rows.append({
            "method": method_name, "validation_macro_f1": validation_macro_f1,
            "parameters": configuration,
        })
        if best_configuration is None or validation_macro_f1 > best_configuration[0]:
            best_configuration = (validation_macro_f1, configuration)

    selected = best_configuration[1]
    selected_model_parameters = {key: value for key, value in selected.items() if key != "learning_rate"}
    final_model = MelCNN(**selected_model_parameters) if method_name == "Log-mel CNN" else CNNBiLSTM(**selected_model_parameters)
    print(f"\nFinal training {method_name} with selected configuration: {selected}")
    trained_model, neural_true, neural_pred, neural_probabilities, _, history = train_neural_model(
        final_model, method_name, learning_rate=selected["learning_rate"], evaluate_test=True,
    )
    trained_neural_models[method_name] = trained_model
    neural_test_probabilities[method_name] = neural_probabilities
    predictions[method_name] = neural_pred
    results.append(metrics_row(method_name, neural_true, neural_pred))
    show_evaluation(method_name, neural_true, neural_pred)

    safe_name = method_name.lower().replace(" ", "_")
    torch.save(trained_model.state_dict(), OUTPUT_DIR / f"{safe_name}.pt")
    pd.DataFrame(history).to_csv(OUTPUT_DIR / f"{safe_name}_history.csv", index=False)

neural_tuning_df = pd.DataFrame(neural_tuning_rows).sort_values(
    ["method", "validation_macro_f1"], ascending=[True, False]
)
neural_tuning_tidy = expanded_parameter_table(
    neural_tuning_df,
    sort_columns=[("method", True), ("validation_macro_f1", False)],
)
display_full_table(neural_tuning_tidy)
neural_tuning_tidy.to_csv(
    OUTPUT_DIR / "neural_hyperparameter_tuning.csv", index=False, encoding="utf-8-sig"
)

# Proposal Sections 3.7–3.8: unigram/bigram TF-IDF, balanced Logistic
# Regression, then score-level fusion using 0.40 CNN audio + 0.60 text.
train_text = train_model_meta[train_model_meta["transcript"].str.strip().ne("")].reset_index(drop=True)
validation_text = validation_meta[validation_meta["transcript"].str.strip().ne("")].reset_index(drop=True)
test_text_mask = test_meta["transcript"].str.strip().ne("").to_numpy()
test_text = test_meta.loc[test_text_mask].reset_index(drop=True)

transcript_coverage = len(test_text) / max(1, len(test_meta))
print(f"Held-out test transcript coverage: {transcript_coverage:.2%}")
if transcript_coverage < 1.0:
    raise ValueError(
        "Transcript coverage is below 100%. Repair the manifest mapping before "
        "reporting text-only or fusion results."
    )

text_grid = {"min_df": [1, 2, 3], "C": [0.5, 1.0, 2.0, 5.0]}
text_candidates = list(ParameterGrid(text_grid))
if not RUN_HYPERPARAMETER_TUNING:
    text_candidates = [text_candidates[0]]
text_tuning_rows = []
for parameters in text_candidates:
    candidate = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=parameters["min_df"],
            sublinear_tf=True,
        )),
        ("model", LogisticRegression(
            C=parameters["C"], max_iter=2000, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=SAFE_PARALLEL_JOBS,
        )),
    ])
    candidate.fit(train_text["transcript"], train_text["label_id"])
    validation_pred = candidate.predict(validation_text["transcript"])
    row = metrics_row("TF-IDF + Logistic Regression", validation_text["label_id"], validation_pred)
    row["parameters"] = parameters
    text_tuning_rows.append(row)

text_tuning_df = pd.DataFrame(text_tuning_rows).sort_values("macro_f1", ascending=False)
best_text_parameters = text_tuning_df.iloc[0]["parameters"]
text_tuning_tidy = expanded_parameter_table(text_tuning_df, sort_columns=[("macro_f1", False)])
display_full_table(text_tuning_tidy)
text_tuning_tidy.to_csv(
    OUTPUT_DIR / "text_hyperparameter_tuning.csv", index=False, encoding="utf-8-sig"
)

text_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=best_text_parameters["min_df"],
        sublinear_tf=True,
    )),
    ("model", LogisticRegression(
        C=best_text_parameters["C"], max_iter=2000, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=SAFE_PARALLEL_JOBS,
    )),
])
text_model.fit(train_text["transcript"], train_text["label_id"])
text_probabilities = text_model.predict_proba(test_text["transcript"])
text_pred = text_model.classes_[text_probabilities.argmax(axis=1)].astype(int)
text_true = test_text["label_id"].to_numpy(dtype=int)

predictions["TF-IDF + Logistic Regression"] = text_pred
results.append(metrics_row("TF-IDF + Logistic Regression", text_true, text_pred))
show_evaluation("TF-IDF + Logistic Regression", text_true, text_pred)

# Reorder text probabilities into the fixed seven-class label order if needed.
aligned_text_probabilities = np.zeros((len(test_text), len(EMOTIONS)), dtype=np.float32)
for source_column, class_id in enumerate(text_model.classes_):
    aligned_text_probabilities[:, int(class_id)] = text_probabilities[:, source_column]

audio_probabilities = neural_test_probabilities["Log-mel CNN"][test_text_mask]
fused_probabilities = 0.40 * audio_probabilities + 0.60 * aligned_text_probabilities
fusion_pred = fused_probabilities.argmax(axis=1)

predictions["Weighted late fusion (0.40 audio / 0.60 text)"] = fusion_pred
results.append(metrics_row("Weighted late fusion (0.40 audio / 0.60 text)", text_true, fusion_pred))
show_evaluation("Weighted late fusion (0.40 audio / 0.60 text)", text_true, fusion_pred)

results_df = pd.DataFrame(results).sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)
results_df.insert(0, "rank", np.arange(1, len(results_df) + 1))
display_full_table(results_df)

for metric, title, filename in [
    ("accuracy", "Test accuracy by anime SER method", "model_accuracy.png"),
    ("macro_f1", "Test macro-F1 by anime SER method", "model_macro_f1.png"),
]:
    metric_df = results_df.sort_values(metric, ascending=True)
    plt.figure(figsize=(11, 6))
    axes = sns.barplot(data=metric_df, y="method", x=metric, color="#4C82B8")
    axes.set_xlim(0, 1)
    axes.set_title(title)
    axes.set_xlabel(metric.replace("_", " ").title())
    axes.set_ylabel("Method")
    for container in axes.containers:
        axes.bar_label(container, fmt="%.3f", padding=3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.show()

results_df.to_csv(OUTPUT_DIR / "anime_only_final_results.csv", index=False, encoding="utf-8-sig")
with open(OUTPUT_DIR / "experiment_config.json", "w", encoding="utf-8") as handle:
    json.dump({
        "resource_name": RESOURCE_NAME,
        "anime_sample_root": str(ANIME_SAMPLE_ROOT),
        "random_state": RANDOM_STATE,
        "target_sample_rate": TARGET_SR,
        "duration_seconds": DURATION_SECONDS,
        "test_size": TEST_SIZE,
        "validation_size": VALIDATION_SIZE,
        "primary_metric": "macro_f1",
        "hyperparameter_tuning": RUN_HYPERPARAMETER_TUNING,
        "tuning_data": "training and validation only",
        "test_set_policy": "held out until final selected-model evaluation",
        "excluded_modes": ["tess_only", "combined", "transfer_clean_to_anime"],
    }, handle, ensure_ascii=False, indent=2)

old_results_path = (
    PROJECT_ROOT / "Python Notebook" / "Test" / "dual_training_outputs" / "experiment_results.csv"
)
method_name_map = {
    "openSMILE_LLD_XGBoost": "openSMILE eGeMAPS + XGBoost",
    "SVM": "MFCC + SVM",
    "k-NN": "MFCC + k-NN",
    "DT": "MFCC + Decision Tree",
    "RF": "MFCC + Random Forest",
    "NB": "MFCC + Gaussian Naive Bayes",
    "SimpleMelCNN": "Log-mel CNN",
    "CNNBiLSTM": "Log-mel CNN-BiLSTM",
    "TFIDF+LogRegText": "TF-IDF + Logistic Regression",
    "SimpleMelCNN+TFIDFLogRegFusion(0.40/0.60)": "Weighted late fusion (0.40 audio / 0.60 text)",
}
if old_results_path.exists():
    old_results = pd.read_csv(old_results_path)
    old_results = old_results[old_results["mode"] == "anime_only"].copy()
    old_results["method"] = old_results["model_name"].map(method_name_map)
    old_results = old_results.dropna(subset=["method"])
    old_results = old_results.sort_values("macro_f1", ascending=False).drop_duplicates("method")
    comparison_df = results_df.merge(
        old_results[["method", "accuracy", "macro_f1"]], on="method", how="left",
        suffixes=("_grouped_current", "_old_random_curated"),
    )
    comparison_df["accuracy_difference"] = (
        comparison_df["accuracy_grouped_current"] - comparison_df["accuracy_old_random_curated"]
    )
    comparison_df["macro_f1_difference"] = (
        comparison_df["macro_f1_grouped_current"] - comparison_df["macro_f1_old_random_curated"]
    )
    comparison_df = comparison_df.sort_values("macro_f1_grouped_current", ascending=False).reset_index(drop=True)
    comparison_df["rank"] = np.arange(1, len(comparison_df) + 1)
    display_full_table(comparison_df)
    comparison_df.to_csv(
        OUTPUT_DIR / "diagnostic_old_vs_grouped_results.csv", index=False, encoding="utf-8-sig"
    )

methodology_difference_df = pd.DataFrame([
    {
        "factor": "Accepted dataset",
        "earlier_notebook": "Different curated pool; saved metadata shows about 1,793 anime clips before balancing/filtering",
        "current_notebook": f"{len(metadata):,} selected clips physically present in Anime_SER_More_Sample/sample",
        "expected_effect": "Scores are not directly comparable because the evaluated populations differ",
    },
    {
        "factor": "Test separation",
        "earlier_notebook": "Random stratified clip split",
        "current_notebook": "Upstream source_media/episode group-disjoint split",
        "expected_effect": "Reduces episode/source leakage; character leakage cannot be ruled out until speaker metadata is completed",
    },
    {
        "factor": "Class distribution",
        "earlier_notebook": "Balanced/filtered saved run",
        "current_notebook": f"Highly imbalanced accepted pool; model oversampling enabled={USE_TRAIN_OVERSAMPLING}; natural validation/test",
        "expected_effect": "Natural test accuracy can fall; macro-F1 remains the primary comparison metric",
    },
    {
        "factor": "CNN implementation",
        "earlier_notebook": "128 Mel bands and 4x4 adaptive feature map",
        "current_notebook": "Now restored to the same 128-Mel and 4x4 architecture",
        "expected_effect": "Fixes the earlier packaged CNN information bottleneck",
    },
    {
        "factor": "Audio preprocessing",
        "earlier_notebook": "Silence trim, peak normalization, 80-4000 Hz band-pass, pre-emphasis",
        "current_notebook": f"Proposal-aligned 16 kHz mono peak normalization; optional enhancement enabled={USE_EXPERIMENTAL_AUDIO_ENHANCEMENT}",
        "expected_effect": "All acoustic branches now receive the same common waveform preprocessing",
    },
    {
        "factor": "Transcript source",
        "earlier_notebook": "Included filename-derived text when reports were unavailable",
        "current_notebook": "Uses manifest transcripts and requires 100% coverage",
        "expected_effect": "Avoids source-name leakage but may reduce artificially predictive text cues",
    },
])
display_full_table(methodology_difference_df)
methodology_difference_df.to_csv(
    OUTPUT_DIR / "methodology_difference_table.csv", index=False, encoding="utf-8-sig"
)

selected_parameter_rows = []
for method_name in mfcc_tuning_results["method"].unique():
    best_row = mfcc_tuning_results[mfcc_tuning_results["method"] == method_name].iloc[0]
    selected_parameter_rows.append({
        "method": method_name,
        "selected_parameters": best_row["parameters"],
        "validation_macro_f1": best_row["macro_f1"],
    })
if "opensmile_tuning_df" in globals():
    selected_parameter_rows.append({
        "method": "openSMILE LLD + XGBoost",
        "selected_parameters": best_opensmile_parameters,
        "validation_macro_f1": opensmile_tuning_df.iloc[0]["macro_f1"],
    })
for _, row in neural_tuning_df.groupby("method", as_index=False).first().iterrows():
    selected_parameter_rows.append({
        "method": row["method"],
        "selected_parameters": row["parameters"],
        "validation_macro_f1": row["validation_macro_f1"],
    })
selected_parameter_rows.append({
    "method": "TF-IDF + Logistic Regression",
    "selected_parameters": best_text_parameters,
    "validation_macro_f1": text_tuning_df.iloc[0]["macro_f1"],
})
selected_parameter_rows.append({
    "method": "Weighted late fusion",
    "selected_parameters": {"audio_weight": 0.40, "text_weight": 0.60},
    "validation_macro_f1": np.nan,
})

selected_parameters_df = pd.DataFrame(selected_parameter_rows)
parameter_columns_df = pd.json_normalize(selected_parameters_df.pop("selected_parameters")).add_prefix("parameter_")
selected_parameters_df = pd.concat([selected_parameters_df, parameter_columns_df], axis=1)
selected_parameters_df = selected_parameters_df.sort_values(
    ["validation_macro_f1", "method"], ascending=[False, True], na_position="last"
).reset_index(drop=True)
selected_parameters_df.insert(0, "rank", np.arange(1, len(selected_parameters_df) + 1))
display(HTML(selected_parameters_df.to_html(index=False, escape=False, float_format=lambda value: f"{value:.4f}")))
selected_parameters_df.to_csv(OUTPUT_DIR / "selected_hyperparameters.csv", index=False, encoding="utf-8-sig")

history_frames = []
for method_name in ["Log-mel CNN", "Log-mel CNN-BiLSTM"]:
    safe_name = method_name.lower().replace(" ", "_")
    history_path = OUTPUT_DIR / f"{safe_name}_history.csv"
    history_frame = pd.read_csv(history_path)
    history_frame["method"] = method_name
    history_frames.append(history_frame)
training_history_df = pd.concat(history_frames, ignore_index=True)

for method_name in training_history_df["method"].unique():
    method_history = training_history_df[training_history_df["method"] == method_name]
    safe_name = method_name.lower().replace(" ", "_")
    for metric, title, y_label in [
        ("train_loss", "Training loss", "Cross-entropy loss"),
        ("validation_macro_f1", "Validation macro-F1", "Macro-F1"),
    ]:
        plt.figure(figsize=(9, 5))
        sns.lineplot(data=method_history, x="epoch", y=metric, marker="o", color="#4C82B8")
        plt.title(f"{method_name} — {title}")
        plt.xlabel("Epoch")
        plt.ylabel(y_label)
        if metric == "validation_macro_f1":
            plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{safe_name}_{metric}.png", dpi=180, bbox_inches="tight")
        plt.show()

classwise_rows = []
for method_name, predicted_labels in predictions.items():
    true_labels = y_test if len(predicted_labels) == len(y_test) else text_true
    precision_values, recall_values, f1_values, support_values = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=np.arange(len(EMOTIONS)), zero_division=0
    )
    for class_index, emotion in enumerate(label_encoder.classes_):
        classwise_rows.append({
            "method": method_name,
            "emotion": emotion,
            "precision": precision_values[class_index],
            "recall": recall_values[class_index],
            "f1": f1_values[class_index],
            "support": int(support_values[class_index]),
        })

classwise_results_df = pd.DataFrame(classwise_rows)
method_order = {method: rank for rank, method in enumerate(results_df["method"], start=1)}
emotion_order = {emotion: rank for rank, emotion in enumerate(label_encoder.classes_, start=1)}
classwise_results_df["method_rank"] = classwise_results_df["method"].map(method_order)
classwise_results_df["emotion_rank"] = classwise_results_df["emotion"].map(emotion_order)
classwise_results_df = classwise_results_df.sort_values(
    ["method_rank", "emotion_rank"]
).drop(columns=["method_rank", "emotion_rank"]).reset_index(drop=True)
classwise_results_df.insert(0, "row", np.arange(1, len(classwise_results_df) + 1))
display_full_table(classwise_results_df)
classwise_results_df.to_csv(OUTPUT_DIR / "classwise_results.csv", index=False, encoding="utf-8-sig")

f1_heatmap = classwise_results_df.pivot(index="method", columns="emotion", values="f1").reindex(
    results_df["method"]
)
plt.figure(figsize=(12, max(5, 0.55 * len(f1_heatmap))))
sns.heatmap(f1_heatmap, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1)
plt.title("Per-class F1 by anime SER method")
plt.xlabel("Emotion")
plt.ylabel("Method")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "classwise_f1_heatmap.png", dpi=180, bbox_inches="tight")
plt.show()

method_names = results_df["method"].tolist()
for method_name in method_names:
    predicted_labels = predictions[method_name]
    true_labels = y_test if len(predicted_labels) == len(y_test) else text_true
    matrix = confusion_matrix(true_labels, predicted_labels, labels=np.arange(len(EMOTIONS)))
    support_by_class = matrix.sum(axis=1)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
    )
    plt.title(f"{method_name} — held-out test confusion matrix")
    plt.xlabel("Predicted emotion")
    plt.ylabel("Actual emotion")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    safe_name = re.sub(r"[^a-z0-9]+", "_", method_name.lower()).strip("_")
    plt.savefig(OUTPUT_DIR / f"confusion_{safe_name}.png", dpi=180, bbox_inches="tight")
    plt.show()

confusion_pair_rows = []
for method_name, predicted_labels in predictions.items():
    true_labels = y_test if len(predicted_labels) == len(y_test) else text_true
    matrix = confusion_matrix(true_labels, predicted_labels, labels=np.arange(len(EMOTIONS)))
    support_by_class = matrix.sum(axis=1)
    for actual_index, actual_emotion in enumerate(label_encoder.classes_):
        for predicted_index, predicted_emotion in enumerate(label_encoder.classes_):
            if actual_index != predicted_index and matrix[actual_index, predicted_index] > 0:
                confusion_pair_rows.append({
                    "method": method_name,
                    "actual_emotion": actual_emotion.replace("_", " ").title(),
                    "predicted_emotion": predicted_emotion.replace("_", " ").title(),
                    "error_count": int(matrix[actual_index, predicted_index]),
                    "class_support": int(support_by_class[actual_index]),
                    "error_rate_percent": 100 * matrix[actual_index, predicted_index] / max(1, support_by_class[actual_index]),
                })
confusion_pairs_df = pd.DataFrame(confusion_pair_rows).sort_values(
    ["method", "error_rate_percent", "error_count"], ascending=[True, False, False]
)
top_confusion_pairs_df = confusion_pairs_df.groupby("method", as_index=False).head(5)
top_confusion_pairs_df["method_rank"] = top_confusion_pairs_df["method"].map(method_order)
top_confusion_pairs_df = top_confusion_pairs_df.sort_values(
    ["method_rank", "error_rate_percent", "error_count"], ascending=[True, False, False]
).drop(columns="method_rank").reset_index(drop=True)
top_confusion_pairs_df.insert(
    1, "error_rank", top_confusion_pairs_df.groupby("method").cumcount() + 1
)
display_full_table(top_confusion_pairs_df)
top_confusion_pairs_df.to_csv(
    OUTPUT_DIR / "top_confusion_pairs.csv", index=False, encoding="utf-8-sig"
)

weak_class_table = classwise_results_df.sort_values(["method", "f1"]).groupby("method", as_index=False).first()
weak_class_table = weak_class_table[["method", "emotion", "precision", "recall", "f1", "support"]]
weak_class_table["method_rank"] = weak_class_table["method"].map(method_order)
weak_class_table = weak_class_table.sort_values("method_rank").drop(columns="method_rank").reset_index(drop=True)
display_full_table(weak_class_table)
weak_class_table.to_csv(
    OUTPUT_DIR / "weakest_class_by_method.csv", index=False, encoding="utf-8-sig"
)

RUN_STATUS_PATH.write_text(json.dumps({
    "execution_status": "completed",
    "completed_methods": results_df["method"].tolist(),
    "run_hyperparameter_tuning": RUN_HYPERPARAMETER_TUNING,
    "use_train_oversampling": USE_TRAIN_OVERSAMPLING,
    "upstream_ready_for_final_training": ready_for_final_training,
    "final_thesis_ready": ready_for_final_training == "yes" and unknown_speaker_share < 1.0,
    "message": "Execution completed. Check proposal_alignment.csv and pipeline_readiness.csv before reporting scores.",
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("Run status saved:", RUN_STATUS_PATH)
