"""
Baseline script for the project milestone.

This file loads the Jigsaw training data, data validation, data cleaning, runs
TF-IDF + One-vs-Rest Logistic Regression baseline, and saves the evaluation
results.
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    classification_report
)


# File locations
DATA_PATH = "data/train.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load the dataset
print("Loading training data...")
df = pd.read_csv(DATA_PATH)

print(f"Data shape: {df.shape}")
print(df.columns.tolist())


# Main columns used in this project
id_col = "id"
text_col = "comment_text"

label_cols = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

required_cols = [id_col, text_col] + label_cols


# Make sure the expected columns are actually in the file
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

print("All required columns found.")


# Basic missing-value check
missing_values = df[required_cols].isnull().sum()
print("\nMissing values:")
print(missing_values)

missing_values.to_csv(os.path.join(OUTPUT_DIR, "missing_values.csv"))


if missing_values.sum() > 0:
    df = df.dropna(subset=required_cols)


# Check that all label columns are binary 0/1
print("\nChecking label values...")
for col in label_cols:
    values = sorted(df[col].dropna().unique().tolist())
    print(f"{col}: {values}")

    if not set(values).issubset({0, 1}):
        raise ValueError(f"{col} has non-binary values: {values}")

df[label_cols] = df[label_cols].astype(int)


# Keep the raw text because we may need it later for error analysis
df["comment_text_original"] = df[text_col].astype(str)

# Minimal text cleaning:
# only normalize whitespace. We are not removing profanity, punctuation,
# capitalization, or slang because those can be important toxicity signals.
df["comment_text_clean"] = (
    df[text_col]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


# Check for possible Excel/spreadsheet corruption like #NAME?

corrupted_values = [
    "#NAME?",
    "#VALUE!",
    "#REF!",
    "#DIV/0!",
    "#N/A",
    "#NULL!",
    "#NUM!"
]

corrupted_mask = df["comment_text_clean"].isin(corrupted_values)
num_corrupted = int(corrupted_mask.sum())

print(f"\nPossible corrupted text rows: {num_corrupted}")

if num_corrupted > 0:
    df.loc[
        corrupted_mask,
        [id_col, "comment_text_original"] + label_cols
    ].to_csv(
        os.path.join(OUTPUT_DIR, "corrupted_text_rows.csv"),
        index=False
    )


# Duplicate checks
duplicate_id_count = int(df[id_col].duplicated().sum())
duplicate_text_count = int(df["comment_text_clean"].duplicated().sum())

print("\nDuplicate check:")
print(f"Duplicate IDs: {duplicate_id_count}")
print(f"Duplicate comments: {duplicate_text_count}")

pd.DataFrame({
    "check": ["duplicate_id", "duplicate_comment_text_clean"],
    "count": [duplicate_id_count, duplicate_text_count]
}).to_csv(
    os.path.join(OUTPUT_DIR, "duplicate_summary.csv"),
    index=False
)


# Simple text length stats, mostly for understanding the data
df["text_length"] = df["comment_text_clean"].str.len()
df["word_count"] = df["comment_text_clean"].str.split().apply(len)

text_stats = df[["text_length", "word_count"]].describe()
print("\nText length statistics:")
print(text_stats)

text_stats.to_csv(os.path.join(OUTPUT_DIR, "text_length_stats.csv"))


# Label distribution: this shows how imbalanced the dataset is
label_counts = df[label_cols].sum()
label_rates = df[label_cols].mean()

label_distribution = pd.DataFrame({
    "positive_count": label_counts,
    "positive_rate": label_rates
}).sort_values(by="positive_count", ascending=False)

print("\nLabel distribution:")
print(label_distribution)

label_distribution.to_csv(
    os.path.join(OUTPUT_DIR, "label_distribution.csv")
)


# Count clean comments, toxic comments, and multi-label comments
df["num_positive_labels"] = df[label_cols].sum(axis=1)

clean_count = int((df["num_positive_labels"] == 0).sum())
toxic_count = int((df["num_positive_labels"] >= 1).sum())
single_label_count = int((df["num_positive_labels"] == 1).sum())
multi_label_count = int((df["num_positive_labels"] >= 2).sum())

label_count_summary = pd.DataFrame({
    "category": [
        "clean_comments_all_labels_0",
        "toxic_comments_at_least_one_label",
        "single_label_toxic_comments",
        "multi_label_toxic_comments"
    ],
    "count": [
        clean_count,
        toxic_count,
        single_label_count,
        multi_label_count
    ],
    "percentage": [
        clean_count / len(df),
        toxic_count / len(df),
        single_label_count / len(df),
        multi_label_count / len(df)
    ]
})

print("\nClean / toxic / multi-label summary:")
print(label_count_summary)

label_count_summary.to_csv(
    os.path.join(OUTPUT_DIR, "label_count_summary.csv"),
    index=False
)


# Co-occurrence matrix for the six labels.
# This helps us see which toxicity labels often appear together.
cooccurrence_matrix = df[label_cols].T.dot(df[label_cols])

print("\nLabel co-occurrence matrix:")
print(cooccurrence_matrix)

cooccurrence_matrix.to_csv(
    os.path.join(OUTPUT_DIR, "label_cooccurrence_matrix.csv")
)


# A simpler pairwise co-occurrence table for the milestone report
pairs = []

for i, label_a in enumerate(label_cols):
    for label_b in label_cols[i + 1:]:
        count = int(((df[label_a] == 1) & (df[label_b] == 1)).sum())
        pairs.append({
            "label_pair": f"{label_a} + {label_b}",
            "cooccurrence_count": count
        })

pairwise_cooccurrence = pd.DataFrame(pairs).sort_values(
    by="cooccurrence_count",
    ascending=False
)

print("\nTop label co-occurrences:")
print(pairwise_cooccurrence)

pairwise_cooccurrence.to_csv(
    os.path.join(OUTPUT_DIR, "pairwise_label_cooccurrence.csv"),
    index=False
)


# Train/validation split.
# This is a simple split for the milestone baseline.
X = df["comment_text_clean"]
y = df[label_cols]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain/validation split:")
print(f"Train size: {X_train.shape[0]}")
print(f"Validation size: {X_val.shape[0]}")


# Convert text into TF-IDF features.
# We use unigrams + bigrams because short toxic phrases can be useful.
tfidf = TfidfVectorizer(
    lowercase=True,
    max_features=50000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95
)

print("\nVectorizing text with TF-IDF...")

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

print(f"TF-IDF train shape: {X_train_tfidf.shape}")
print(f"TF-IDF validation shape: {X_val_tfidf.shape}")


# Baseline model:
# One-vs-Rest Logistic Regression treats each label as its own binary task.
# We are not using class weights yet because imbalance mitigation is part of
# the next stage of the project
base_lr = LogisticRegression(
    max_iter=1000,
    solver="liblinear"
)

model = OneVsRestClassifier(base_lr)

print("\nTraining baseline model...")
model.fit(X_train_tfidf, y_train)

print("Baseline training finished.")


# Get predictions on the validation set.
# y_pred gives 0/1 labels. y_prob is saved for later threshold tuning.
print("\nPredicting on validation set...")

y_pred = model.predict(X_val_tfidf)
y_prob = model.predict_proba(X_val_tfidf)


# Overall multi-label metrics
micro_f1 = f1_score(y_val, y_pred, average="micro", zero_division=0)
macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
hamming = hamming_loss(y_val, y_pred)

overall_metrics = pd.DataFrame({
    "metric": [
        "micro_f1",
        "macro_f1",
        "weighted_f1",
        "hamming_loss"
    ],
    "value": [
        micro_f1,
        macro_f1,
        weighted_f1,
        hamming
    ]
})

print("\nOverall baseline metrics:")
print(overall_metrics)

overall_metrics.to_csv(
    os.path.join(OUTPUT_DIR, "baseline_overall_metrics.csv"),
    index=False
)


# Per-label metrics and error counts.

per_label_rows = []

for idx, label in enumerate(label_cols):
    y_true_label = y_val.iloc[:, idx]
    y_pred_label = y_pred[:, idx]

    precision = precision_score(y_true_label, y_pred_label, zero_division=0)
    recall = recall_score(y_true_label, y_pred_label, zero_division=0)
    f1 = f1_score(y_true_label, y_pred_label, zero_division=0)

    tp = int(((y_true_label == 1) & (y_pred_label == 1)).sum())
    fp = int(((y_true_label == 0) & (y_pred_label == 1)).sum())
    fn = int(((y_true_label == 1) & (y_pred_label == 0)).sum())
    tn = int(((y_true_label == 0) & (y_pred_label == 0)).sum())

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    per_label_rows.append({
        "label": label,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "positive_count_in_validation": int(y_true_label.sum())
    })

per_label_metrics = pd.DataFrame(per_label_rows)

print("\nPer-label baseline metrics:")
print(per_label_metrics)

per_label_metrics.to_csv(
    os.path.join(OUTPUT_DIR, "baseline_per_label_metrics.csv"),
    index=False
)


# Save the usual sklearn classification report
report = classification_report(
    y_val,
    y_pred,
    target_names=label_cols,
    zero_division=0
)

print("\nClassification report:")
print(report)

with open(os.path.join(OUTPUT_DIR, "baseline_classification_report.txt"), "w") as f:
    f.write(report)


# Save validation-level predictions.
# This file will be useful later when we inspect false positives/negatives.
val_results = pd.DataFrame({
    "comment_text": X_val.values
})

for idx, label in enumerate(label_cols):
    val_results[f"true_{label}"] = y_val.iloc[:, idx].values
    val_results[f"pred_{label}"] = y_pred[:, idx]
    val_results[f"prob_{label}"] = y_prob[:, idx]

true_cols = [f"true_{label}" for label in label_cols]
pred_cols = [f"pred_{label}" for label in label_cols]

val_results["num_true_labels"] = val_results[true_cols].sum(axis=1)
val_results["num_pred_labels"] = val_results[pred_cols].sum(axis=1)

val_results["is_true_multilabel"] = val_results["num_true_labels"] >= 2
val_results["is_pred_multilabel"] = val_results["num_pred_labels"] >= 2

val_results.to_csv(
    os.path.join(OUTPUT_DIR, "baseline_validation_predictions.csv"),
    index=False
)


# Save false positives and false negatives separately by label.
# These files are mainly for the later structured error analysis.
for label in label_cols:
    fp_rows = val_results[
        (val_results[f"true_{label}"] == 0) &
        (val_results[f"pred_{label}"] == 1)
    ]

    fn_rows = val_results[
        (val_results[f"true_{label}"] == 1) &
        (val_results[f"pred_{label}"] == 0)
    ]

    fp_rows.to_csv(
        os.path.join(OUTPUT_DIR, f"false_positives_{label}.csv"),
        index=False
    )

    fn_rows.to_csv(
        os.path.join(OUTPUT_DIR, f"false_negatives_{label}.csv"),
        index=False
    )


# A short summary file 
summary_path = os.path.join(OUTPUT_DIR, "milestone_summary.txt")

with open(summary_path, "w") as f:
    f.write("Project Milestone Baseline Summary\n")
    f.write("==================================\n\n")

    f.write(f"Dataset shape: {df.shape}\n\n")

    f.write("Baseline model:\n")
    f.write("TF-IDF + One-vs-Rest Logistic Regression\n")
    f.write("Default threshold: 0.5\n\n")

    f.write("Overall metrics:\n")
    f.write(overall_metrics.to_string(index=False))
    f.write("\n\n")

    f.write("Per-label metrics:\n")
    f.write(per_label_metrics.to_string(index=False))
    f.write("\n\n")

    f.write("Next steps:\n")
    f.write("- Class weighting for imbalance mitigation\n")
    f.write("- Per-label threshold tuning for rare labels\n")
    f.write("- Structured error analysis on false positives and false negatives\n")
    f.write("- Single-label vs multi-label error comparison\n")

print("\nDone.")
print(f"All outputs saved in: {OUTPUT_DIR}/")