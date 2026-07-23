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
     random_state=42
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





# ============================================================
# FINAL PROJECT EXTENSION
# ============================================================

"""
For final report use.

The following sections extend the milestone baseline with additional
evaluation metrics, structured error analysis, imbalance mitigation,
and a BERT-based model comparison.
"""

# ============================================================
# Part 1. EXTENDED EVALUATION - ROC-AUC and exact match / subset accuracy
# ============================================================

from sklearn.metrics import roc_auc_score, accuracy_score

print("\n" + "=" * 60)
print("PART 1: EXTENDED EVALUATION")
print("=" * 60)

# Convert validation labels to NumPy array for easier comparison
y_val_array = y_val.to_numpy()

# In multi-label classification, exact match accuracy and subset accuracy
# mean the same thing: all six labels must be predicted correctly.
subset_accuracy = accuracy_score(y_val_array, y_pred)

# Overall probability-based ROC-AUC metrics
micro_roc_auc = roc_auc_score(
    y_val_array,
    y_prob,
    average="micro"
)

macro_roc_auc = roc_auc_score(
    y_val_array,
    y_prob,
    average="macro"
)

extended_overall_metrics = pd.DataFrame({
    "metric": [
        "micro_f1",
        "macro_f1",
        "weighted_f1",
        "hamming_loss",
        "subset_accuracy_exact_match",
        "micro_roc_auc",
        "macro_roc_auc"
    ],
    "value": [
        micro_f1,
        macro_f1,
        weighted_f1,
        hamming,
        subset_accuracy,
        micro_roc_auc,
        macro_roc_auc
    ]
})

print("\nExtended overall baseline metrics:")
print(extended_overall_metrics)

extended_overall_metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "baseline_extended_overall_metrics.csv"
    ),
    index=False
)


# Per-label ROC-AUC
per_label_roc_auc = []

for idx, label in enumerate(label_cols):
    y_true_label = y_val_array[:, idx]
    y_prob_label = y_prob[:, idx]

    # ROC-AUC requires both positive and negative examples
    if len(np.unique(y_true_label)) < 2:
        roc_auc = np.nan
    else:
        roc_auc = roc_auc_score(
            y_true_label,
            y_prob_label
        )

    per_label_roc_auc.append({
        "label": label,
        "roc_auc": roc_auc
    })

per_label_roc_auc_df = pd.DataFrame(per_label_roc_auc)

print("\nPer-label ROC-AUC:")
print(per_label_roc_auc_df)

per_label_roc_auc_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "baseline_per_label_roc_auc.csv"
    ),
    index=False
)


# Merge ROC-AUC with the existing precision, recall, and F1 table
per_label_metrics_extended = per_label_metrics.merge(
    per_label_roc_auc_df,
    on="label",
    how="left"
)

per_label_metrics_extended.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "baseline_per_label_metrics_extended.csv"
    ),
    index=False
)

print("\nPart 1 completed.")
print("Extended evaluation outputs saved in outputs/.")


# ============================================================
# PART 2: STRUCTURED ERROR ANALYSIS
# Single-label vs multi-label errors, partial misses, label co-occurrence, and representative FP/FN examples
# ============================================================
"""
This section analyzes where the baseline fails, especially on
multi-label comments and rare, specific toxicity categories.
"""

print("\n" + "=" * 60)
print("PART 2: STRUCTURED ERROR ANALYSIS")
print("=" * 60)

y_true = y_val.to_numpy()
y_pred_np = np.asarray(y_pred)
y_prob_np = np.asarray(y_prob)
val_texts = X_val.reset_index(drop=True)

true_label_counts = y_true.sum(axis=1)

# ------------------------------------------------------------
# 2.1 Clean vs single-label vs multi-label performance

group_masks = {
    "clean_comments": true_label_counts == 0,
    "single_label_comments": true_label_counts == 1,
    "multi_label_comments": true_label_counts >= 2
}

group_rows = []

for group_name, mask in group_masks.items():
    true_group = y_true[mask]
    pred_group = y_pred_np[mask]

    if len(true_group) == 0:
        continue

    has_positive_labels = true_group.sum() > 0

    group_rows.append({
        "group": group_name,
        "comment_count": int(mask.sum()),
        "exact_match_accuracy": np.mean(
            np.all(true_group == pred_group, axis=1)
        ),
        "micro_recall": (
            recall_score(
                true_group,
                pred_group,
                average="micro",
                zero_division=0
            )
            if has_positive_labels else np.nan
        ),
        "micro_f1": (
            f1_score(
                true_group,
                pred_group,
                average="micro",
                zero_division=0
            )
            if has_positive_labels else np.nan
        ),
        "false_negative_comment_rate": np.mean(
            np.any(
                (true_group == 1) & (pred_group == 0),
                axis=1
            )
        )
    })

group_performance = pd.DataFrame(group_rows)

print("\nClean, single-label, and multi-label performance:")
print(group_performance)

group_performance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "baseline_group_performance.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# 2.2 Partial misses
# Toxic is detected, but a more specific label is missed.

toxic_idx = label_cols.index("toxic")

specific_labels = [
    label for label in label_cols
    if label != "toxic"
]

rare_labels = {
    "severe_toxic",
    "threat",
    "identity_hate"
}

partial_rows = []
partial_example_rows = []

for label in specific_labels:
    label_idx = label_cols.index(label)

    true_pair = (
        (y_true[:, toxic_idx] == 1) &
        (y_true[:, label_idx] == 1)
    )

    toxic_caught = (
        true_pair &
        (y_pred_np[:, toxic_idx] == 1)
    )

    partial_miss = (
        toxic_caught &
        (y_pred_np[:, label_idx] == 0)
    )

    caught_count = int(toxic_caught.sum())
    miss_count = int(partial_miss.sum())

    partial_rows.append({
        "specific_label": label,
        "is_rare_label": label in rare_labels,
        "true_pair_count": int(true_pair.sum()),
        "toxic_caught_count": caught_count,
        "partial_miss_count": miss_count,
        "partial_miss_rate": (
            miss_count / caught_count
            if caught_count > 0 else np.nan
        )
    })

    # Save five confident partial-miss examples.
    miss_indices = np.where(partial_miss)[0]

    selected_indices = miss_indices[
        np.argsort(
            y_prob_np[miss_indices, label_idx]
        )
    ][:5]

    for row_idx in selected_indices:
        true_labels = [
            label_cols[j]
            for j in range(len(label_cols))
            if y_true[row_idx, j] == 1
        ]

        predicted_labels = [
            label_cols[j]
            for j in range(len(label_cols))
            if y_pred_np[row_idx, j] == 1
        ]

        partial_example_rows.append({
            "missed_label": label,
            "true_labels": ", ".join(true_labels),
            "predicted_labels": ", ".join(predicted_labels),
            "specific_label_probability":
                y_prob_np[row_idx, label_idx],
            "comment_text": val_texts.iloc[row_idx]
        })

partial_miss_analysis = pd.DataFrame(partial_rows)

print("\nPartial miss analysis:")
print(partial_miss_analysis)

partial_miss_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "baseline_partial_miss_analysis.csv"
    ),
    index=False
)

pd.DataFrame(partial_example_rows).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "baseline_partial_miss_examples.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# 2.3 Label co-occurrence and joint detection

cooccurrence_rows = []

for i, label_a in enumerate(label_cols):
    for j in range(i + 1, len(label_cols)):
        label_b = label_cols[j]

        pair_mask = (
            (y_true[:, i] == 1) &
            (y_true[:, j] == 1)
        )

        pair_count = int(pair_mask.sum())

        if pair_count == 0:
            continue

        pred_a = y_pred_np[pair_mask, i]
        pred_b = y_pred_np[pair_mask, j]

        both_detected = (
            (pred_a == 1) &
            (pred_b == 1)
        )

        both_missed = (
            (pred_a == 0) &
            (pred_b == 0)
        )

        cooccurrence_rows.append({
            "label_a": label_a,
            "label_b": label_b,
            "true_cooccurrence_count": pair_count,
            "both_labels_detected_rate":
                both_detected.mean(),
            "at_least_one_label_missed_rate":
                1 - both_detected.mean(),
            "both_labels_missed_rate":
                both_missed.mean()
        })

cooccurrence_analysis = pd.DataFrame(
    cooccurrence_rows
).sort_values(
    by="true_cooccurrence_count",
    ascending=False
)

print("\nLabel co-occurrence error analysis:")
print(cooccurrence_analysis)

cooccurrence_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "baseline_cooccurrence_error_analysis.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# 2.4 Representative false-positive and false-negative examples

error_rows = []
examples_per_type = 3

for label_idx, label in enumerate(label_cols):

    fp_indices = np.where(
        (y_true[:, label_idx] == 0) &
        (y_pred_np[:, label_idx] == 1)
    )[0]

    fn_indices = np.where(
        (y_true[:, label_idx] == 1) &
        (y_pred_np[:, label_idx] == 0)
    )[0]

    # Highest-probability false positives
    selected_fp = fp_indices[
        np.argsort(
            y_prob_np[fp_indices, label_idx]
        )[::-1]
    ][:examples_per_type]

    # Lowest-probability false negatives
    selected_fn = fn_indices[
        np.argsort(
            y_prob_np[fn_indices, label_idx]
        )
    ][:examples_per_type]

    for error_type, selected_indices in {
        "false_positive": selected_fp,
        "false_negative": selected_fn
    }.items():

        for row_idx in selected_indices:
            true_labels = [
                label_cols[j]
                for j in range(len(label_cols))
                if y_true[row_idx, j] == 1
            ]

            predicted_labels = [
                label_cols[j]
                for j in range(len(label_cols))
                if y_pred_np[row_idx, j] == 1
            ]

            error_rows.append({
                "label": label,
                "error_type": error_type,
                "predicted_probability":
                    y_prob_np[row_idx, label_idx],
                "true_labels": ", ".join(true_labels),
                "predicted_labels": ", ".join(predicted_labels),
                "comment_text": val_texts.iloc[row_idx]
            })

error_examples = pd.DataFrame(error_rows)

error_examples.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "baseline_representative_error_examples.csv"
    ),
    index=False
)

print("\nPart 2 completed.")
print("Structured error-analysis outputs saved in outputs/.")


# ============================================================
# PART 3: IMBALANCE MITIGATION
# Label-specific threshold tuning and class-weighted Logistic Regression
# ============================================================
"""
This section tests whether label-specific thresholds and class
weighting can improve rare-label recall and reduce partial misses.
All final models are evaluated on the same validation set.
"""

from sklearn.metrics import precision_recall_curve, accuracy_score, roc_auc_score

print("\n" + "=" * 60)
print("PART 3: IMBALANCE MITIGATION")
print("=" * 60)

RANDOM_SEED = 42
y_val_array = y_val.to_numpy()

# ------------------------------------------------------------
# 3.1 Tune one probability threshold for each label

# Create an inner split using only the original training data.
X_inner_train, X_inner_tune, y_inner_train, y_inner_tune = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=RANDOM_SEED
)

# Use the same TF-IDF configuration as the baseline.
inner_tfidf = TfidfVectorizer(
    lowercase=True,
    max_features=50000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95
)

X_inner_train_tfidf = inner_tfidf.fit_transform(X_inner_train)
X_inner_tune_tfidf = inner_tfidf.transform(X_inner_tune)

inner_model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=RANDOM_SEED
    )
)

print("\nTraining model for threshold tuning...")
inner_model.fit(X_inner_train_tfidf, y_inner_train)

inner_prob = inner_model.predict_proba(X_inner_tune_tfidf)
y_inner_tune_array = y_inner_tune.to_numpy()

tuned_thresholds = []
threshold_rows = []

for idx, label in enumerate(label_cols):
    y_true_label = y_inner_tune_array[:, idx]
    y_prob_label = inner_prob[:, idx]

    precision_values, recall_values, thresholds = precision_recall_curve(
        y_true_label,
        y_prob_label
    )

    # The last precision/recall point has no matching threshold.
    precision_values = precision_values[:-1]
    recall_values = recall_values[:-1]

    f1_values = (
        2 * precision_values * recall_values /
        np.maximum(precision_values + recall_values, 1e-10)
    )

    best_index = int(np.argmax(f1_values))
    best_threshold = float(thresholds[best_index])

    # Prevent extreme thresholds.
    best_threshold = float(np.clip(best_threshold, 0.01, 0.99))
    tuned_thresholds.append(best_threshold)

    default_pred = (y_prob_label >= 0.5).astype(int)
    tuned_pred = (y_prob_label >= best_threshold).astype(int)

    threshold_rows.append({
        "label": label,
        "selected_threshold": best_threshold,
        "default_precision": precision_score(
            y_true_label,
            default_pred,
            zero_division=0
        ),
        "default_recall": recall_score(
            y_true_label,
            default_pred,
            zero_division=0
        ),
        "default_f1": f1_score(
            y_true_label,
            default_pred,
            zero_division=0
        ),
        "tuned_precision": precision_score(
            y_true_label,
            tuned_pred,
            zero_division=0
        ),
        "tuned_recall": recall_score(
            y_true_label,
            tuned_pred,
            zero_division=0
        ),
        "tuned_f1": f1_score(
            y_true_label,
            tuned_pred,
            zero_division=0
        )
    })

tuned_thresholds = np.array(tuned_thresholds)

threshold_results = pd.DataFrame(threshold_rows)

print("\nSelected thresholds:")
print(threshold_results)

threshold_results.to_csv(
    os.path.join(OUTPUT_DIR, "threshold_tuning_results.csv"),
    index=False
)

# Apply tuned thresholds to the original baseline probabilities.
threshold_predictions = (
    y_prob >= tuned_thresholds.reshape(1, -1)
).astype(int)

# ------------------------------------------------------------
# 3.2 Train class-weighted Logistic Regression

balanced_model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_SEED
    )
)

print("\nTraining class-weighted Logistic Regression...")
balanced_model.fit(X_train_tfidf, y_train)

balanced_prob = balanced_model.predict_proba(X_val_tfidf)
balanced_pred = (balanced_prob >= 0.5).astype(int)

print("Class-weighted model finished.")

# ------------------------------------------------------------
# 3.3 Overall model comparison

model_results = {
    "baseline": {
        "pred": np.asarray(y_pred),
        "prob": np.asarray(y_prob)
    },
    "threshold_tuned": {
        "pred": threshold_predictions,
        "prob": np.asarray(y_prob)
    },
    "class_weight_balanced": {
        "pred": balanced_pred,
        "prob": balanced_prob
    }
}

overall_rows = []

for model_name, result in model_results.items():
    overall_rows.append({
        "model": model_name,
        "micro_f1": f1_score(
            y_val_array,
            result["pred"],
            average="micro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            y_val_array,
            result["pred"],
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            y_val_array,
            result["pred"],
            average="weighted",
            zero_division=0
        ),
        "hamming_loss": hamming_loss(
            y_val_array,
            result["pred"]
        ),
        "exact_match_accuracy": accuracy_score(
            y_val_array,
            result["pred"]
        ),
        "macro_roc_auc": roc_auc_score(
            y_val_array,
            result["prob"],
            average="macro"
        ),
        "average_predicted_labels":
            result["pred"].sum(axis=1).mean()
    })

overall_comparison = pd.DataFrame(overall_rows)

print("\nOverall model comparison:")
print(overall_comparison)

overall_comparison.to_csv(
    os.path.join(OUTPUT_DIR, "part3_overall_comparison.csv"),
    index=False
)

# ------------------------------------------------------------
# 3.4 Per-label comparison

per_label_rows = []

for model_name, result in model_results.items():
    for idx, label in enumerate(label_cols):
        true_label = y_val_array[:, idx]
        predicted_label = result["pred"][:, idx]

        fp = int(
            ((true_label == 0) & (predicted_label == 1)).sum()
        )
        fn = int(
            ((true_label == 1) & (predicted_label == 0)).sum()
        )

        per_label_rows.append({
            "model": model_name,
            "label": label,
            "precision": precision_score(
                true_label,
                predicted_label,
                zero_division=0
            ),
            "recall": recall_score(
                true_label,
                predicted_label,
                zero_division=0
            ),
            "f1": f1_score(
                true_label,
                predicted_label,
                zero_division=0
            ),
            "false_positive": fp,
            "false_negative": fn
        })

per_label_comparison = pd.DataFrame(per_label_rows)

print("\nRare-label comparison:")
print(
    per_label_comparison[
        per_label_comparison["label"].isin(
            ["severe_toxic", "threat", "identity_hate"]
        )
    ]
)

per_label_comparison.to_csv(
    os.path.join(OUTPUT_DIR, "part3_per_label_comparison.csv"),
    index=False
)

# ------------------------------------------------------------
# 3.5 Single-label vs multi-label comparison

true_label_count = y_val_array.sum(axis=1)

group_masks = {
    "single_label": true_label_count == 1,
    "multi_label": true_label_count >= 2
}

group_rows = []

for model_name, result in model_results.items():
    for group_name, mask in group_masks.items():
        true_group = y_val_array[mask]
        pred_group = result["pred"][mask]

        group_rows.append({
            "model": model_name,
            "group": group_name,
            "comment_count": int(mask.sum()),
            "exact_match_accuracy": accuracy_score(
                true_group,
                pred_group
            ),
            "micro_recall": recall_score(
                true_group,
                pred_group,
                average="micro",
                zero_division=0
            ),
            "micro_f1": f1_score(
                true_group,
                pred_group,
                average="micro",
                zero_division=0
            ),
            "false_negative_comment_rate": np.mean(
                np.any(
                    (true_group == 1) & (pred_group == 0),
                    axis=1
                )
            )
        })

group_comparison = pd.DataFrame(group_rows)

print("\nSingle-label vs multi-label comparison:")
print(group_comparison)

group_comparison.to_csv(
    os.path.join(OUTPUT_DIR, "part3_group_comparison.csv"),
    index=False
)

# ------------------------------------------------------------
# 3.6 Rare-label partial miss comparison

toxic_idx = label_cols.index("toxic")
rare_specific_labels = [
    "severe_toxic",
    "threat",
    "identity_hate"
]

partial_rows = []

for model_name, result in model_results.items():
    for label in rare_specific_labels:
        label_idx = label_cols.index(label)

        true_pair = (
            (y_val_array[:, toxic_idx] == 1) &
            (y_val_array[:, label_idx] == 1)
        )

        toxic_caught = (
            true_pair &
            (result["pred"][:, toxic_idx] == 1)
        )

        partial_miss = (
            toxic_caught &
            (result["pred"][:, label_idx] == 0)
        )

        caught_count = int(toxic_caught.sum())
        miss_count = int(partial_miss.sum())

        partial_rows.append({
            "model": model_name,
            "specific_label": label,
            "toxic_caught_count": caught_count,
            "partial_miss_count": miss_count,
            "partial_miss_rate":
                miss_count / caught_count
                if caught_count > 0 else np.nan
        })

partial_comparison = pd.DataFrame(partial_rows)

print("\nRare-label partial miss comparison:")
print(partial_comparison)

partial_comparison.to_csv(
    os.path.join(OUTPUT_DIR, "part3_partial_miss_comparison.csv"),
    index=False
)

print("\nPart 3 completed.")
print("Imbalance-mitigation results saved in outputs/.")



# ============================================================
# PART 4: BERT-BASED MODEL
# DistilBERT multi-label classification and error comparison
# ============================================================
"""
This section fine-tunes DistilBERT with six independent label outputs.
It evaluates overall performance, rare-label recall, multi-label
errors, and partial misses using the same validation split.
"""

import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)

print("\n" + "=" * 60)
print("PART 4: BERT-BASED MODEL")
print("=" * 60)

# ------------------------------------------------------------
# 4.1 Configuration

RUN_BERT = False
BERT_DEBUG = False

BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_SEED = 42
BERT_MAX_LENGTH = 128
BERT_THRESHOLD = 0.5


if RUN_BERT:

    set_seed(BERT_SEED)

    # Debug mode checks whether the code works.
    # Final results require BERT_DEBUG = False.
    if BERT_DEBUG:
        bert_X_train = X_train.iloc[:3000].reset_index(drop=True)
        bert_y_train = y_train.iloc[:3000].reset_index(drop=True)

        bert_X_val = X_val.iloc[:1000].reset_index(drop=True)
        bert_y_val = y_val.iloc[:1000].reset_index(drop=True)

        bert_epochs = 1
        bert_prefix = "bert_debug"

        print("\nBERT debug mode: results are not final.")

    else:
        bert_X_train = X_train.reset_index(drop=True)
        bert_y_train = y_train.reset_index(drop=True)

        bert_X_val = X_val.reset_index(drop=True)
        bert_y_val = y_val.reset_index(drop=True)

        bert_epochs = 2
        bert_prefix = "bert_final"

    print(f"Training comments: {len(bert_X_train)}")
    print(f"Validation comments: {len(bert_X_val)}")

    # --------------------------------------------------------
    # 4.2 Prepare Hugging Face datasets

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL_NAME
    )

    train_dataset = Dataset.from_dict({
        "text": bert_X_train.tolist(),
        "labels": bert_y_train.to_numpy(
            dtype=np.float32
        ).tolist()
    })

    validation_dataset = Dataset.from_dict({
        "text": bert_X_val.tolist(),
        "labels": bert_y_val.to_numpy(
            dtype=np.float32
        ).tolist()
    })


    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=BERT_MAX_LENGTH
        )


    train_dataset = train_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"]
    )

    validation_dataset = validation_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"]
    )

    # --------------------------------------------------------
    # 4.3 Train DistilBERT

    bert_model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=len(label_cols),
        problem_type="multi_label_classification"
    )

    training_arguments = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "bert_training"),
        num_train_epochs=bert_epochs,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_strategy="no",
        report_to="none",
        seed=BERT_SEED,
        fp16=torch.cuda.is_available()
    )

    bert_trainer = Trainer(
        model=bert_model,
        args=training_arguments,
        train_dataset=train_dataset,
        processing_class=tokenizer
    )

    print("\nTraining DistilBERT...")
    bert_trainer.train()

    prediction_output = bert_trainer.predict(
        validation_dataset
    )

    bert_logits = prediction_output.predictions
    bert_prob = torch.sigmoid(
        torch.tensor(bert_logits)
    ).numpy()

    bert_true = bert_y_val.to_numpy()
    bert_pred = (
        bert_prob >= BERT_THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # 4.4 Overall BERT results

    try:
        bert_macro_auc = roc_auc_score(
            bert_true,
            bert_prob,
            average="macro"
        )
    except ValueError:
        bert_macro_auc = np.nan

    bert_overall = pd.DataFrame([{
        "model": "distilbert",
        "micro_f1": f1_score(
            bert_true,
            bert_pred,
            average="micro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            bert_true,
            bert_pred,
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            bert_true,
            bert_pred,
            average="weighted",
            zero_division=0
        ),
        "hamming_loss": hamming_loss(
            bert_true,
            bert_pred
        ),
        "exact_match_accuracy": accuracy_score(
            bert_true,
            bert_pred
        ),
        "macro_roc_auc": bert_macro_auc
    }])

    print("\nBERT overall results:")
    print(bert_overall)

    bert_overall.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{bert_prefix}_overall_results.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # 4.5 Per-label and rare-label results

    bert_label_rows = []

    for idx, label in enumerate(label_cols):
        bert_label_rows.append({
            "model": "distilbert",
            "label": label,
            "precision": precision_score(
                bert_true[:, idx],
                bert_pred[:, idx],
                zero_division=0
            ),
            "recall": recall_score(
                bert_true[:, idx],
                bert_pred[:, idx],
                zero_division=0
            ),
            "f1": f1_score(
                bert_true[:, idx],
                bert_pred[:, idx],
                zero_division=0
            ),
            "false_positive": int(
                (
                    (bert_true[:, idx] == 0) &
                    (bert_pred[:, idx] == 1)
                ).sum()
            ),
            "false_negative": int(
                (
                    (bert_true[:, idx] == 1) &
                    (bert_pred[:, idx] == 0)
                ).sum()
            )
        })

    bert_per_label = pd.DataFrame(bert_label_rows)

    print("\nBERT rare-label results:")
    print(
        bert_per_label[
            bert_per_label["label"].isin(
                ["severe_toxic", "threat", "identity_hate"]
            )
        ]
    )

    bert_per_label.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{bert_prefix}_per_label_results.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # 4.6 Single-label vs multi-label results

    bert_label_count = bert_true.sum(axis=1)

    bert_group_rows = []

    for group_name, mask in {
        "single_label": bert_label_count == 1,
        "multi_label": bert_label_count >= 2
    }.items():

        if mask.sum() == 0:
            continue

        bert_group_rows.append({
            "model": "distilbert",
            "group": group_name,
            "comment_count": int(mask.sum()),
            "exact_match_accuracy": accuracy_score(
                bert_true[mask],
                bert_pred[mask]
            ),
            "micro_recall": recall_score(
                bert_true[mask],
                bert_pred[mask],
                average="micro",
                zero_division=0
            ),
            "micro_f1": f1_score(
                bert_true[mask],
                bert_pred[mask],
                average="micro",
                zero_division=0
            )
        })

    bert_group_results = pd.DataFrame(
        bert_group_rows
    )

    bert_group_results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{bert_prefix}_group_results.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # 4.7 Rare-label partial misses

    toxic_idx = label_cols.index("toxic")
    bert_partial_rows = []

    for label in [
        "severe_toxic",
        "threat",
        "identity_hate"
    ]:
        label_idx = label_cols.index(label)

        true_pair = (
            (bert_true[:, toxic_idx] == 1) &
            (bert_true[:, label_idx] == 1)
        )

        toxic_caught = (
            true_pair &
            (bert_pred[:, toxic_idx] == 1)
        )

        partial_miss = (
            toxic_caught &
            (bert_pred[:, label_idx] == 0)
        )

        caught_count = int(toxic_caught.sum())
        miss_count = int(partial_miss.sum())

        bert_partial_rows.append({
            "model": "distilbert",
            "specific_label": label,
            "toxic_caught_count": caught_count,
            "partial_miss_count": miss_count,
            "partial_miss_rate":
                miss_count / caught_count
                if caught_count > 0 else np.nan
        })

    bert_partial_results = pd.DataFrame(
        bert_partial_rows
    )

    print("\nBERT rare-label partial misses:")
    print(bert_partial_results)

    bert_partial_results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{bert_prefix}_partial_miss_results.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # 4.8 Save predictions for Part 5

    bert_predictions = pd.DataFrame({
        "comment_text": bert_X_val
    })

    for idx, label in enumerate(label_cols):
        bert_predictions[f"true_{label}"] = bert_true[:, idx]
        bert_predictions[f"pred_{label}"] = bert_pred[:, idx]
        bert_predictions[f"prob_{label}"] = bert_prob[:, idx]

    bert_predictions.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{bert_prefix}_predictions.csv"
        ),
        index=False
    )

    print("\nPart 4 completed.")

    if BERT_DEBUG:
        print(
            "Debug only. Set BERT_DEBUG = False "
            "for the final comparable results."
        )



# ============================================================
# PART 5: FINAL COMPARISON AND FIGURES
# Report-ready tables and visualizations
# ============================================================

"""
This section combines the Logistic Regression and DistilBERT
results into final report-ready tables and figures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("\n" + "=" * 60)
print("PART 5: FINAL COMPARISON AND FIGURES")
print("=" * 60)

# ------------------------------------------------------------
# 5.1 Load Part 3 and Part 4 results

required_files = {
    "part3_overall": "part3_overall_comparison.csv",
    "part3_labels": "part3_per_label_comparison.csv",
    "part3_groups": "part3_group_comparison.csv",
    "part3_partial": "part3_partial_miss_comparison.csv",
    "bert_overall": "bert_final_overall_results.csv",
    "bert_labels": "bert_final_per_label_results.csv",
    "bert_groups": "bert_final_group_results.csv",
    "bert_partial": "bert_final_partial_miss_results.csv"
}

for filename in required_files.values():
    file_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Missing required result file: {file_path}"
        )

part3_overall = pd.read_csv(
    os.path.join(OUTPUT_DIR, required_files["part3_overall"])
)

part3_labels = pd.read_csv(
    os.path.join(OUTPUT_DIR, required_files["part3_labels"])
)

part3_groups = pd.read_csv(
    os.path.join(OUTPUT_DIR, required_files["part3_groups"])
)

part3_partial = pd.read_csv(
    os.path.join(OUTPUT_DIR, required_files["part3_partial"])
)

bert_overall = pd.read_csv(
    os.path.join(OUTPUT_DIR, required_files["bert_overall"])
)

bert_labels = pd.read_csv(
    os.path.join(OUTPUT_DIR, required_files["bert_labels"])
)

bert_groups = pd.read_csv(
    os.path.join(OUTPUT_DIR, required_files["bert_groups"])
)

bert_partial = pd.read_csv(
    os.path.join(OUTPUT_DIR, required_files["bert_partial"])
)

# ------------------------------------------------------------
# 5.2 Create final comparison tables

model_order = [
    "baseline",
    "threshold_tuned",
    "class_weight_balanced",
    "distilbert"
]

overall_columns = [
    "model",
    "micro_f1",
    "macro_f1",
    "weighted_f1",
    "hamming_loss",
    "exact_match_accuracy",
    "macro_roc_auc"
]

final_overall = pd.concat(
    [part3_overall, bert_overall],
    ignore_index=True
)

final_overall = final_overall[
    overall_columns
].drop_duplicates(
    subset="model",
    keep="last"
)

final_overall["model"] = pd.Categorical(
    final_overall["model"],
    categories=model_order,
    ordered=True
)

final_overall = final_overall.sort_values(
    "model"
).reset_index(drop=True)

final_per_label = pd.concat(
    [part3_labels, bert_labels],
    ignore_index=True
).drop_duplicates(
    subset=["model", "label"],
    keep="last"
)

final_groups = pd.concat(
    [part3_groups, bert_groups],
    ignore_index=True
).drop_duplicates(
    subset=["model", "group"],
    keep="last"
)

# Keep only metrics available for all four models.
final_groups = final_groups[
    [
        "model",
        "group",
        "comment_count",
        "exact_match_accuracy",
        "micro_recall",
        "micro_f1"
    ]
]

final_partial = pd.concat(
    [part3_partial, bert_partial],
    ignore_index=True
).drop_duplicates(
    subset=["model", "specific_label"],
    keep="last"
)

rare_labels = [
    "severe_toxic",
    "threat",
    "identity_hate"
]

final_rare_labels = final_per_label[
    final_per_label["label"].isin(rare_labels)
].copy()

final_overall.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_overall_model_comparison.csv"
    ),
    index=False
)

final_per_label.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_per_label_model_comparison.csv"
    ),
    index=False
)

final_rare_labels.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_rare_label_comparison.csv"
    ),
    index=False
)

final_groups.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_group_comparison.csv"
    ),
    index=False
)

final_partial.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_partial_miss_comparison.csv"
    ),
    index=False
)

print("\nFinal overall model comparison:")
print(final_overall)

print("\nFinal rare-label comparison:")
print(
    final_rare_labels[
        ["model", "label", "precision", "recall", "f1"]
    ]
)

# ------------------------------------------------------------
# 5.3 Overall F1 comparison figure

overall_plot = final_overall.set_index("model")[
    ["micro_f1", "macro_f1"]
]

ax = overall_plot.plot(
    kind="bar",
    figsize=(9, 5)
)

ax.set_title("Overall Model Performance")
ax.set_xlabel("Model")
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=20)
ax.legend(["Micro-F1", "Macro-F1"])

plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "final_overall_f1_comparison.png"
    ),
    dpi=300
)
plt.close()

# ------------------------------------------------------------
# 5.4 Rare-label recall figure

rare_recall_plot = final_rare_labels.pivot(
    index="model",
    columns="label",
    values="recall"
)

rare_recall_plot = rare_recall_plot.reindex(
    model_order
)

ax = rare_recall_plot.plot(
    kind="bar",
    figsize=(9, 5)
)

ax.set_title("Rare-Label Recall by Model")
ax.set_xlabel("Model")
ax.set_ylabel("Recall")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "final_rare_label_recall.png"
    ),
    dpi=300
)
plt.close()

# ------------------------------------------------------------
# 5.5 Partial-miss comparison figure

partial_plot = final_partial.pivot(
    index="model",
    columns="specific_label",
    values="partial_miss_rate"
)

partial_plot = partial_plot.reindex(
    model_order
)

ax = partial_plot.plot(
    kind="bar",
    figsize=(9, 5)
)

ax.set_title("Rare-Label Partial-Miss Rate")
ax.set_xlabel("Model")
ax.set_ylabel("Partial-Miss Rate")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "final_partial_miss_comparison.png"
    ),
    dpi=300
)
plt.close()


print("\nPart 5 completed.")
print("Final tables and figures saved in outputs/.")