<div align="center">

# MSE641 Toxic Comment Classification

### Systematic Error Analysis for Multi-Label Toxic Comment Classification

**Team**  
Cecilia Wang (21226500)  
Junhao Liu (20960352)

</div>

---

## Project Overview

This project investigates multi-label toxic comment classification using the Kaggle Jigsaw Toxic Comment Classification dataset. The dataset contains Wikipedia comments labeled with six non-exclusive toxicity categories:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

The goal of this project is not only to build toxic comment classifiers, but also to understand where the models fail. In particular, we focus on rare labels, multi-label comments, label co-occurrence, false positives, false negatives, and partial misses where a model detects general toxicity but misses a more specific toxicity label.

---

## Repository Structure

```text
MSE641-Project-Toxic-Comment-Classification/
│
├── data/
│   └── train.csv
│
├── outputs/
│   ├── baseline_overall_metrics.csv
│   ├── baseline_per_label_metrics.csv
│   ├── baseline_validation_predictions.csv
│   ├── part3_overall_comparison.csv
│   ├── part3_per_label_comparison.csv
│   ├── bert_final_overall_results.csv
│   ├── bert_final_per_label_results.csv
│   ├── final_overall_model_comparison.csv
│   ├── final_rare_label_comparison.csv
│   ├── final_group_comparison.csv
│   └── final_partial_miss_comparison.csv
│         Only selected report-relevant outputs are shown below.
├── baseline.py
├── final_project.py
├── requirements.txt
├── ShortCut.txt
├── Project-guide.pdf
├── Project_Milestone_Toxic_Comment.pdf
└── README.md
```

---

## Methods

### 1. Data Validation and Preprocessing

The project first loads the Jigsaw training dataset and checks for:

- missing values
- duplicate IDs and comments
- binary label validity
- corrupted spreadsheet-style entries
- label distribution
- clean, toxic, single-label, and multi-label comment counts
- label co-occurrence patterns

Text cleaning is intentionally minimal. We normalize whitespace but do not remove profanity, punctuation, capitalization, or slang because these may carry toxicity signals.

### 2. Baseline Model

The baseline model uses:

- TF-IDF vectorization
- unigrams and bigrams
- maximum 50,000 features
- One-vs-Rest Logistic Regression
- default probability threshold of 0.5

This baseline provides an interpretable reference point for later model comparison and error analysis.

### 3. Extended Evaluation

The project evaluates models using both aggregate and per-label metrics:

- micro-F1
- macro-F1
- weighted-F1
- Hamming loss
- exact match accuracy
- ROC-AUC
- per-label precision
- per-label recall
- per-label F1-score
- false positives and false negatives

### 4. Structured Error Analysis

The final project extends the milestone baseline with structured error analysis, including:

- clean vs. single-label vs. multi-label performance
- label co-occurrence error analysis
- partial miss analysis
- rare-label performance comparison
- representative false-positive and false-negative examples

A partial miss occurs when toxic is correctly detected but another true, more specific toxicity label is missed. The final comparison focuses particularly on the rare labels severe_toxic, threat, and identity_hate.

### 5. Imbalance Mitigation

Because the dataset is highly imbalanced, we test two imbalance mitigation methods:

1. **Label-specific threshold tuning**  
   Each label receives its own probability threshold based on an inner tuning split.

2. **Class-weighted Logistic Regression**  
   Logistic Regression is trained with `class_weight="balanced"` to increase the influence of rare positive examples.

### 6. DistilBERT Model

We also fine-tune a DistilBERT-based transformer model for multi-label classification.

Configuration:

- model: `distilbert-base-uncased`
- max sequence length: 128
- learning rate: 2e-5
- train batch size: 8
- evaluation batch size: 16
- weight decay: 0.01
- epochs: 2
- output layer: six-label multi-label classification head with sigmoid probabilities
- prediction threshold: 0.5

---

## Key Results

| Model | Micro-F1 | Macro-F1 | Weighted-F1 | Hamming Loss | Exact Match | Macro ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.661 | 0.457 | 0.651 | 0.0199 | 0.919 | 0.976 |
| Threshold tuned | 0.726 | 0.600 | 0.734 | 0.0203 | 0.912 | 0.976 |
| Class-weighted LR | 0.685 | 0.563 | 0.704 | 0.0291 | 0.881 | 0.978 |
| DistilBERT | 0.796 | 0.636 | 0.787 | 0.0146 | 0.929 | 0.991 |

DistilBERT achieves the strongest overall performance, while threshold tuning and class-weighted Logistic Regression improve recall on rare labels. The results show an important trade-off between overall performance, rare-label recall, precision, and partial-miss reduction.

---

## Reproducibility

### 1. Create environment

```bash
conda create -n mse641 python=3.11 -y
conda activate mse641
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If needed, install the core packages manually:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn matplotlib scipy torch transformers accelerate datasets tqdm
```

### 3. Run the baseline

```bash
python baseline.py
```

This runs the TF-IDF + One-vs-Rest Logistic Regression baseline and saves baseline metrics, validation predictions, and error files into the `outputs/` folder.

### 4. Run the final project pipeline

```bash
python final_project.py
```

By default, the final script uses saved DistilBERT output files in the `outputs/` folder for final comparison. This avoids retraining DistilBERT every time because transformer training is time-consuming and may produce small numerical differences across hardware.

---

## Running DistilBERT from Scratch

To reproduce the full DistilBERT results from scratch, open `final_project.py` and set:

```python
RUN_BERT = True
BERT_DEBUG = False
```

Then run:

```bash
python final_project.py
```

This will train the full DistilBERT model and regenerate the `bert_final_*.csv` files.

For a quick debug test, set:

```python
RUN_BERT = True
BERT_DEBUG = True
```

This uses a smaller sample and only checks whether the DistilBERT pipeline runs successfully. Debug results should not be used in the final report.

For routine runs without retraining DistilBERT, use:

```python
RUN_BERT = False
BERT_DEBUG = False
```

---

## Main Output Files

The most important final output files are:

```text
outputs/final_overall_model_comparison.csv
outputs/final_per_label_model_comparison.csv
outputs/final_rare_label_comparison.csv
outputs/final_group_comparison.csv
outputs/final_partial_miss_comparison.csv
outputs/final_overall_f1_comparison.png
outputs/final_rare_label_recall.png
outputs/final_partial_miss_comparison.png
```

These files are used to support the final report results and discussion.

---

## Notes

The submitted final report uses the saved final output files in the `outputs/` folder. Re-running the full DistilBERT training may take several hours depending on hardware and may produce slightly different results because of hardware and library differences. For this reason, the saved final CSV files should be treated as the report-ready results.

---

## Project Report

The final report discusses:

- dataset characteristics
- baseline model design
- imbalance mitigation methods
- DistilBERT comparison
- rare-label performance
- single-label vs. multi-label performance
- partial-miss analysis
- qualitative error patterns
- limitations and future work

---

