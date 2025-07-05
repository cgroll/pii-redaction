# %%
"""
Main analysis script for PII detection and evaluation.
This script is designed to be run cell-by-cell in an IDE like VS Code.
"""

# --- Imports ---
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

from pii_redaction.data_loader import load_pii_dataset
from pii_redaction.detection import inspect_text_with_dlp
from pii_redaction.evaluation import (
    highlight_spans_in_text,
    calculate_char_level_metrics
)

# %%
# --- 1. Load Data ---
print("Loading dataset...")
dataset = load_pii_dataset()
print(f"Dataset loaded with {len(dataset)} entries.")


# %%
# --- 2. Sample Data and Run DLP Detection ---
N_SAMPLES = 5  # Using a smaller number for a quicker run
random_indices = np.random.choice(len(dataset), size=N_SAMPLES, replace=False)

all_dlp_findings = []
all_ground_truths = []

print(f"Processing {N_SAMPLES} random samples...")
for idx in tqdm(random_indices, desc="Running DLP on samples"):
    entry = dataset[int(idx)]
    source_text = entry['source_text']

    # Get DLP findings
    dlp_findings = inspect_text_with_dlp(source_text)
    if not dlp_findings.empty:
        dlp_findings['sample_idx'] = idx
        all_dlp_findings.append(dlp_findings)

    # Get ground truth annotations
    ground_truth = pd.DataFrame(entry['privacy_mask'])
    if not ground_truth.empty:
        ground_truth['sample_idx'] = idx
        all_ground_truths.append(ground_truth)

dlp_results_df = pd.concat(all_dlp_findings, ignore_index=True) if all_dlp_findings else pd.DataFrame()
ground_truth_df = pd.concat(all_ground_truths, ignore_index=True) if all_ground_truths else pd.DataFrame()

print(f"\nFound {len(dlp_results_df)} total DLP findings across {N_SAMPLES} samples.")
display(dlp_results_df.head())


# %%
# --- 3. Calculate Metrics for Each Sample ---
metrics_list = []

for idx in tqdm(random_indices, desc="Calculating metrics"):
    entry = dataset[int(idx)]
    text_length = len(entry['source_text'])

    sample_truth = ground_truth_df[ground_truth_df['sample_idx'] == idx]
    sample_preds = dlp_results_df[dlp_results_df['sample_idx'] == idx]

    if not sample_truth.empty:
        metrics = calculate_char_level_metrics(sample_truth, sample_preds, text_length)
        metrics['sample_idx'] = idx
        metrics['text_length'] = text_length
        metrics['truth_masked_chars'] = sample_truth.apply(lambda row: row['end'] - row['start'], axis=1).sum()
        metrics['dlp_masked_chars'] = sample_preds.apply(lambda row: row['end'] - row['start'], axis=1).sum()
        metrics_list.append(metrics)

metrics_df = pd.DataFrame(metrics_list)
print("\nMetrics Summary (DLP vs. Ground Truth):")
display(metrics_df.describe())


# %%
# --- 4. Visualize Results for a Few Samples ---
N_VISUALIZATIONS = 4
print(f"\nVisualizing results for {N_VISUALIZATIONS} samples...")

for idx in random_indices[:N_VISUALIZATIONS]:
    source_text = dataset[int(idx)]['source_text']
    sample_truth = ground_truth_df[ground_truth_df['sample_idx'] == idx]
    sample_dlp = dlp_results_df[dlp_results_df['sample_idx'] == idx]

    print(f"\n--- Sample ID: {idx} ---")

    print("\nGround Truth Mask (Gold Standard):")
    display(highlight_spans_in_text(source_text, sample_truth, highlight_color='#ffcccb')) # Light Red

    print("\nDLP Findings (Predictions):")
    display(highlight_spans_in_text(source_text, sample_dlp, highlight_color='#c8e6c9')) # Light Green


# %%
# --- 5. Plot Aggregate Metrics ---
print("\nPlotting aggregate metric distributions...")
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.boxplot(y=metrics_df['precision'], ax=axes[0])
axes[0].set_title('Distribution of Precision Scores')
axes[0].set_ylabel('Precision')

sns.boxplot(y=metrics_df['recall'], ax=axes[1])
axes[1].set_title('Distribution of Recall Scores')
axes[1].set_ylabel('Recall')

sns.boxplot(y=metrics_df['f1_score'], ax=axes[2])
axes[2].set_title('Distribution of F1-Scores')
axes[2].set_ylabel('F1-Score')

plt.tight_layout()
plt.show()