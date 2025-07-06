# %%
"""
Main analysis script for PII detection and evaluation.
This script is designed to be run cell-by-cell in an IDE like VS Code.
"""

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
from pii_redaction.paths import ProjPaths

# %%

dataset = load_pii_dataset()

# %%
dataset_ids = pd.Series(dataset['id'], name='id').reset_index()

# %%

sampled_df = pd.read_csv(ProjPaths.data_path / 'pii_masking_300k_samples.csv')
sampled_df = sampled_df.merge(dataset_ids, on='id', how='left')
sampled_df.rename(columns={'index': 'sample_idx'}, inplace=True)

# load dlp results
dlp_results_df = pd.read_csv(ProjPaths.data_path / 'pii_masking_300k_dlp_results.csv')

# load gemini results
gemini_results_df = pd.read_csv(ProjPaths.data_path / 'pii_masking_300k_gemini_results.csv')
gemini_costs_df = pd.read_csv(ProjPaths.data_path / 'pii_masking_300k_gemini_costs.csv')
unmatched_df = pd.read_csv(ProjPaths.data_path / 'pii_masking_300k_gemini_unmatched.csv')

# %% some stats for gemini

gemini_total_costs = gemini_costs_df['total_cost'].sum()
gemini_avg_latency = gemini_costs_df['latency'].mean()
gemini_total_latency = gemini_costs_df['latency'].sum() / 60 # convert to minutes

print(f"Gemini total costs: ${gemini_total_costs:.2f}")
print(f"Gemini average latency: {gemini_avg_latency:.2f} seconds")
print(f"Gemini total latency: {gemini_total_latency:.2f} minutes")

# %%

all_ground_truths = []

print(f"Processing {len(sampled_df)} samples...")
for _, row in tqdm(sampled_df.iterrows(), desc="Running Gemini on samples"):

    idx = row['sample_idx']
    entry = dataset[int(idx)]

    # Get ground truth annotations
    ground_truth = pd.DataFrame(entry['privacy_mask'])
    if not ground_truth.empty:
        ground_truth['sample_idx'] = idx
        all_ground_truths.append(ground_truth)

ground_truth_df = pd.concat(all_ground_truths, ignore_index=True) if all_ground_truths else pd.DataFrame()

# %%

n_samples_with_pii = ground_truth_df['sample_idx'].nunique()
print(f"Number of samples with ground truth PII: {n_samples_with_pii}")

# %% calculate metrics for dlp

def calculate_metrics_for_samples(pii_results_df, ground_truth_df, sampled_df, dataset):
    metrics_list = []

    for _, row in tqdm(sampled_df.iterrows(), desc="Calculating metrics"):

        this_idx = row['sample_idx']

        entry = dataset[int(this_idx)]
        text_length = len(entry['source_text'])

        sample_truth = ground_truth_df[ground_truth_df['sample_idx'] == this_idx]
        sample_preds = pii_results_df[pii_results_df['sample_idx'] == this_idx]

        metrics = calculate_char_level_metrics(sample_truth, sample_preds, text_length)
        
        metrics['sample_idx'] = this_idx
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df.merge(sampled_df, on='sample_idx', how='left')

    return metrics_df

# %%

dlp_metrics_df = calculate_metrics_for_samples(dlp_results_df, ground_truth_df, sampled_df, dataset)
gemini_metrics_df = calculate_metrics_for_samples(gemini_results_df, ground_truth_df, sampled_df, dataset)

# %%

dlp_metrics_df['f1_score'].mean()
gemini_metrics_df['f1_score'].mean()

# %% scatterplots for metrics

plt.figure(figsize=(8, 6))
sns.scatterplot(data=dlp_metrics_df, x='recall', y='precision')
plt.title('Precision vs. Recall for DLP')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()


plt.figure(figsize=(8, 6))
sns.scatterplot(data=gemini_metrics_df, x='recall', y='precision')
plt.title('Precision vs. Recall for Gemini')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# %% boxplots for metrics

print("\nPlotting aggregate metric distributions...")
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Combine metrics into single dataframe for comparison
dlp_metrics_long = dlp_metrics_df[['precision', 'recall', 'f1_score']].melt(var_name='metric', value_name='value')
dlp_metrics_long['model'] = 'DLP'
gemini_metrics_long = gemini_metrics_df[['precision', 'recall', 'f1_score']].melt(var_name='metric', value_name='value') 
gemini_metrics_long['model'] = 'Gemini'
combined_metrics = pd.concat([dlp_metrics_long, gemini_metrics_long])

# Create boxplots
sns.boxplot(data=combined_metrics[combined_metrics.metric == 'precision'], 
            x='model', y='value', ax=axes[0])
axes[0].set_title('Distribution of Precision Scores')
axes[0].set_ylabel('Precision')

sns.boxplot(data=combined_metrics[combined_metrics.metric == 'recall'],
            x='model', y='value', ax=axes[1])
axes[1].set_title('Distribution of Recall Scores') 
axes[1].set_ylabel('Recall')

sns.boxplot(data=combined_metrics[combined_metrics.metric == 'f1_score'],
            x='model', y='value', ax=axes[2])
axes[2].set_title('Distribution of F1-Scores')
axes[2].set_ylabel('F1-Score')

plt.tight_layout()
plt.show()

# %% metrics for dlp for each language

# Create boxplot of F1 scores by language
plt.figure(figsize=(10, 6))
sns.boxplot(data=dlp_metrics_df, x='language', y='f1_score')
plt.title('Distribution of F1 Scores by Language (DLP)')
plt.xlabel('Language')
plt.ylabel('F1 Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Create boxplot of F1 scores by language
plt.figure(figsize=(10, 6))
sns.boxplot(data=gemini_metrics_df, x='language', y='f1_score')
plt.title('Distribution of F1 Scores by Language (Gemini)')
plt.xlabel('Language')
plt.ylabel('F1 Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% DEV: started | analyse different PII types

print("Ground truth unique labels:")
print(ground_truth_df['label'].unique())

print("\nGemini unique types:")
print(gemini_results_df['type'].unique())

print("\nDLP unique labels:")
print(dlp_results_df['label'].unique())








# %% DEV show some samples

plt.figure(figsize=(10, 6))
plt.scatter(gemini_metrics_df['accuracy'], dlp_metrics_df['accuracy'], alpha=0.5)
plt.xlabel('Gemini Accuracy')
plt.ylabel('DLP Accuracy')
plt.title('Comparison of Gemini vs DLP Accuracy')
plt.plot([0, 1], [0, 1], 'r--')  # Add diagonal reference line
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(gemini_metrics_df['f1_score'], dlp_metrics_df['f1_score'], alpha=0.5)
plt.xlabel('Gemini F1 Score')
plt.ylabel('DLP F1 Score')
plt.title('Comparison of Gemini vs DLP F1 Score')
plt.plot([0, 1], [0, 1], 'r--')  # Add diagonal reference line
plt.grid(True)
plt.tight_layout()
plt.show()


# %%

metrics_diff = sampled_df.copy()
metrics_diff['f1_score_diff'] = gemini_metrics_df['f1_score'] - dlp_metrics_df['f1_score']
metrics_diff['accuracy_diff'] = gemini_metrics_df['accuracy'] - dlp_metrics_df['accuracy']
metrics_diff['recall_diff'] = gemini_metrics_df['recall'] - dlp_metrics_df['recall']
metrics_diff['precision_diff'] = gemini_metrics_df['precision'] - dlp_metrics_df['precision']

metrics_diff

# %%

# Find 5 German samples with largest absolute F1 score difference
german_samples = metrics_diff[metrics_diff['language'] == 'German'].copy()
german_samples = german_samples.nlargest(5, 'f1_score_diff')
german_samples[['sample_idx', 'language', 'f1_score_diff']]


# %%

for this_idx in german_samples['sample_idx']:

    source_text = dataset[int(this_idx)]['source_text']
    sample_truth = ground_truth_df[ground_truth_df['sample_idx'] == this_idx]
    sample_dlp = dlp_results_df[dlp_results_df['sample_idx'] == this_idx]
    sample_gemini = gemini_results_df[gemini_results_df['sample_idx'] == this_idx]

    print(f"\n--- Sample ID: {this_idx} ---")

    print("\nGround Truth Mask (Gold Standard):")
    display(highlight_spans_in_text(source_text, sample_truth, highlight_color='#ffcccb')) # Light Red

    print("\nDLP Findings (Predictions):")
    display(highlight_spans_in_text(source_text, sample_dlp, highlight_color='#c8e6c9')) # Light Green

    print("\nGemini Findings (Predictions):")
    display(highlight_spans_in_text(source_text, sample_gemini, highlight_color='#c8e6c9')) # Light Green
