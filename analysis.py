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

# load gemma results
gemma_results_df = pd.read_csv(ProjPaths.data_path / 'pii_masking_300k_gemma_results.csv')
unmatched_df_gemma = pd.read_csv(ProjPaths.data_path / 'pii_masking_300k_gemma_unmatched.csv')

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

# %% calculate metrics for detected PII results

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
gemma_metrics_df = calculate_metrics_for_samples(gemma_results_df, ground_truth_df, sampled_df, dataset)

# %%

dlp_metrics_df['f1_score'].mean()
gemini_metrics_df['f1_score'].mean()
gemma_metrics_df['f1_score'].mean()

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

plt.figure(figsize=(8, 6))
sns.scatterplot(data=gemma_metrics_df, x='recall', y='precision')
plt.title('Precision vs. Recall for Gemma')
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
gemma_metrics_long = gemma_metrics_df[['precision', 'recall', 'f1_score']].melt(var_name='metric', value_name='value') 
gemma_metrics_long['model'] = 'Gemma'
combined_metrics = pd.concat([dlp_metrics_long, gemini_metrics_long, gemma_metrics_long])

# Create boxplots
sns.boxplot(data=combined_metrics[(combined_metrics.metric == 'precision') & (~combined_metrics.value.isna())],
            x='model', y='value', ax=axes[0])
axes[0].set_title('Distribution of Precision Scores')
axes[0].set_ylabel('Precision')

sns.boxplot(data=combined_metrics[(combined_metrics.metric == 'recall') & (~combined_metrics.value.isna())],
            x='model', y='value', ax=axes[1])
axes[1].set_title('Distribution of Recall Scores')
axes[1].set_ylabel('Recall')

sns.boxplot(data=combined_metrics[(combined_metrics.metric == 'f1_score') & (~combined_metrics.value.isna())],
            x='model', y='value', ax=axes[2])
axes[2].set_title('Distribution of F1-Scores')
axes[2].set_ylabel('F1-Score')
plt.tight_layout()

# Save figure to disk before showing
ProjPaths.figures_path.mkdir(parents=True, exist_ok=True)
plt.savefig(ProjPaths.figures_path / 'metric_distributions_boxplots.png', dpi=300, bbox_inches='tight')

plt.show()


# %% metrics for dlp for each language

# Create subplots for F1 scores by language for each model
fig, axes = plt.subplots(3, 1, figsize=(12, 15))

# DLP plot
sns.boxplot(data=dlp_metrics_df, x='language', y='f1_score', ax=axes[0])
axes[0].set_title('Distribution of F1 Scores by Language (DLP)')
axes[0].set_xlabel('')  # Remove x label since it's not the bottom plot
axes[0].set_ylabel('F1 Score')
axes[0].tick_params(axis='x', rotation=45)

# Gemini plot
sns.boxplot(data=gemini_metrics_df, x='language', y='f1_score', ax=axes[1])
axes[1].set_title('Distribution of F1 Scores by Language (Gemini)') 
axes[1].set_xlabel('')  # Remove x label since it's not the bottom plot
axes[1].set_ylabel('F1 Score')
axes[1].tick_params(axis='x', rotation=45)

# Gemma plot
sns.boxplot(data=gemma_metrics_df, x='language', y='f1_score', ax=axes[2])
axes[2].set_title('Distribution of F1 Scores by Language (Gemma)')
axes[2].set_xlabel('Language')
axes[2].set_ylabel('F1 Score')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()

# Save figure to disk
ProjPaths.figures_path.mkdir(parents=True, exist_ok=True)
plt.savefig(ProjPaths.figures_path / 'f1_scores_by_language.png', dpi=300, bbox_inches='tight')

plt.show()
 

# %% DEV: started | analyse different PII types

print("Ground truth unique labels:\n", sorted(ground_truth_df['label'].unique()))

print("\nGemini unique types:\n", sorted(gemini_results_df['type'].unique()))

print("\nDLP unique labels:\n", sorted(dlp_results_df['label'].unique()))

print("\nGemma unique types:\n", sorted(gemma_results_df['type'].unique()))



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


# %% compare gemini and dlp

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
    sample_gemma = gemma_results_df[gemma_results_df['sample_idx'] == this_idx]

    print(f"\n--- Sample ID: {this_idx} ---")

    print("\nGround Truth Mask (Gold Standard):")
    display(highlight_spans_in_text(source_text, sample_truth, highlight_color='#ffcccb')) # Light Red

    print("\nDLP Findings (Predictions):")
    display(highlight_spans_in_text(source_text, sample_dlp, highlight_color='#c8e6c9')) # Light Green

    print("\nGemini Findings (Predictions):")
    display(highlight_spans_in_text(source_text, sample_gemini, highlight_color='#c8e6c9')) # Light Green

    print("\nGemma Findings (Predictions):")
    display(highlight_spans_in_text(source_text, sample_gemma, highlight_color='#c8e6c9')) # Light Green

# %%

n_counts_ground_truth = ground_truth_df['label'].value_counts()
n_counts_ground_truth

# %%

for this_label in n_counts_ground_truth.index[0:3]:
    print(f"{this_label}: {n_counts_ground_truth[this_label]}")

# %%

this_label = 'LASTNAME1'

this_label_data = ground_truth_df[ground_truth_df['label'] == this_label]
this_label_data

# %%

this_label_data.groupby('sample_idx')['label'].count().value_counts().plot(kind='bar')
plt.xlabel('Number of PIIs per sample')
plt.ylabel('Count')
plt.title(f'Distribution of {this_label} by Sample')
plt.show()

# %%

this_label_dlp_metrics_df = calculate_metrics_for_samples(dlp_results_df, this_label_data, sampled_df, dataset)
this_label_gemini_metrics_df = calculate_metrics_for_samples(gemini_results_df, this_label_data, sampled_df, dataset)
this_label_gemma_metrics_df = calculate_metrics_for_samples(gemma_results_df, this_label_data, sampled_df, dataset)

# %%

# Create a DataFrame with recall values from each model
recall_data = pd.DataFrame({
    'DLP': this_label_dlp_metrics_df['recall'],
    'Gemini': this_label_gemini_metrics_df['recall'],
    'Gemma': this_label_gemma_metrics_df['recall']
})

# Create boxplot
ax = recall_data.boxplot(column=['DLP', 'Gemini', 'Gemma'], 
                        figsize=(10, 6),
                        whis=1.5)

# Customize plot
plt.title(f'Recall Distribution by Model for {this_label}')
plt.ylabel('Recall')
plt.grid(True, linestyle='--', alpha=0.7)

# %%

this_idx = this_label_dlp_metrics_df.iloc[710]['sample_idx']
this_idx

# %%

source_text = dataset[int(this_idx)]['source_text']
sample_truth = this_label_data[this_label_data['sample_idx'] == this_idx]
sample_dlp = dlp_results_df[dlp_results_df['sample_idx'] == this_idx]
sample_gemini = gemini_results_df[gemini_results_df['sample_idx'] == this_idx]
sample_gemma = gemma_results_df[gemma_results_df['sample_idx'] == this_idx]

print(f"\n--- Sample ID: {this_idx} ---")

print("\nGround Truth Mask (Gold Standard):")
display(highlight_spans_in_text(source_text, sample_truth, highlight_color='#ffcccb')) # Light Red

print("\nDLP Findings (Predictions):")
display(highlight_spans_in_text(source_text, sample_dlp, highlight_color='#c8e6c9')) # Light Green

print("\nGemini Findings (Predictions):")
display(highlight_spans_in_text(source_text, sample_gemini, highlight_color='#c8e6c9')) # Light Green

print("\nGemma Findings (Predictions):")
display(highlight_spans_in_text(source_text, sample_gemma, highlight_color='#c8e6c9')) # Light Green


# %%

all_avg_recall_rates = []
label_names = []

for this_label in n_counts_ground_truth.index:

    this_label_data = ground_truth_df[ground_truth_df['label'] == this_label]
    n_samples = this_label_data['sample_idx'].nunique()
    print(f"{this_label}: {n_samples} samples")

    this_label_dlp_metrics_df = calculate_metrics_for_samples(dlp_results_df, this_label_data, sampled_df, dataset)
    this_label_gemini_metrics_df = calculate_metrics_for_samples(gemini_results_df, this_label_data, sampled_df, dataset)
    this_label_gemma_metrics_df = calculate_metrics_for_samples(gemma_results_df, this_label_data, sampled_df, dataset)

    recall_data = pd.DataFrame({
        'DLP': this_label_dlp_metrics_df['recall'],
        'Gemini': this_label_gemini_metrics_df['recall'],
        'Gemma': this_label_gemma_metrics_df['recall']
    })

    all_avg_recall_rates.append(recall_data.mean())
    label_names.append(this_label)

# %%

recall_data = pd.DataFrame(all_avg_recall_rates, index=label_names)

# Add count to index labels
recall_data.index = [f"{label}: {n_counts_ground_truth[label]}" for label in recall_data.index]

recall_data

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from pii_redaction.paths import ProjPaths

plt.figure(figsize=(10, 8))
sns.heatmap(recall_data * 100, annot=True, fmt='.0f', cmap='RdYlGn', 
            cbar_kws={'label': 'Recall Rate (%)'})
plt.title('Recall Rates by Model and Label Type')
plt.tight_layout()

# Save figure
plt.savefig(ProjPaths.figures_path / 'pii_recall_rates_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()


# %%


