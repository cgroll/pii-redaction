# %%
"""
Development script for testing LLM-based PII detection.
This script is designed to be run cell-by-cell in an IDE like VS Code.
"""

# --- Imports ---
import pandas as pd
import numpy as np
from tqdm import tqdm
from IPython.display import display

from pii_redaction.data_loader import load_pii_dataset
from pii_redaction.detection import detect_pii_with_gemini, find_fuzzy_matches
from pii_redaction.evaluation import (
    highlight_spans_in_text,
    calculate_char_level_metrics
)

# %%

# 136984: 
# - date of birth recognized altough cut off: 
# - "Data di nascita: 198"

# %%
# --- 1. Load Data ---
print("Loading dataset...")
dataset = load_pii_dataset()
print(f"Dataset loaded with {len(dataset)} entries.")

# %%
# --- 2. Sample Data and Run Gemini Detection ---
N_SAMPLES = 1  # Using small number since LLM calls are slower
random_indices = np.random.choice(len(dataset), size=N_SAMPLES, replace=False)

all_gemini_findings = []
all_ground_truths = []
all_unmatched = []

print(f"Processing {N_SAMPLES} random samples...")
for idx in tqdm(random_indices, desc="Running Gemini on samples"):
    entry = dataset[int(idx)]
    source_text = entry['source_text']
    
    # Get Gemini findings and convert to spans
    gemini_findings = detect_pii_with_gemini(source_text)

    # Create DataFrame with all Gemini findings
    findings_df = pd.DataFrame([{
        'value': finding['value'],
        'type': finding['type']
    } for finding in gemini_findings])

    matches, unmatched = find_fuzzy_matches(findings_df, source_text)
    matches['sample_idx'] = idx
    all_gemini_findings.append(matches)
    all_unmatched.append(unmatched)

    # Get ground truth annotations
    ground_truth = pd.DataFrame(entry['privacy_mask'])
    if not ground_truth.empty:
        ground_truth['sample_idx'] = idx
        all_ground_truths.append(ground_truth)

gemini_results_df = pd.concat(all_gemini_findings, ignore_index=True) if all_gemini_findings else pd.DataFrame()
ground_truth_df = pd.concat(all_ground_truths, ignore_index=True) if all_ground_truths else pd.DataFrame()
unmatched_df = pd.concat(all_unmatched, ignore_index=True) if all_unmatched else pd.DataFrame()

print(f"\nFound {len(gemini_results_df)} total Gemini findings across {N_SAMPLES} samples.")
display(gemini_results_df.head())

# %%

display(highlight_spans_in_text(source_text, gemini_results_df, highlight_color='#ffcccb'))

all_unmatched

# %%

display(highlight_spans_in_text(source_text, ground_truth_df, highlight_color='#ffcccb'))
ground_truth_df

# %%
# --- 3. Calculate Metrics for Each Sample ---
metrics_list = []

for idx in tqdm(random_indices, desc="Calculating metrics"):
    entry = dataset[int(idx)]
    text_length = len(entry['source_text'])
    
    sample_truth = ground_truth_df[ground_truth_df['sample_idx'] == idx]
    sample_preds = gemini_results_df[gemini_results_df['sample_idx'] == idx]
    
    if not sample_truth.empty:
        metrics = calculate_char_level_metrics(sample_truth, sample_preds, text_length)
        metrics['sample_idx'] = idx
        metrics['text_length'] = text_length
        metrics['truth_masked_chars'] = sample_truth.apply(lambda row: row['end'] - row['start'], axis=1).sum()
        metrics['gemini_masked_chars'] = sample_preds.apply(lambda row: row['end'] - row['start'], axis=1).sum()
        metrics_list.append(metrics)

metrics_df = pd.DataFrame(metrics_list)
print("\nMetrics Summary (Gemini vs. Ground Truth):")
display(metrics_df.describe())

# %%
# --- 4. Visualize Results ---
print("\nVisualizing results for all samples...")

for idx in random_indices:
    source_text = dataset[int(idx)]['source_text']
    sample_truth = ground_truth_df[ground_truth_df['sample_idx'] == idx]
    sample_gemini = gemini_results_df[gemini_results_df['sample_idx'] == idx]
    
    print(f"\n--- Sample ID: {idx} ---")
    
    print("\nGround Truth Mask (Gold Standard):")
    display(highlight_spans_in_text(source_text, sample_truth, highlight_color='#ffcccb'))  # Light Red
    
    print("\nGemini Findings (Predictions):")
    display(highlight_spans_in_text(source_text, sample_gemini, highlight_color='#c8e6c9'))  # Light Green
