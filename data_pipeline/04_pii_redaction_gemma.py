# %%

from pii_redaction.data_loader import load_pii_dataset
import pandas as pd
from pii_redaction.paths import ProjPaths
from pii_redaction.detection import detect_pii_with_gemma, detect_pii_with_ollama_gemma3_1b, find_fuzzy_matches
from tqdm import tqdm
import time
from pii_redaction import config

# %% load data

dataset = load_pii_dataset()
dataset_ids = pd.Series(dataset['id'], name='id')

sampled_df = pd.read_csv(ProjPaths.data_path / 'pii_masking_300k_samples.csv')

# %% compute gemini findings

all_gemma_findings = []
all_unmatched = []

print(f"Processing {len(sampled_df)} random samples...")
for this_id in tqdm(sampled_df['id'], desc="Running Gemma on samples"):
    idx = dataset_ids[dataset_ids == this_id].index[0]

    entry = dataset[int(idx)]
    source_text = entry['source_text']

    gemma_findings = detect_pii_with_ollama_gemma3_1b(source_text)

    # Create DataFrame with all Gemini findings
    findings_df = pd.DataFrame([{
        'value': finding['value'],
        'type': finding['type'],
        'is_valid_type': finding['is_valid_type']
    } for finding in gemma_findings])
    findings_df.drop_duplicates(inplace=True)

    matches, unmatched = find_fuzzy_matches(findings_df, source_text)
    matches['sample_idx'] = idx
    all_gemma_findings.append(matches)
    all_unmatched.append(unmatched)

gemma_results_df = pd.concat(all_gemma_findings, ignore_index=True) if all_gemma_findings else pd.DataFrame()
unmatched_df = pd.concat(all_unmatched, ignore_index=True) if all_unmatched else pd.DataFrame()

# %%

gemma_results_df.to_csv(ProjPaths.data_path / 'pii_masking_300k_ollama_gemma3_1b_results.csv', index=False)
unmatched_df.to_csv(ProjPaths.data_path / 'pii_masking_300k_ollama_gemma3_1b_unmatched.csv', index=False)

# %%
