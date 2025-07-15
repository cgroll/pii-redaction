# %%

from pii_redaction.data_loader import load_pii_dataset
import pandas as pd
from pii_redaction.paths import ProjPaths
from pii_redaction.detection import detect_pii_with_gemma_api, detect_pii_with_ollama, find_fuzzy_matches
from tqdm import tqdm
import time
from pii_redaction import config
from pathvalidate import sanitize_filename

# %% load data

dataset = load_pii_dataset()
dataset_ids = pd.Series(dataset['id'], name='id')

sampled_df = pd.read_csv(ProjPaths.data_path / 'pii_masking_300k_samples.csv')

ollama_model_list = [
                     #'gemma3:12b-it-qat',
                     #'gemma3n:e4b',
                     #'gemma3:4b-it-qat',
                     'olmo2:13b',
                     'gemma3:27b-it-qat',
                     # 'smollm3:3b,'
                     #'llama3.2:3b',
                     #'llama3.1:8b',
                     ]


for this_model in ollama_model_list:
    print(this_model)
    
    safe_name_stem = sanitize_filename(this_model)

    all_gemma_findings = []
    all_unmatched = []

    print(f"Processing {len(sampled_df)} random samples...")
    for this_id in tqdm(sampled_df['id'], desc=f"Running {this_model} on samples"):
        idx = dataset_ids[dataset_ids == this_id].index[0]

        entry = dataset[int(idx)]
        source_text = entry['source_text']

        gemma_findings = detect_pii_with_ollama(source_text, model=this_model)

        # Create DataFrame with all Gemini findings
        findings_df = pd.DataFrame([{
            'value': finding['value'],
            'type': finding['type'],
            'is_valid_type': finding['is_valid_type']
        } for finding in gemma_findings])
        findings_df.drop_duplicates(inplace=True)

        try:
            matches, unmatched = find_fuzzy_matches(findings_df, source_text)
            matches['sample_idx'] = idx
            all_gemma_findings.append(matches)
            all_unmatched.append(unmatched)
        except Exception as e:
            print(f"An error occurred: {e}")

    gemma_results_df = pd.concat(all_gemma_findings, ignore_index=True) if all_gemma_findings else pd.DataFrame()
    unmatched_df = pd.concat(all_unmatched, ignore_index=True) if all_unmatched else pd.DataFrame()

    gemma_results_df.to_csv(ProjPaths.data_path / f'pii_masking_300k_ollama_{safe_name_stem}_results.csv', index=False)
    unmatched_df.to_csv(ProjPaths.data_path / f'pii_masking_300k_ollama_{safe_name_stem}_unmatched.csv', index=False)

# %%
