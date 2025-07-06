# %%

from pii_redaction.data_loader import load_pii_dataset
import pandas as pd
from pii_redaction.paths import ProjPaths
from pii_redaction.detection import detect_pii_with_gemini, find_fuzzy_matches
from tqdm import tqdm
import time
from pii_redaction import config

# %% load data

dataset = load_pii_dataset()
dataset_ids = pd.Series(dataset['id'], name='id')

sampled_df = pd.read_csv(ProjPaths.data_path / 'pii_masking_300k_samples.csv')

# %% compute gemini findings

all_gemini_findings = []
all_unmatched = []
all_input_tokens = []
all_output_tokens = []
all_latency = []

print(f"Processing {len(sampled_df)} random samples...")
for this_id in tqdm(sampled_df['id'], desc="Running Gemini on samples"):
    start_time = time.time()

    idx = dataset_ids[dataset_ids == this_id].index[0]

    entry = dataset[int(idx)]
    source_text = entry['source_text']
    # Get Gemini findings and convert to spans
    gemini_findings, usage_metadata = detect_pii_with_gemini(source_text)
    all_input_tokens.append(usage_metadata['prompt_token_count'])
    all_output_tokens.append(usage_metadata['candidates_token_count'])

    # Create DataFrame with all Gemini findings
    findings_df = pd.DataFrame([{
        'value': finding['value'],
        'type': finding['type']
    } for finding in gemini_findings])
    findings_df.drop_duplicates(inplace=True)

    matches, unmatched = find_fuzzy_matches(findings_df, source_text)
    matches['sample_idx'] = idx
    all_gemini_findings.append(matches)
    all_unmatched.append(unmatched)

    end_time = time.time()
    all_latency.append(end_time - start_time)

gemini_results_df = pd.concat(all_gemini_findings, ignore_index=True) if all_gemini_findings else pd.DataFrame()
unmatched_df = pd.concat(all_unmatched, ignore_index=True) if all_unmatched else pd.DataFrame()

# %%

gemini_costs_df = pd.DataFrame({
    'input_tokens': all_input_tokens,
    'output_tokens': all_output_tokens
})
gemini_costs_df['input_cost'] = config.GEMINI_INPUT_TOKEN_COST * gemini_costs_df['input_tokens'] / 1_000_000
gemini_costs_df['output_cost'] = config.GEMINI_OUTPUT_TOKEN_COST * gemini_costs_df['output_tokens'] / 1_000_000 * 10
gemini_costs_df['total_cost'] = gemini_costs_df['input_cost'] + gemini_costs_df['output_cost']
gemini_costs_df['latency'] = all_latency

# %%

gemini_results_df.to_csv(ProjPaths.data_path / 'pii_masking_300k_gemini_results.csv', index=False)
gemini_costs_df.to_csv(ProjPaths.data_path / 'pii_masking_300k_gemini_costs.csv', index=False)
unmatched_df.to_csv(ProjPaths.data_path / 'pii_masking_300k_gemini_unmatched.csv', index=False)
