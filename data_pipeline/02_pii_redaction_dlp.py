# %%

from pii_redaction.data_loader import load_pii_dataset
import pandas as pd
from pii_redaction.paths import ProjPaths
from pii_redaction.detection import inspect_text_with_dlp
from tqdm import tqdm

# %% load data

dataset = load_pii_dataset()
dataset_ids = pd.Series(dataset['id'], name='id')

sampled_df = pd.read_csv(ProjPaths.data_path / 'pii_masking_300k_samples.csv')

# %% compute dlp findings

all_dlp_findings = []

print(f"Processing {len(sampled_df)} random samples...")
for this_id in tqdm(sampled_df['id'], desc="Running DLP on samples"):

    idx = dataset_ids[dataset_ids == this_id].index[0]

    entry = dataset[int(idx)]
    source_text = entry['source_text']

    # Get DLP findings
    dlp_findings = inspect_text_with_dlp(source_text)
    if not dlp_findings.empty:
        dlp_findings['sample_idx'] = idx
        dlp_findings['id'] = this_id
        all_dlp_findings.append(dlp_findings)

dlp_results_df = pd.concat(all_dlp_findings, ignore_index=True) if all_dlp_findings else pd.DataFrame()

# %%

dlp_results_df.to_csv(ProjPaths.data_path / 'pii_masking_300k_dlp_results.csv', index=False)

# Note: took 40 minutes to run the 1200 samples.