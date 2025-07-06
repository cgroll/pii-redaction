# %%

from pii_redaction.data_loader import load_pii_dataset
import pandas as pd
from pii_redaction.paths import ProjPaths

# %% load data

dataset = load_pii_dataset()

# %% extract relevant columns / metrics

source_texts = pd.Series(dataset['source_text'], name='source_text')
text_lengths = source_texts.str.len()

ids = dataset['id']
languages = dataset['language']

data_df = pd.DataFrame([ids, text_lengths, languages]).T
data_df.columns = ['id', 'text_length', 'language']

# %% sample data

n_samples_per_language = 200

sampled_df = data_df.groupby('language').apply(
    lambda x: x.sample(n=min(len(x), n_samples_per_language), random_state=42)
).reset_index(drop=True)

# %% write to csv

sampled_df.to_csv(ProjPaths.data_path / 'pii_masking_300k_samples.csv', index=False)
