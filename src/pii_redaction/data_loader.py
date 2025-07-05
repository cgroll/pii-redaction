# %%
"""
Module for loading the PII masking dataset.
"""

from datasets import load_dataset, Dataset

def load_pii_dataset() -> Dataset:
    """
    Loads the 'ai4privacy/pii-masking-300k' dataset.

    Returns:
        Dataset: The training split of the dataset.
    """
    try:
        dataset = load_dataset("ai4privacy/pii-masking-300k")
        return dataset['train']
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None