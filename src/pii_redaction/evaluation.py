# %%
"""
Functions for evaluating, visualizing, and processing PII detection results.
"""

import pandas as pd
import difflib
from IPython.display import HTML
from difflib import SequenceMatcher
from typing import List, Tuple
import html
import numpy as np

# --- Visualization ---
def highlight_spans_in_text(text: str, spans_df: pd.DataFrame, highlight_color: str = '#ffcccb') -> HTML:
    """
    Displays text with specified character spans highlighted in HTML.

    Args:
        text: The original text.
        spans_df: DataFrame with 'start' and 'end' columns for highlighting.
        highlight_color: The HTML color for the highlight.

    Returns:
        An IPython HTML object with the highlighted text.
    """
    mask_array = [False] * len(text)
    for _, span in spans_df.iterrows():
        start, end = int(span['start']), int(span['end'])
        # Clamp end index to prevent out-of-bounds errors
        end = min(end, len(text))
        for i in range(start, end):
            mask_array[i] = True

    html_parts = []
    in_highlight = False
    for i, char in enumerate(text):
        if mask_array[i] and not in_highlight:
            html_parts.append(f'<span style="background-color: {highlight_color}">')
            in_highlight = True
        elif not mask_array[i] and in_highlight:
            html_parts.append('</span>')
            in_highlight = False
        html_parts.append(html.escape(char))

    if in_highlight:
        html_parts.append('</span>')

    return HTML(f'<pre style="white-space: pre-wrap; word-wrap: break-word;">{"".join(html_parts)}</pre>')

# --- Metrics ---
def calculate_char_level_metrics(
    ground_truth_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    text_length: int
) -> dict:
    """
    Calculates character-level precision, recall, F1, and accuracy.

    Args:
        ground_truth_df: DataFrame of true PII spans with 'start' and 'end'.
        predictions_df: DataFrame of predicted PII spans with 'start' and 'end'.
        text_length: The total length of the source text.

    Returns:
        A dictionary containing the calculated metrics.
    """
    truth_mask = [0] * text_length
    pred_mask = [0] * text_length

    for _, row in ground_truth_df.iterrows():
        for i in range(int(row['start']), min(int(row['end']), text_length)):
            truth_mask[i] = 1

    for _, row in predictions_df.iterrows():
        for i in range(int(row['start']), min(int(row['end']), text_length)):
            pred_mask[i] = 1

    tp = sum(1 for t, p in zip(truth_mask, pred_mask) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(truth_mask, pred_mask) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(truth_mask, pred_mask) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(truth_mask, pred_mask) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan
    accuracy = (tp + tn) / text_length if text_length > 0 else np.nan

    return {
        'precision': precision, 'recall': recall, 'f1_score': f1,
        'accuracy': accuracy, 'true_positives': tp, 'false_positives': fp,
        'false_negatives': fn, 'true_negatives': tn,
        'n_ground_truth_positives': truth_mask.count(1),
        'n_predicted_positives': pred_mask.count(1),
        'n_ground_truth_negatives': truth_mask.count(0),
        'n_predicted_negatives': pred_mask.count(0),
    }