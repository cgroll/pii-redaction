# %%

from datasets import load_dataset
dataset = load_dataset("ai4privacy/pii-masking-300k")
# https://huggingface.co/datasets/ai4privacy/pii-masking-300k

list_of_relevant_types = [
    'USERNAME',
    'IDCARD',
    'SOCIALNUMBER',
    'EMAIL',
    'PASSPORT',
    'DRIVERLICENSE',
    'BOD',
    'LASTNAME1',
    'IP',
    'GIVENNAME1',
    'TEL',
    'STREET',
    'PASS',
    'SECADDRESS',
    'LASTNAME2',
    'GIVENNAME2',
    'GEOCOORD',
    'LASTNAME3'
]

# %%

import difflib
from IPython.display import HTML


def create_html_diff(source_text, masked_source_text):
    diff = difflib.HtmlDiff(wrapcolumn=70).make_file(source_text.splitlines(),
                                                     masked_source_text.splitlines())
    return HTML(diff)

def create_flexible_diff_visualization(text1, text2):
    # Create HTML wrapper with side-by-side layout
    html = ['<div style="font-family: monospace; white-space: pre;">']
    html.append('<table style="width: 100%;"><tr><td style="width: 50%; vertical-align: top; padding: 8px;">')
    html.append('<div style="border: 1px solid #ccc; padding: 8px;"><div>Original text:</div>')
    
    # Helper function to encode special characters
    def html_encode(s):
        return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Split texts into lines
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    
    # Process original text
    for i in range(max(len(lines1), len(lines2))):
        line1 = lines1[i] if i < len(lines1) else ""
        line2 = lines2[i] if i < len(lines2) else ""
        
        if line1 == line2:
            html.append(html_encode(line1))
        else:
            matcher = difflib.SequenceMatcher(None, line1, line2)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    html.append(html_encode(line1[i1:i2]))
                elif tag in ('delete', 'replace'):
                    html.append(f'<span style="background-color: #ffcdd2;">{html_encode(line1[i1:i2])}</span>')
        html.append('<br>')
    
    # Close original text column and start modified text column
    html.append('</div></td><td style="width: 50%; vertical-align: top; padding: 8px;">')
    html.append('<div style="border: 1px solid #ccc; padding: 8px;"><div>Modified text:</div>')
    
    # Process modified text
    for i in range(max(len(lines1), len(lines2))):
        line1 = lines1[i] if i < len(lines1) else ""
        line2 = lines2[i] if i < len(lines2) else ""
        
        if line1 == line2:
            html.append(html_encode(line2))
        else:
            matcher = difflib.SequenceMatcher(None, line1, line2)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    html.append(html_encode(line2[j1:j2]))
                elif tag in ('insert', 'replace'):
                    html.append(f'<span style="background-color: #c8e6c9;">{html_encode(line2[j1:j2])}</span>')
        html.append('<br>')
    
    # Close table structure
    html.append('</div></td></tr></table>')
    html.append('</div>')
    return HTML(''.join(html))

def create_mask_visualization(original, masked):
    # Ensure strings are same length
    assert len(original) == len(masked)
    
    # Create HTML with colored highlighting
    html = ['<div style="font-family: monospace; white-space: pre;">']
    
    in_mask = False
    for i, (orig_char, mask_char) in enumerate(zip(original, masked)):
        if mask_char == '*':
            if not in_mask:
                # Start of masked section
                html.append('<span style="background-color: #ffcdd2;">')
                in_mask = True
        elif in_mask:
            # End of masked section
            html.append('</span>')
            in_mask = False
            
        html.append(orig_char)
        
    if in_mask:
        html.append('</span>')
    html.append('</div>')
    
    return HTML(''.join(html))

def display_text_with_privacy_masks(source_text, privacy_mask_df, highlight_color='#ffcccb'):
    """
    Display text with privacy masks highlighted
    
    Args:
        source_text (str): Original text
        privacy_mask_df (pd.DataFrame): DataFrame containing privacy mask entries with start/end columns
        highlight_color (str): Color to use for highlighting masked regions (default: light red)
    """
    from IPython.display import HTML
    import html
    
    # Create array marking which characters should be masked
    mask_array = [False] * len(source_text)
    for _, mask in privacy_mask_df.iterrows():
        start = int(mask['start'])
        end = int(mask['end'])
        for i in range(start, end):
            mask_array[i] = True
    
    # Build HTML by finding consecutive masked regions
    html_text = ""
    in_masked_region = False
    
    for i, char in enumerate(source_text):
        if mask_array[i] and not in_masked_region:
            # Start new masked region
            html_text += f'<span style="background-color: {highlight_color}">'
            in_masked_region = True
        elif not mask_array[i] and in_masked_region:
            # End masked region
            html_text += '</span>'
            in_masked_region = False
        # Escape HTML characters to display text verbatim
        html_text += html.escape(char)
    
    # Close final span if needed
    if in_masked_region:
        html_text += '</span>'
    
    # Display the HTML in a pre tag to preserve formatting
    return HTML(f'<pre style="white-space: pre-wrap; word-wrap: break-word;">{html_text}</pre>')

# %%

len(dataset['train'])

this_ind = 99923

this_entry = dataset['train'][this_ind]

source_text = this_entry['source_text']
target_text = this_entry['target_text']
masked_source_text = source_text

for this_mask in this_entry['privacy_mask']:
    this_start = this_mask['start']
    this_end = this_mask['end']
    replace_str = '*' * (this_end - this_start)
    masked_source_text = masked_source_text[:this_start] + replace_str + masked_source_text[this_end:]

create_html_diff(source_text, masked_source_text)
# create_mask_visualization(source_text, masked_source_text)
create_flexible_diff_visualization(source_text, target_text)

# %%

def create_flexible_diff_visualization(text1, text2):
    # Use difflib's HtmlDiff to create the side-by-side comparison
    diff = difflib.HtmlDiff(wrapcolumn=70, tabsize=4).make_file(
        text1.splitlines(),
        text2.splitlines(),
        fromdesc="Original text",
        todesc="Modified text"
    )
    
    # Add custom styling to make it more readable
    styled_diff = diff.replace(
        '<style type="text/css">',
        '''<style type="text/css">
        table.diff {font-family: monospace; width: 100%;}
        td {padding: 8px; vertical-align: top;}
        .diff_header {display: none;}
        td.diff_header {display: none;}
        table.diff td {border: 1px solid #ccc;}
        .diff_next {display: none;}
        '''
    )
    
    return HTML(styled_diff)

# %%

target_text.splitlines()

# %%

import google.cloud.dlp_v2

def call_dlp_api(project_id: str, content_string: str):
    """
    Makes a call to the Google Cloud DLP API to detect PII in a string.

    Args:
        project_id: The ID of your Google Cloud project.
        content_string: The string to inspect for PII.
    
    Returns:
        The raw DLP API response.
    """
    # Instantiate a DLP client
    dlp_client = google.cloud.dlp_v2.DlpServiceClient()

    # Configure the types of PII to detect
    info_types_to_detect = [
        {"name": "PHONE_NUMBER"},
        {"name": "EMAIL_ADDRESS"},
        {"name": "CREDIT_CARD_NUMBER"},
        {"name": "US_SOCIAL_SECURITY_NUMBER"},
        {"name": "PERSON_NAME"},
        {"name": "FEMALE_NAME"},
        {"name": "MALE_NAME"},
        {"name": "FIRST_NAME"},
        {"name": "LAST_NAME"},
        {"name": "DATE_OF_BIRTH"},
        {"name": "LOCATION"},
        {"name": "STREET_ADDRESS"}
    ]

    # Prepare the request
    request = {
        "parent": f"projects/{project_id}/locations/global",
        "inspect_config": {
            "info_types": info_types_to_detect,
            "include_quote": True,
        },
        "item": {"value": content_string}
    }

    # Make the API call and return raw response
    return dlp_client.inspect_content(request=request)

# Replace 'your-gcp-project-id' with your actual project ID.
gcp_project_id = "firm-dimension-461208-d1"

this_ind = 99923
this_entry = dataset['train'][this_ind]

import pandas as pd

def get_dlp_findings_df(project_id: str, content_string: str) -> pd.DataFrame:
    """
    Calls DLP API and returns findings as a DataFrame.
    
    Args:
        project_id: The ID of your Google Cloud project
        content_string: The string to inspect for PII
        
    Returns:
        pd.DataFrame containing the DLP findings, with columns for quote, info_type,
        start position, end position and likelihood. Returns empty DataFrame if no findings.
    """
    response = call_dlp_api(project_id, content_string)
    
    findings_data = []
    if response.result.findings:
        for finding in response.result.findings:
            findings_data.append({
                'quote': finding.quote,
                'info_type': finding.info_type.name,
                'start': finding.location.byte_range.start,
                'end': finding.location.byte_range.end,
                'likelihood': finding.likelihood.value
            })
    
    return pd.DataFrame(findings_data)

# Call function with current data
findings_df = get_dlp_findings_df(gcp_project_id, this_entry['source_text'])

# %%

def create_privacy_mask_df(findings_list):
    """
    Creates a DataFrame from DLP findings to track privacy mask entries.
    
    Args:
        findings_list: List of DLP findings
        
    Returns:
        pd.DataFrame with columns: value, label, start, end
    """
    mask_entries = []
    for finding in findings_list:

        if finding['label'] in list_of_relevant_types:
            mask_entries.append({
                'value': finding['value'],
                'label': finding['label'],
                'start': finding['start'],
                'end': finding['end']
            })
    return pd.DataFrame(mask_entries)

# Create privacy mask DataFrame
privacy_mask_df = create_privacy_mask_df(this_entry['privacy_mask'])
privacy_mask_df.head()

# %%

findings_df

# %%

print(this_entry['source_text'])

# %%

import numpy as np
import tqdm

n_samples = 100

# Randomly sample entries from training set
random_indices = np.random.choice(len(dataset['train']), size=n_samples, replace=False)

# Initialize list to store results
all_findings = []

# Process each sampled entry
for idx in tqdm.tqdm(random_indices):
    entry = dataset['train'][int(idx)]
    source_text = entry['source_text']
    
    # Get DLP findings
    findings = get_dlp_findings_df(gcp_project_id, source_text)
    
    # Add index column to findings
    findings['sample_idx'] = idx
    
    # Append to results
    all_findings.append(findings)

# Combine all findings into single DataFrame
combined_findings_df = pd.concat(all_findings, ignore_index=True)

# Display results
print(f"Found {len(combined_findings_df)} findings across {n_samples} random samples")
combined_findings_df.head()

# %% get privacy masks

# Initialize list to store privacy masks
all_privacy_masks = []

# Process each sampled entry
for idx in random_indices:
    entry = dataset['train'][int(idx)]
    privacy_mask = create_privacy_mask_df(entry['privacy_mask'])
    privacy_mask['sample_idx'] = idx
    all_privacy_masks.append(privacy_mask)

# Combine all privacy masks into single DataFrame 
privacy_mask_df = pd.concat(all_privacy_masks, ignore_index=True)

# %% 
# Compare privacy masks and DLP findings for each sample
# Loop through all random samples
for idx in random_indices:
    # Get original text
    entry = dataset['train'][int(idx)]
    source_text = entry['source_text']

    # Get privacy mask entries for this sample
    sample_privacy_mask = privacy_mask_df[privacy_mask_df['sample_idx'] == idx]

    # Get DLP findings for this sample  
    sample_findings = combined_findings_df[combined_findings_df['sample_idx'] == idx]

    print(f"\nSample {idx}:")
    print("Ground truth masks:")
    print('-'*100)
    display(display_text_with_privacy_masks(source_text, sample_privacy_mask, highlight_color='#ffcccb'))
    print("DLP findings:")
    print('-'*100)
    display(display_text_with_privacy_masks(source_text, sample_findings, highlight_color='#c8e6c9'))

# %%

# Initialize lists to store metrics
metrics = []

# Process each sampled entry
for idx in random_indices:
    # Get original text
    entry = dataset['train'][int(idx)]
    source_text = entry['source_text']
    text_length = len(source_text)
    
    # Get privacy mask entries for this sample
    sample_privacy_mask = privacy_mask_df[privacy_mask_df['sample_idx'] == idx]
    
    # Get DLP findings for this sample
    sample_findings = combined_findings_df[combined_findings_df['sample_idx'] == idx]
    
    # Create full masks (1 for masked character, 0 for unmasked)
    truth_mask = [0] * text_length
    dlp_mask = [0] * text_length
    
    # Fill in truth mask
    for _, row in sample_privacy_mask.iterrows():
        start, end = int(row['start']), int(row['end'])
        if end >= text_length:
            end = text_length # TODO: why does this happen?
        for i in range(start, end):
            truth_mask[i-1] = 1
            
    # Fill in DLP mask
    for _, row in sample_findings.iterrows():
        start, end = int(row['start']), int(row['end'])
        if end >= text_length:
            end = text_length # TODO: why does this happen?
        for i in range(start, end):
            dlp_mask[i-1] = 1
    
    # Calculate metrics
    true_positives = sum(1 for t, d in zip(truth_mask, dlp_mask) if t == 1 and d == 1)
    true_negatives = sum(1 for t, d in zip(truth_mask, dlp_mask) if t == 0 and d == 0)
    false_positives = sum(1 for t, d in zip(truth_mask, dlp_mask) if t == 0 and d == 1)
    false_negatives = sum(1 for t, d in zip(truth_mask, dlp_mask) if t == 1 and d == 0)
    
    # Calculate precision, recall and accuracy
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    accuracy = (true_positives + true_negatives) / text_length
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics.append({
        'sample_idx': idx,
        'text_length': text_length,
        'truth_masked_chars': sum(truth_mask),
        'dlp_masked_chars': sum(dlp_mask),
        'overlapping_chars': true_positives,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1_score': f1_score
    })

# Create metrics DataFrame and display summary
metrics_df = pd.DataFrame(metrics)
print("\nMetrics Summary:")
print(metrics_df.describe())
print("\nDetailed metrics per sample:")
display(metrics_df)

# %%

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
sns.boxplot(y=metrics_df['accuracy'])
plt.title('Distribution of Accuracy Scores')
plt.ylabel('Accuracy')
plt.show()