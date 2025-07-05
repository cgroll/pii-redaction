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
        if end >= len(source_text):
            end = len(source_text) # TODO: why does this happen?
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

# %% Gemini

MODEL_ID="gemini-2.5-flash-lite-preview-06-17"
PROJECT_ID="firm-dimension-461208-d1"

from google import genai
from google.genai import types
client = genai.Client(
vertexai=True, project=PROJECT_ID, location="global",
)
# If your image is stored in Google Cloud Storage, you can use the from_uri class method to create a Part object.
model = MODEL_ID
response = client.models.generate_content(
model=model,
contents=[
  "Tell me a joke about ducks"
],
)
print(response.text, end="")


# %%

from google import genai
from pydantic import BaseModel

class PIIdata(BaseModel):
    value: str
    type: str

class PIIList(BaseModel):
    pii_items: list[PIIdata]

client = genai.Client(
vertexai=True, project=PROJECT_ID, location="global",
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Find all PII in this text and list each piece with its type: " + this_text,
    config={
        "response_mime_type": "application/json",
        "response_schema": PIIList,
    },
)
# Use the response as a JSON string.
print(response.text)

# Use instantiated objects to show PII found
pii_list: PIIList = response.parsed

# %%

pii_list.pii_items

# %%

from difflib import SequenceMatcher
import pandas as pd

def find_substring_indices(text: str, substring: str, threshold: float = 0.85) -> list[tuple[int, int]]:
    """
    Find all occurrences of a substring in text using fuzzy matching.
    Returns list of (start, end) index tuples.
    
    Args:
        text: Text to search in
        substring: Substring to find
        threshold: Minimum similarity ratio to consider a match (0-1)
        
    Returns:
        List of (start, end) index tuples for matches
    """
    matches = []
    text_len = len(text)
    sub_len = len(substring)
    
    # Slide window of substring length across text
    for i in range(text_len - sub_len + 1):
        window = text[i:i + sub_len]
        similarity = SequenceMatcher(None, window.lower(), substring.lower()).ratio()
        if similarity >= threshold:
            matches.append((i, i + sub_len))
            
    return matches

# Find indices for each PII item
pii_with_indices = []
for item in pii_list.pii_items:
    matches = find_substring_indices(this_text, item.value)
    if matches:
        # Take the first match if multiple found
        start, end = matches[0]
        pii_with_indices.append({
            'value': item.value,
            'type': item.type,
            'start': start,
            'end': end
        })
    else:
        print(f"Could not find match for PII value: {item.value}")

# Create DataFrame with results        
pii_indices_df = pd.DataFrame(pii_with_indices)
print("\nPII items with their locations in text:")
print(pii_indices_df)

# %%

from IPython.display import HTML

def highlight_pii_in_text(text: str, pii_df: pd.DataFrame, highlight_color: str = '#ffcdd2') -> HTML:
    """
    Display text with PII locations highlighted
    
    Args:
        text: Text to display
        pii_df: DataFrame containing PII entries with start/end columns
        highlight_color: Color to use for highlighting PII regions (default: light red)
        
    Returns:
        IPython HTML object with highlighted text
    """
    from IPython.display import HTML
    import html
    
    # Create array marking which characters contain PII
    mask_array = [False] * len(text)
    for _, pii in pii_df.iterrows():
        start = int(pii['start'])
        end = int(pii['end'])
        if end >= len(text):
            end = len(text)
        for i in range(start, end):
            mask_array[i] = True
    
    # Build HTML with highlighted regions
    html_text = ""
    in_pii_region = False
    
    for i, char in enumerate(text):
        if mask_array[i] and not in_pii_region:
            # Start new PII region
            html_text += f'<span style="background-color: {highlight_color}">'
            in_pii_region = True
        elif not mask_array[i] and in_pii_region:
            # End PII region
            html_text += '</span>'
            in_pii_region = False
        html_text += html.escape(char)
    
    if in_pii_region:
        html_text += '</span>'
    
    return HTML(f'<pre style="white-space: pre-wrap; word-wrap: break-word;">{html_text}</pre>')

# Display text with PII highlighted
display(highlight_pii_in_text(this_text, pii_indices_df))



# %% gemma3 API

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="featherless-ai",
    api_key=HF_TOKEN,
)


def call_llm(text: str) -> str:
    """
    Call LLM with a single text message and return the response.
    
    Args:
        text: The text message to send to the LLM
        
    Returns:
        The LLM's response text
    """
    completion = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=[
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ],
    )
    return completion.choices[0].message

# %%

response = call_llm("Tell me a joke about ducks")
print(response.content)

# %%

import json
import os
import json
from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field
from typing import List

class DetectedPII(BaseModel):
    """A single detected piece of Personally Identifiable Information (PII)."""
    value: str = Field(..., description="The detected PII value.")
    type: str = Field(..., description="The type of PII.", enum=["name", "email", "telephone"])

class PIIList(BaseModel):
    """A list of all PII entities found in the text."""
    pii: List[DetectedPII]

def get_pii_from_text(text: str) -> PIIList:
    """
    Extract PII entities from input text using LLM.
    
    Args:
        text: Input text to analyze for PII
        
    Returns:
        PIIList containing detected PII entities
    """
    # Build the tool schema from Pydantic model
    pii_tool = {
        "type": "function",
        "function": {
            "name": "pii_redactor",
            "description": "Extracts PII entities from the text based on the provided schema.",
            "parameters": PIIList.model_json_schema()
        },
    }

    # Call LLM API with tool schema
    completion = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=[
            {"role": "system", "content": "You are a PII detection assistant. Use the pii_redactor tool to extract all entities"},
            {"role": "user", "content": text},
        ],
        tools=[pii_tool],
        tool_choice={"type": "function", "function": {"name": "pii_redactor"}},
    )

    # Parse and validate response
    message = completion.choices[0].message
    if message.tool_calls:
        tool_arguments_string = message.tool_calls[0].function.arguments
        return PIIList.model_validate_json(tool_arguments_string)
    
    return message


# %%

this_ind = 3234
this_text = dataset['train'][this_ind]['source_text']

pii_list = get_pii_from_text(this_text)

print(this_text)
print(pii_list)

# %%

text_to_analyze = "You can reach out to Jane Doe at jane.doe@example.com or by phone at 123-456-7890."

xx = get_pii_from_text(text_to_analyze)
xx

# %%

import os
import json
from huggingface_hub import InferenceClient

# The text we want to analyze for PII
text_to_analyze = this_text

# 1. A detailed prompt instructing the model to generate JSON
prompt = """
You are an expert PII (Personally Identifiable Information) detection tool.
Your task is to analyze the user's text and identify all instances of names, emails, and telephone numbers.

Your output must be a JSON object with a single key "pii".
The value of "pii" should be a list of objects, where each object has two fields:
1. "value": The detected PII string.
2. "type": The type of PII, which must be one of the following strings: "name", "email", or "telephone".

If no PII is found, return an empty list: {"pii": []}.
"""

# 2. Call the client with the new prompt and response_format
completion = client.chat.completions.create(
    model="google/gemma-3-27b-it",
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": text_to_analyze},
    ],
    response_format={"type": "json_object"}, # 3. Request JSON output
)

# 4. Parse the JSON string from the response
message_content = completion.choices[0].message.content
structured_output = json.loads(message_content)

# Print the pretty-printed structured output
print(json.dumps(structured_output, indent=2))

# %%

completion.choices[0].message

# %%



# %%

structured_output


# %%

pii_list

# %%


# %% gemma3 

import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

# Set your Google API key as an environment variable
# os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

# 1. Define the desired structured output
class DetectedPII(BaseModel):
    """A single detected piece of Personally Identifiable Information (PII)."""
    value: str = Field(..., description="The detected PII value.")
    type: str = Field(..., description="The type of PII.",
                    enum=["name", "email", "telephone"])

class PIIList(BaseModel):
    """A list of detected PII."""
    pii: List[DetectedPII]

# 2. Create the language model instance with structured output
# Make sure to use a model that supports structured output, like "gemma-3"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
structured_llm = llm.with_structured_output(PIIList)

# 3. Formulate a system prompt for PII detection
system_prompt = """
You are an expert in PII (Personally Identifiable Information) detection.
Your task is to analyze the given text and extract any PII.
The only allowed PII types are: 'name', 'email', and 'telephone'.
Return a list of all detected PII, each with its value and type.
If no PII is found, return an empty list.
"""

# 4. Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{text_to_analyze}"),
])

# 5. Chain the prompt and the structured LLM
chain = prompt | structured_llm

# 6. Run the chain with some example text
text_with_pii = "Contact John Doe at john.doe@email.com or call him at 123-456-7890. Jane's number is (987) 654-3210."
response = chain.invoke({"text_to_analyze": text_with_pii})

# Print the structured output
for pii_item in response.pii:
    print(f"Value: {pii_item.value}, Type: {pii_item.type}")

print("-" * 20)

# Example with no PII
text_without_pii = "This is a sample sentence without any personal information."
response_no_pii = chain.invoke({"text_to_analyze": text_without_pii})

if not response_no_pii.pii:
    print("No PII detected.")
else:
    for pii_item in response_no_pii.pii:
        print(f"Value: {pii_item.value}, Type: {pii_item.type}")

# %%

print(this_entry['source_text'])

# %% run DLP on random samples

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
    findings['text_length'] = len(source_text)
    
    # Append to results
    all_findings.append(findings)

# Combine all findings into single DataFrame
combined_findings_df = pd.concat(all_findings, ignore_index=True)

# Display results
print(f"Found {len(combined_findings_df)} findings across {n_samples} random samples")
combined_findings_df.head()

# %%

# check text length < end index
combined_findings_df[combined_findings_df['text_length'] < combined_findings_df['end']]

# %% get privacy masks for all samples from annotated ground truth

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
for idx in random_indices[:4]:
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

# %% compute quality metrics

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