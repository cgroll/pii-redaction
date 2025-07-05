# %%
"""
PII detection logic using Google Cloud DLP and LLM wrappers.
"""

import pandas as pd
import google.cloud.dlp_v2
from pii_redaction.llm import clients
from typing import List, Dict
import config
from pydantic import BaseModel
from enum import Enum
import regex
from config import FUZZY_MATCH_THRESHOLDS, FUZZY_MATCH_DEFAULT_EDITS

# --- Pydantic Schemas for Structured Output ---
class PIIType(str, Enum):
    """Valid PII types that can be detected."""
    # Dynamically create enum values from config.LLM_PII_TYPES
    locals().update({
        pii_type["name"]: pii_type["name"] 
        for pii_type in config.LLM_PII_TYPES
    })

class PIIEntity(BaseModel):
    """A single detected piece of Personally Identifiable Information (PII)."""
    value: str
    type: PIIType

class PIIList(BaseModel):
    """A list of all PII entities found in the text."""
    pii_items: List[PIIEntity]

# --- Google Cloud DLP ---
def inspect_text_with_dlp(text: str) -> pd.DataFrame:
    """
    Calls the Google Cloud DLP API to detect PII and returns findings as a DataFrame.

    Args:
        text: The string to inspect for PII.

    Returns:
        A DataFrame with DLP findings (quote, info_type, start, end, likelihood).
    """
    try:
        dlp_client = google.cloud.dlp_v2.DlpServiceClient()
        parent = f"projects/{config.GCP_PROJECT_ID}/locations/global"
        inspect_config = {
            "info_types": config.DLP_INFO_TYPES,
            "include_quote": True,
        }
        item = {"value": text}
        request = {"parent": parent, "inspect_config": inspect_config, "item": item}
        response = dlp_client.inspect_content(request=request)

        findings_data = [
            {
                'value': finding.quote,
                'label': finding.info_type.name,
                'start': finding.location.byte_range.start,
                'end': finding.location.byte_range.end,
                'likelihood': finding.likelihood.name,
            }
            for finding in response.result.findings
        ]
        return pd.DataFrame(findings_data)
    except Exception as e:
        print(f"Error calling DLP API: {e}")
        return pd.DataFrame()


# --- LLM-based PII Detection ---
def detect_pii_with_gemini(text: str) -> List[Dict]:
    """
    Uses the Gemini client to detect PII in text with structured output.

    Args:
        text: The text to analyze.

    Returns:
        A list of detected PII entities (dictionaries with value, type).
    """
    prompt = f"Find all PII in this text and list each piece with its type: {text}"
    gemini_client = clients.get_gemini_client()
    parsed_response = clients.query_gemini_structured(gemini_client, prompt, PIIList)

    if parsed_response and hasattr(parsed_response, 'pii_items'):
        return [{
            'value': item.value,
            'type': item.type.value
        } for item in parsed_response.pii_items]
    return []

def get_allowed_edits(word_length: int) -> int:
    """
    Determine allowed edit distance based on word length using thresholds.
    
    Args:
        word_length: Length of the target word
        
    Returns:
        Number of allowed edits for fuzzy matching
    """
    for max_length, edits in FUZZY_MATCH_THRESHOLDS:
        if word_length <= max_length:
            return edits
    return FUZZY_MATCH_DEFAULT_EDITS


def find_fuzzy_matches(findings_df: pd.DataFrame, text: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find fuzzy matches for PII findings in text, handling multiple matches per finding.
    
    Args:
        findings_df: DataFrame with 'value' and 'type' columns for PII findings
        text: Text to search within
        
    Returns:
        Tuple of (matches_df, unmatched_df) where:
            matches_df: DataFrame with columns start, end, value, type for all matches
            unmatched_df: DataFrame of findings that had no matches
    """

    matches_data = []
    unmatched = []
    
    for _, finding in findings_df.iterrows():
        allowed_edits = get_allowed_edits(len(finding['value']))
        pattern = regex.compile(r'(' + finding['value'] + '){e<=' + str(allowed_edits) + '}', flags=regex.BESTMATCH)
        
        # Find all non-overlapping matches
        pos = 0
        found_match = False
        while pos < len(text):
            match = pattern.search(text, pos)
            if not match:
                break
                
            matches_data.append({
                'start': match.start(),
                'end': match.end(),
                'value': finding['value'],
                'type': finding['type']
            })
            found_match = True
            pos = match.end()
            
        if not found_match:
            unmatched.append({
                'value': finding['value'],
                'type': finding['type']
            })
            
    return pd.DataFrame(matches_data), pd.DataFrame(unmatched)
