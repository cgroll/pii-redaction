# %%
"""
PII detection logic using Google Cloud DLP and LLM wrappers.
"""

import pandas as pd
import google.cloud.dlp_v2
from pii_redaction.llm import clients
from typing import List, Dict
from pydantic import BaseModel
from enum import Enum
import regex
from pii_redaction import config
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import os
from dotenv import load_dotenv
import ollama
import json

load_dotenv()

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

class PIIEntityPermissive(BaseModel):
    """A single detected piece of Personally Identifiable Information (PII)."""
    value: str
    type: str

class PIIList(BaseModel):
    """A list of all PII entities found in the text."""
    pii_items: List[PIIEntity]

class PIIListPermissive(BaseModel):
    """A list of all PII entities found in the text."""
    pii_items: List[PIIEntityPermissive]

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
    parsed_response, usage_metadata = clients.query_gemini_structured(gemini_client, prompt, PIIList)

    if parsed_response and hasattr(parsed_response, 'pii_items'):
        return [{
            'value': item.value,
            'type': item.type.value
        } for item in parsed_response.pii_items], usage_metadata
    return [], usage_metadata

def detect_pii_with_gemma_api(text: str) -> List[Dict]:
    """
    Uses the Gemini client to detect PII in text with structured output.

    Args:
        text: The text to analyze.

    Returns:
        A list of detected PII entities. Each entity is a dictionary with
        'value', 'type', and a boolean 'is_valid_type' to indicate if the
        type conforms to the allowed PIIType enum.
    """

    model = ChatOpenAI(
        model="google/gemma-3-27b-it-fast",
        base_url="https://router.huggingface.co/nebius/v1",
        api_key=os.environ["HF_TOKEN"]
    )

    parser = PydanticOutputParser(pydantic_object=PIIListPermissive)

    prompt = PromptTemplate(
        template="Find all PII in this text and list each piece with its type. Allowed types are: {allowed_types}\n{query}\n{format_instructions}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions(),
                           "allowed_types": "\n".join([f"- {t}" for t in config.LLM_PII_TYPES])}
    )

    chain = prompt | model | parser
    pii_list = chain.invoke({"query": text})

    valid_types = {member.value for member in PIIType}

    return [{
        'value': item.value,
        'type': item.type,
        'is_valid_type': item.type in valid_types
    } for item in pii_list.pii_items]

def detect_pii_with_ollama(text: str, model='gemma3:1b'):
    """
    Detects PII in text using a local Ollama model with JSON output.
    """
    try:
        
        # parser = PydanticOutputParser(pydantic_object=PIIListPermissive)
        # json_schema = parser.get_format_instructions()

        system_prompt = """
        You are an expert PII detection system. Your task is to identify and extract
        Personally Identifiable Information (PII) from the given text.

        You MUST format your output as a JSON object that strictly adheres to the
        following JSON Schema. Do not include any other explanatory text.
        The schema is a list of objects, where each object has a 'value' and a 'type' key.

        {'pii_items': [
            {'value': 'the value of the PII entity found',
              'type': 'the type of the PII entity found',
            },
            {'value': 'the value of another PII entity found',
              'type': 'the type of another PII entity found',
            }
        ]}

        Allowed types are:
        - PHONE_NUMBER
        - EMAIL_ADDRESS
        - CREDIT_CARD_NUMBER
        - US_SOCIAL_SECURITY_NUMBER
        - PERSON_NAME
        - DATE_OF_BIRTH
        - LOCATION
        - STREET_ADDRESS

        Here is the text to analyze:
        """

        # Make the call to the Ollama API using the ollama-python library
        response = ollama.chat(
            model=model, 
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': text},
            ],
            format='json',
            options={'num_predict': 300
                }
        )

        content_string = response['message']['content']

        parsed_json = json.loads(content_string)

        valid_types = {member.value for member in PIIType}

        if 'pii_items' not in parsed_json:
            return []

        return [{
            'value': item['value'],
            'type': item['type'],
            'is_valid_type': item['type'] in valid_types
        } for item in parsed_json['pii_items']]

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def get_allowed_edits(word_length: int) -> int:
    """
    Determine allowed edit distance based on word length using thresholds.
    
    Args:
        word_length: Length of the target word
        
    Returns:
        Number of allowed edits for fuzzy matching
    """
    for max_length, edits in config.FUZZY_MATCH_THRESHOLDS:
        if word_length <= max_length:
            return edits
    return config.FUZZY_MATCH_DEFAULT_EDITS


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
        escaped_value = regex.escape(finding['value'])
        pattern_str = r'(' + escaped_value + '){e<=' + str(allowed_edits) + '}'
        pattern = regex.compile(pattern_str, flags=regex.BESTMATCH)
        
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
