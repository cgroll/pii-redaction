# %%
"""
Central configuration file for the PII analysis project.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Project Configuration ---
GCP_PROJECT_ID = "firm-dimension-461208-d1"
HF_TOKEN = os.getenv('HF_TOKEN')


# --- Model Configuration ---
GEMINI_MODEL_ID = "gemini-2.5-flash-lite-preview-06-17"
GEMINI_INPUT_TOKEN_COST = 0.1 # $0.1 per 1M tokens
GEMINI_OUTPUT_TOKEN_COST = 0.4 # $0.4 per 1M tokens

HUGGINGFACE_MODEL_ID = "google/gemma-3-27b-it"


# --- PII & DLP Configuration ---
RELEVANT_PII_TYPES = [
    'USERNAME', 'IDCARD', 'SOCIALNUMBER', 'EMAIL', 'PASSPORT',
    'DRIVERLICENSE', 'BOD', 'LASTNAME1', 'IP', 'GIVENNAME1',
    'TEL', 'STREET', 'PASS', 'SECADDRESS', 'LASTNAME2',
    'GIVENNAME2', 'GEOCOORD', 'LASTNAME3'
]

DLP_INFO_TYPES = [
    {"name": "PHONE_NUMBER"}, {"name": "EMAIL_ADDRESS"},
    {"name": "CREDIT_CARD_NUMBER"}, {"name": "US_SOCIAL_SECURITY_NUMBER"},
    {"name": "PERSON_NAME"}, {"name": "FEMALE_NAME"}, {"name": "MALE_NAME"},
    {"name": "FIRST_NAME"}, {"name": "LAST_NAME"}, {"name": "DATE_OF_BIRTH"},
    {"name": "LOCATION"}, {"name": "STREET_ADDRESS"}
]

LLM_PII_TYPES = [
    {"name": "PHONE_NUMBER"}, {"name": "EMAIL_ADDRESS"},
    {"name": "CREDIT_CARD_NUMBER"}, {"name": "US_SOCIAL_SECURITY_NUMBER"},
    {"name": "PERSON_NAME"}, {"name": "DATE_OF_BIRTH"},
    {"name": "LOCATION"}, {"name": "STREET_ADDRESS"}
]

# --- Fuzzy Matching Configuration ---
# Configuration for fuzzy matching thresholds
FUZZY_MATCH_THRESHOLDS = [
    (3, 0),   # Words <= 3 chars: case differences only
    (5, 1),   # Words <= 5 chars: 1 edit
    (8, 2),   # Words <= 8 chars: 2 edits
    (12, 3),  # Words <= 12 chars: 3 edits  
    (15, 4),  # Words <= 15 chars: 4 edits
]
FUZZY_MATCH_DEFAULT_EDITS = 5  # For words > 15 chars    