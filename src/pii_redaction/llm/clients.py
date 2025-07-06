# %%
"""
General clients for interacting with Google and Hugging Face LLMs.
"""

from pydantic import BaseModel
from typing import List, Type
from google import genai
from huggingface_hub import InferenceClient
from pii_redaction import config

# --- Google Gemini Client ---
def get_gemini_client():
    """Initializes and returns the Google GenAI client."""
    return genai.Client(
        vertexai=True,
        project=config.GCP_PROJECT_ID,
        location="global",
    )

def query_gemini_structured(
    client: genai.Client,
    prompt: str,
    response_schema: Type[BaseModel]
) -> BaseModel:
    """
    Queries Gemini for structured JSON output based on a Pydantic model.

    Args:
        client: The initialized Google GenAI client.
        prompt: The full prompt to send to the model.
        response_schema: The Pydantic model to structure the output.

    Returns:
        An instantiated Pydantic model with the LLM's response.
    """
    try:
        response = client.models.generate_content(
            model=config.GEMINI_MODEL_ID,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": response_schema,
            },
        )

        usage_metadata = {
            'prompt_token_count': response.usage_metadata.prompt_token_count,
            'candidates_token_count': response.usage_metadata.candidates_token_count,
            'total_token_count': response.usage_metadata.total_token_count
        }

        return response.parsed, usage_metadata
    except Exception as e:
        print(f"Error querying Gemini: {e}")
        return None, None
