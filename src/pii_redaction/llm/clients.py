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

# --- Hugging Face Client ---
def get_huggingface_client():
    """Initializes and returns the Hugging Face Inference client."""
    return InferenceClient(provider="featherless-ai", api_key=config.HF_TOKEN)

def query_huggingface_with_tools(
    client: InferenceClient,
    prompt: str,
    tool_schema: dict
) -> dict:
    """
    Queries a Hugging Face model using a tool schema for structured output.

    Args:
        client: The initialized Hugging Face client.
        prompt: The user prompt.
        tool_schema: The JSON schema definition of the tool.

    Returns:
        The arguments of the tool call as a dictionary.
    """
    try:
        completion = client.chat.completions.create(
            model=config.HUGGINGFACE_MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that uses tools to answer questions."},
                {"role": "user", "content": prompt},
            ],
            tools=[tool_schema],
            tool_choice={"type": "function", "function": {"name": tool_schema["function"]["name"]}},
        )
        message = completion.choices[0].message
        if message.tool_calls:
            return message.tool_calls[0].function.arguments
        return None
    except Exception as e:
        print(f"Error querying Hugging Face with tools: {e}")
        return None