# %%
# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
GEMMA_3_API_KEY = os.environ.get("GEMMA_3_API_KEY")

# %%
def generate(input_text):
    client = genai.Client(
        api_key=GEMMA_3_API_KEY,
    )

    model = "gemma-3-27b-it"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=input_text),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

# %%

generate("Tell me a joke about ducks")

# %%



