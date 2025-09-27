import os
import google.genai as genai
from google.genai import types

def print_llm_response(prompt: str):
    """
    Sends the given prompt to the Gemini model and prints the response.
    """
    client = genai.Client(api_key='')

    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]

    # Remove ThinkingConfig (not supported in your version)
    generate_content_config = types.GenerateContentConfig()

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

def get_llm_response(prompt: str) -> str:
    """
    Sends the given prompt to the Gemini model and returns the response as a string.
    """
    client = genai.Client(api_key='')

    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]

    generate_content_config = types.GenerateContentConfig()

    response_text = ""  # collect streamed response
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        response_text += chunk.text or ""

    return response_text.strip()
