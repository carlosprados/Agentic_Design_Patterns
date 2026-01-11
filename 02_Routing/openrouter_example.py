import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def call_openrouter_example(query: str = "What is the meaning of life?"):
    """
    Demonstrates how to call the OpenRouter API to route requests to various models.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "<OPENROUTER_API_KEY>")
    site_url = os.getenv("YOUR_SITE_URL", "<YOUR_SITE_URL>")
    site_name = os.getenv("YOUR_SITE_NAME", "<YOUR_SITE_NAME>")

    print(f"\n--- Calling OpenRouter with query: '{query}' ---")
    
    if api_key == "<OPENROUTER_API_KEY>":
        print("Warning: OPENROUTER_API_KEY not set in environment.")

    response = requests.post(
      url="https://openrouter.ai/api/v1/chat/completions",
      headers={
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": site_url,
        "X-Title": site_name,
      },
      data=json.dumps({
        "model": "google/gemini-2.5-flash",
        "messages": [
          {
            "role": "user",
            "content": query
          }
        ]
      })
    )

    if response.status_code == 200:
        print("Success!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    call_openrouter_example()
