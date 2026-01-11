import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_google_deep_research(query: str = "Research the economic impact of semaglutide on global healthcare systems."):
    """
    Demonstrates using Google Gemini with Google Search grounding for research.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set.")
        return

    genai.configure(api_key=api_key)

    print(f"--- Running Google-powered Research for: '{query}' ---")
    
    try:
        # Using gemini-2.5-flash with Google Search grounding
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            tools=[{"google_search_retrieval": {}}]
        )

        response = model.generate_content(query)

        print("\n--- FINAL REPORT ---")
        print(response.text)

        # Check for grounding metadata (citations)
        if response.candidates[0].grounding_metadata:
            print("\n--- GROUNDING METADATA (Citations) ---")
            metadata = response.candidates[0].grounding_metadata
            if hasattr(metadata, 'search_entry_point'):
                print(f"Search Entry Point: {metadata.search_entry_point.rendered_content}")
            
            # Note: Detailed chunk-level citations are available in metadata.grounding_chunks
            # and metadata.grounding_supports for more granular display.

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_google_deep_research()
