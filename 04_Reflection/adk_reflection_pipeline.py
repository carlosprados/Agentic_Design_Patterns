import os
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.agents import SequentialAgent, LlmAgent
except ImportError:
    print("Error: Google ADK components not found. Ensure the library is installed.")
    SequentialAgent = LlmAgent = None

# Load environment variables
load_dotenv()

def setup_adk_reflection():
    if not all([SequentialAgent, LlmAgent]):
        return None

    # The first agent generates the initial draft.
    generator = LlmAgent(
        name="DraftWriter",
        model="gemini-2.5-flash",
        description="Generates initial draft content on a given subject.",
        instruction="Write a short, informative paragraph about the user's subject.",
        output_key="draft_text" 
    )

    # The second agent critiques the draft from the first agent.
    reviewer = LlmAgent(
        name="FactChecker",
        model="gemini-2.5-flash",
        description="Reviews a given text for factual accuracy and provides a structured critique.",
        instruction="""
        You are a meticulous fact-checker.
        1. Read the text provided in the state key 'draft_text'.
        2. Carefully verify the factual accuracy of all claims.
        3. Your final output must be a dictionary containing two keys:
           - "status": A string, either "ACCURATE" or "INACCURATE".
           - "reasoning": A string providing a clear explanation for your status, citing specific issues if any are found.
        """,
        output_key="review_output"
    )

    # The SequentialAgent ensures the generator runs before the reviewer.
    review_pipeline = SequentialAgent(
        name="WriteAndReview_Pipeline",
        sub_agents=[generator, reviewer]
    )
    
    return review_pipeline

def main():
    print("--- Google ADK Reflection Pipeline Example ---")
    pipeline = setup_adk_reflection()
    if pipeline:
        print("Reflection pipeline configured successfully.")
        # Logic to run with runner...
    else:
        print("Configuration failed.")

if __name__ == "__main__":
    main()
