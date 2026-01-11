import os
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.tools import google_search
    from google.adk.agents import Agent
except ImportError:
    print("Error: Google ADK components not found.")
    google_search = Agent = None

# Load environment variables
load_dotenv()

def setup_search_rag_agent():
    if not Agent or not google_search:
        return None

    return Agent(
        name="research_assistant",
        model="gemini-2.5-flash",
        instruction="You help users research topics. When asked, use the Google Search tool to retrieve fresh information.",
        tools=[google_search]
    )

if __name__ == "__main__":
    print("--- ADK Search-based RAG Demo ---")
    agent = setup_search_rag_agent()
    if agent:
        print(f"Agent '{agent.name}' initialized with Google Search tool.")
    else:
        print("Agent initialization failed.")
