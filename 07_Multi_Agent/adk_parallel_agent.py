import os
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.agents import Agent, ParallelAgent
except ImportError:
    print("Error: Google ADK components not found.")
    Agent = ParallelAgent = None

# Load environment variables
load_dotenv()

def setup_parallel_agents():
    if not Agent or not ParallelAgent:
        return None

    weather_fetcher = Agent(
        name="weather_fetcher",
        model="gemini-2.5-flash",
        instruction="Fetch the weather for the given location and return only the weather report.",
        output_key="weather_data"
    )

    news_fetcher = Agent(
        name="news_fetcher",
        model="gemini-2.5-flash",
        instruction="Fetch the top news story for the given topic and return only that story.",
        output_key="news_data"
    )

    data_gatherer = ParallelAgent(
        name="data_gatherer",
        sub_agents=[
            weather_fetcher,
            news_fetcher
        ]
    )
    
    return data_gatherer

def main():
    print("--- ADK Parallel Agent Example ---")
    gatherer = setup_parallel_agents()
    if gatherer:
        print("Parallel agent 'data_gatherer' configured with weather and news fetchers.")
    else:
        print("Configuration failed.")

if __name__ == "__main__":
    main()
