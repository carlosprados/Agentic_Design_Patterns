import os
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.agents import SequentialAgent, Agent
except ImportError:
    print("Error: Google ADK components not found.")
    SequentialAgent = Agent = None

# Load environment variables
load_dotenv()

def setup_sequential_pipeline():
    if not SequentialAgent or not Agent:
        return None

    # Step 1: Fetch Data
    step1 = Agent(
        name="Step1_Fetch", 
        model="gemini-2.5-flash",
        instruction="Fetch relevant data for the user's query.",
        output_key="data"
    )

    # Step 2: Process Data
    step2 = Agent(
        name="Step2_Process",
        model="gemini-2.5-flash",
        instruction="Analyze the information found in state['data'] and provide a summary."
    )

    pipeline = SequentialAgent(
        name="MyPipeline",
        sub_agents=[step1, step2]
    )
    
    return pipeline

def main():
    print("--- ADK Sequential Agent Pipeline Example ---")
    pipeline = setup_sequential_pipeline()
    if pipeline:
        print("Sequential pipeline 'MyPipeline' configured with 2 steps.")
    else:
        print("Configuration failed.")

if __name__ == "__main__":
    main()
