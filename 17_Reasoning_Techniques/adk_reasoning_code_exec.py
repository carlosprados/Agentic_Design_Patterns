import os
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.agents import Agent
    from google.adk.tools import agent_tool, google_search
    from google.adk.code_executors import BuiltInCodeExecutor
except ImportError:
    print("Error: Google ADK components not found.")
    Agent = agent_tool = google_search = BuiltInCodeExecutor = None

# Load environment variables
load_dotenv()

def setup_reasoning_agents():
    if not Agent:
        return None

    # Search Agent specialized in retrieval
    search_agent = Agent(
        model='gemini-2.5-flash',
        name='SearchAgent',
        instruction="You are a specialist in information retrieval using Google Search.",
        tools=[google_search],
    )

    # Code Agent specialized in execution
    coding_agent = Agent(
        model='gemini-2.5-flash',
        name='CodeAgent',
        instruction="You are a specialist in Python code execution for calculations and data analysis.",
        code_executor=BuiltInCodeExecutor(),
    )

    # Root Agent orchestrating tool-use (including other agents as tools)
    root_agent = Agent(
        name="RootAgent",
        model="gemini-2.5-flash",
        description="Coordinates search and code execution agents to solve complex reasoning problems.",
        tools=[
            agent_tool.AgentTool(agent=search_agent), 
            agent_tool.AgentTool(agent=coding_agent)
        ],
    )
    
    return root_agent

if __name__ == "__main__":
    print("--- ADK Reasoning via Agent Tools & Code Execution ---")
    root = setup_reasoning_agents()
    if root:
        print("Root agent with Search and Code sub-agents initialized.")
    else:
        print("Agent initialization failed.")
