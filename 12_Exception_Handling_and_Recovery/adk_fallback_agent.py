import os
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.agents import Agent, SequentialAgent
except ImportError:
    print("Error: Google ADK components not found.")
    Agent = SequentialAgent = None

# Load environment variables
load_dotenv()

# Mock tools for demonstration
def get_precise_location_info(address: str) -> dict:
    """
    Simulated tool that might fail for precise lookup.
    """
    print(f"TOOL: Attempting precise lookup for '{address}'...")
    # Simulate a failure for specific addresses
    if "unknown" in address.lower():
        raise ValueError("Precise location not found.")
    return {"status": "success", "data": f"Precise info for {address}"}

def get_general_area_info(city: str) -> dict:
    """
    Simulated tool for general area fallback.
    """
    print(f"TOOL: Falling back to general area info for '{city}'...")
    return {"status": "success", "data": f"General info for {city}"}

def setup_robust_location_agent():
    if not Agent or not SequentialAgent:
        return None

    # Agent 1: Primary Handler
    primary_handler = Agent(
        name="primary_handler",
        model="gemini-2.5-flash",
        instruction="""
        Your job is to get precise location information.
        Use the get_precise_location_info tool with the user's provided address.
        If the tool fails, set state["primary_location_failed"] = True.
        """,
        tools=[get_precise_location_info]
    )

    # Agent 2: Fallback Handler
    fallback_handler = Agent(
        name="fallback_handler",
        model="gemini-2.5-flash",
        instruction="""
        Check if the primary location lookup failed by looking at state["primary_location_failed"].
        - If it is True, extract the city from the user's original query and use the get_general_area_info tool.
        - If it is False, do nothing.
        """,
        tools=[get_general_area_info]
    )

    # Agent 3: Response Agent
    response_agent = Agent(
        name="response_agent",
        model="gemini-2.5-flash",
        instruction="""
        Review the location information stored in state["location_result"].
        Present this information clearly and concisely to the user.
        If state["location_result"] does not exist or is empty, apologize that you could not retrieve the location.
        """,
    )

    # Orchestration
    robust_location_agent = SequentialAgent(
        name="robust_location_agent",
        sub_agents=[primary_handler, fallback_handler, response_agent]
    )
    
    return robust_location_agent

if __name__ == "__main__":
    print("--- ADK Exception Handling & Recovery Demo ---")
    agent = setup_robust_location_agent()
    if agent:
        print("Robust location agent with sequential fallback initialized.")
    else:
        print("Agent initialization failed.")
