import uuid
from typing import Dict, Any, Optional
import os
import asyncio
from dotenv import load_dotenv

# Google ADK imports (Assuming these are available in the environment)
try:
    from google.adk.agents import Agent
    from google.adk.runners import InMemoryRunner
    from google.adk.tools import FunctionTool
    from google.genai import types
except ImportError:
    print("Error: Google ADK components not found. Ensure the library is installed.")
    # Placeholders for type checking or IDE support if needed
    Agent = InMemoryRunner = FunctionTool = types = None

# Load environment variables
load_dotenv()

# --- Define Tool Functions ---
def booking_handler(request: str) -> str:
    """
    Handles booking requests for flights and hotels.
    """
    print("\n-------------------------- Booking Handler Called ----------------------------")
    return f"Booking action for '{request}' has been simulated."

def info_handler(request: str) -> str:
    """
    Handles general information requests.
    """
    print("\n-------------------------- Info Handler Called ----------------------------")
    return f"Information request for '{request}'. Result: Simulated information retrieval."

def unclear_handler(request: str) -> str:
    """
    Handles requests that couldn't be delegated.
    """
    return f"Coordinator could not delegate request: '{request}'. Please clarify."

# --- Create Tools and Agents ---
def setup_adk_agents():
    if not all([Agent, FunctionTool]):
        return None

    booking_tool = FunctionTool(booking_handler)
    info_tool = FunctionTool(info_handler)

    # Define specialized sub-agents
    booking_agent = Agent(
        name="Booker",
        model="gemini-2.5-flash",
        description="A specialized agent that handles all flight and hotel booking requests.",
        tools=[booking_tool]
    )

    info_agent = Agent(
        name="Info",
        model="gemini-2.5-flash",
        description="A specialized agent that provides general information and answers user questions.",
        tools=[info_tool]
    )

    # Define the coordinator agent
    coordinator = Agent(
        name="Coordinator",
        model="gemini-2.5-flash",
        instruction=(
            "You are the main coordinator. Your only task is to analyze incoming user requests "
            "and delegate them to the appropriate specialist agent. Do not try to answer the user directly.\n"
            "- For any requests related to booking flights or hotels, delegate to the 'Booker' agent.\n"
            "- For all other general information questions, delegate to the 'Info' agent."
        ),
        description="A coordinator that routes user requests to the correct specialist agent.",
        sub_agents=[booking_agent, info_agent]
    )
    
    return coordinator

def run_coordinator(runner: InMemoryRunner, request: str):
    """
    Runs the coordinator agent with a given request.
    """
    print(f"\n--- Running Coordinator with request: '{request}' ---")
    final_result = ""
    try:
        user_id = "user_123"
        session_id = str(uuid.uuid4())
        # We use asyncio.run to await the async create_session call in a sync context
        asyncio.run(runner.session_service.create_session(
            app_name=runner.app_name, user_id=user_id, session_id=session_id
        ))

        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role='user',
                parts=[types.Part(text=request)]
            ),
        ):
            if event.is_final_response() and event.content:
                if hasattr(event.content, 'text') and event.content.text:
                     final_result = event.content.text
                elif event.content.parts:
                    text_parts = [part.text for part in event.content.parts if part.text]
                    final_result = "".join(text_parts)
                break

        print(f"Coordinator Final Response: {final_result}")
        return final_result
    except Exception as e:
        error_msg = f"An error occurred while processing: {e}"
        print(error_msg)
        return error_msg

def main():
    print("--- Google ADK Routing Example ---")
    
    coordinator = setup_adk_agents()
    if not coordinator or not InMemoryRunner:
        print("Skipping execution: Required libraries or agent setup failed.")
        return

    runner = InMemoryRunner(coordinator)
    
    # Test cases
    run_coordinator(runner, "Book me a hotel in Paris.")
    run_coordinator(runner, "What is the highest mountain in the world?")
    run_coordinator(runner, "Find flights to Tokyo next month.")

if __name__ == "__main__":
    main()
