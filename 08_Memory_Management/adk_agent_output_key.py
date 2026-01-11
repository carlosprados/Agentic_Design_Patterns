import os
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.agents import LlmAgent
    from google.adk.sessions import InMemorySessionService
    from google.adk.runners import Runner
    from google.genai.types import Content, Part
except ImportError:
    print("Error: Google ADK components not found.")
    LlmAgent = InMemorySessionService = Runner = Content = Part = None

# Load environment variables
load_dotenv()

def setup_and_run_agent():
    if not LlmAgent or not Runner:
        return

    # Define an LlmAgent with an output_key to store its response in session state
    greeting_agent = LlmAgent(
        name="Greeter",
        model="gemini-2.5-flash",
        instruction="Generate a short, friendly greeting.",
        output_key="last_greeting"
    )

    # Setup environment
    app_name, user_id, session_id = "state_app", "user1", "session1"
    session_service = InMemorySessionService()
    
    runner = Runner(
        agent=greeting_agent,
        app_name=app_name,
        session_service=session_service
    )
    
    session = session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )

    print(f"Initial state: {session.state}")

    # Run the agent
    user_message = Content(parts=[Part(text="Hello")])
    print("\n--- Running the Greeter Agent ---")
    
    # Normally we'd use runner.run or runner.run_async
    # Here we simulate the process
    events = runner.run(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message
    )
    
    for event in events:
        if event.is_final_response():
            print("Agent responded.")

    # Check updated state
    updated_session = session_service.get_session(app_name, user_id, session_id)
    print(f"\nState after agent run (should contain 'last_greeting'):\n{updated_session.state}")

if __name__ == "__main__":
    setup_and_run_agent()
