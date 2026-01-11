import asyncio
import os
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.genai import types
    from google.adk import agents
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
except ImportError:
    print("Error: Google ADK components not found.")
    types = agents = Runner = InMemorySessionService = None

# Load environment variables
load_dotenv()

DATASTORE_ID = os.environ.get("DATASTORE_ID")
APP_NAME = "vsearch_app"
USER_ID = "user_123"
SESSION_ID = "session_456"

def setup_vsearch_agent():
    if not agents or not DATASTORE_ID:
        return None
        
    return agents.VSearchAgent(
        name="q2_strategy_vsearch_agent",
        description="Answers questions about documents using Vertex AI Search.",
        model="gemini-2.5-flash",
        datastore_id=DATASTORE_ID,
        model_parameters={"temperature": 0.0}
    )

async def call_vsearch_agent_async(runner, query: str):
    print(f"User: {query}")
    print("Agent: ", end="", flush=True)

    try:
        content = types.Content(role='user', parts=[types.Part(text=query)])
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=content
        ):
            if hasattr(event, 'content_part_delta') and event.content_part_delta:
                print(event.content_part_delta.text, end="", flush=True)

            if event.is_final_response():
                print()
                if event.grounding_metadata:
                    print(f"  (Source Attributions: {len(event.grounding_metadata.grounding_attributions)} sources found)")
                print("-" * 30)

    except Exception as e:
        print(f"\nAn error occurred: {e}")

async def main():
    agent = setup_vsearch_agent()
    if agent and Runner:
        runner = Runner(
            agent=agent,
            app_name=APP_NAME,
            session_service=InMemorySessionService(),
        )
        # In ADK 1.22.0, explicit session creation is required before running if providing a session_id
        await runner.session_service.create_session(
            app_name=runner.app_name, 
            user_id=USER_ID, 
            session_id=SESSION_ID
        )
        await call_vsearch_agent_async(runner, "Summarize the main points about the strategy document.")
    else:
        print("Vertex Search agent setup failed. Ensure DATASTORE_ID is set.")

if __name__ == "__main__":
    asyncio.run(main())
