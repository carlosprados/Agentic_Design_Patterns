import os
import asyncio
import uuid
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.agents import SequentialAgent, LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
except ImportError:
    print("Error: Google ADK components not found. Ensure the library is installed.")
    SequentialAgent = LlmAgent = Runner = InMemorySessionService = types = None

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
        Read the text provided in the state key 'draft_text' and verify its factual accuracy.
        Provide a detailed breakdown of your findings.
        """,
        output_key="review_output"
    )

    # The SequentialAgent ensures the generator runs before the reviewer.
    review_pipeline = SequentialAgent(
        name="WriteAndReview_Pipeline",
        sub_agents=[generator, reviewer]
    )
    
    return review_pipeline

async def main():
    print("--- Google ADK Reflection (Draft & Review) Pipeline ---")
    pipeline = setup_adk_reflection()
    if not pipeline or not Runner:
        print("Configuration failed.")
        return

    # Setup runner with session service
    session_service = InMemorySessionService()
    runner = Runner(
        agent=pipeline,
        app_name="reflection_app",
        session_service=session_service
    )
    
    user_id, session_id = "user_123", str(uuid.uuid4())
    
    # ADK 1.22.0 requires explicit session creation
    await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id
    )

    subject = "The impact of artificial intelligence on modern healthcare."
    print(f"\nSubject: {subject}")
    print("--- Running Pipeline ---\n")

    try:
        content = types.Content(role='user', parts=[types.Part(text=subject)])
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content
        ):
            # Capture final response which will be from the overall pipeline
            if event.is_final_response():
                # We extract the state manually from the session service to show both outputs
                session = await runner.session_service.get_session(
                    app_name=runner.app_name, 
                    user_id=user_id, 
                    session_id=session_id
                )
                
                print("\n" + "="*50)
                print("STEP 1: INITIAL DRAFT (DraftWriter)")
                print("="*50)
                print(session.state.get('draft_text', "No draft generated."))
                
                print("\n" + "="*50)
                print("STEP 2: FACT-CHECK REVIEW (FactChecker)")
                print("="*50)
                print(session.state.get('review_output', "No review generated."))
                print("="*50)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
