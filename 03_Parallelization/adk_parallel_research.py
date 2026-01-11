import os
import uuid
import asyncio
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools import google_search
    from google.genai import types
except ImportError:
    print("Error: Google ADK components not found. Ensure the library is installed.")
    LlmAgent = ParallelAgent = SequentialAgent = Runner = InMemorySessionService = google_search = types = None

# Load environment variables
load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"

def setup_parallel_research_agents():
    if not all([LlmAgent, ParallelAgent, SequentialAgent]):
        return None

    # Researcher 1: Renewable Energy
    researcher_agent_1 = LlmAgent(
        name="RenewableEnergyResearcher",
        model=GEMINI_MODEL,
        instruction="""You are an AI Research Assistant specializing in energy.
    Research the latest advancements in 'renewable energy sources'.
    Use the Google Search tool provided.
    Summarize your key findings concisely (1-2 sentences).
    Output *only* the summary.
    """,
        description="Researches renewable energy sources.",
        tools=[google_search] if google_search else [],
        output_key="renewable_energy_result"
    )

    # Researcher 2: Electric Vehicles
    researcher_agent_2 = LlmAgent(
        name="EVResearcher",
        model=GEMINI_MODEL,
        instruction="""You are an AI Research Assistant specializing in transportation.
    Research the latest developments in 'electric vehicle technology'.
    Use the Google Search tool provided.
    Summarize your key findings concisely (1-2 sentences).
    Output *only* the summary.
    """,
        description="Researches electric vehicle technology.",
        tools=[google_search] if google_search else [],
        output_key="ev_technology_result"
    )

    # Researcher 3: Carbon Capture
    researcher_agent_3 = LlmAgent(
        name="CarbonCaptureResearcher",
        model=GEMINI_MODEL,
        instruction="""You are an AI Research Assistant specializing in climate solutions.
    Research the current state of 'carbon capture methods'.
    Use the Google Search tool provided.
    Summarize your key findings concisely (1-2 sentences).
    Output *only* the summary.
    """,
        description="Researches carbon capture methods.",
        tools=[google_search] if google_search else [],
        output_key="carbon_capture_result"
    )

    # Parallel Agent for concurrent research
    parallel_research_agent = ParallelAgent(
        name="ParallelWebResearchAgent",
        sub_agents=[researcher_agent_1, researcher_agent_2, researcher_agent_3],
        description="Runs multiple research agents in parallel to gather information."
    )

    # Merger Agent for synthesis
    merger_agent = LlmAgent(
        name="SynthesisAgent",
        model=GEMINI_MODEL,
        instruction="""You are an AI Assistant responsible for combining research findings into a structured report.

    Your primary task is to synthesize the following research summaries, clearly attributing findings to their source areas. Structure your response using headings for each topic. Ensure the report is coherent and integrates the key points smoothly.

    **Crucially: Your entire response MUST be grounded *exclusively* on the information provided in the 'Input Summaries' below. Do NOT add any external knowledge, facts, or details not present in these specific summaries.**

    **Input Summaries:**

    *   **Renewable Energy:**
        {renewable_energy_result}

    *   **Electric Vehicles:**
        {ev_technology_result}

    *   **Carbon Capture:**
        {carbon_capture_result}

    **Output Format:**

    ## Summary of Recent Sustainable Technology Advancements

    ### Renewable Energy Findings
    (Based on RenewableEnergyResearcher's findings)
    [Synthesize and elaborate *only* on the renewable energy input summary provided above.]

    ### Electric Vehicle Findings
    (Based on EVResearcher's findings)
    [Synthesize and elaborate *only* on the EV input summary provided above.]

    ### Carbon Capture Findings
    (Based on CarbonCaptureResearcher's findings)
    [Synthesize and elaborate *only* on the carbon capture input summary provided above.]

    ### Overall Conclusion
    [Provide a brief (1-2 sentence) concluding statement that connects *only* the findings presented above.]

    Output *only* the structured report following this format.
    """,
        description="Combines research findings into a structured report."
    )

    # Sequential Agent to orchestrate the whole pipeline
    sequential_pipeline_agent = SequentialAgent(
        name="ResearchAndSynthesisPipeline",
        sub_agents=[parallel_research_agent, merger_agent],
        description="Coordinates parallel research and synthesizes the results."
    )
    
    return sequential_pipeline_agent

async def main():
    print("--- Google ADK Parallelization Example (Async) ---")
    pipeline = setup_parallel_research_agents()
    if not pipeline or not Runner:
        print("Configuration failed due to missing components.")
        return

    # Using the async Runner and explicit session service
    session_service = InMemorySessionService()
    runner = Runner(
        agent=pipeline, 
        app_name="parallel_research_app", 
        session_service=session_service
    )
    
    user_id = "researcher_001"
    session_id = str(uuid.uuid4())
    
    # In ADK 1.22.0, explicit session creation is required
    await runner.session_service.create_session(
        app_name=runner.app_name, 
        user_id=user_id, 
        session_id=session_id
    )

    request = "Conduct research on the current state of green technologies."
    print(f"\n--- Running Pipeline: {request} ---")

    try:
        final_report = ""
        # Using run_async for a clean async lifecycle
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role='user',
                parts=[types.Part(text=request)]
            ),
        ):
            if event.is_final_response() and event.content:
                if hasattr(event.content, 'text') and event.content.text:
                     final_report = event.content.text
                elif event.content.parts:
                    text_parts = [part.text for part in event.content.parts if part.text]
                    final_report = "".join(text_parts)
                # We don't break here to ensure the generator finishes its lifecycle

        print("\n--- FINAL SYNTHESIZED REPORT ---")
        print(final_report)

    except Exception as e:
        print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())
