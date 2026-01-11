import os
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
except ImportError:
    print("Error: Google ADK components not found. Ensure the library is installed.")
    LlmAgent = ParallelAgent = SequentialAgent = None

# Load environment variables
load_dotenv()

# Placeholder for tools (e.g., google_search)
# In a real scenario, this would be imported or initialized
google_search = None 
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

def main():
    print("--- Google ADK Parallelization Example ---")
    pipeline = setup_parallel_research_agents()
    if pipeline:
        print("Pipeline agents configured successfully.")
        # Execution logic would follow using a runner as seen in Chapter 2
    else:
        print("Configuration failed due to missing components.")

if __name__ == "__main__":
    main()
