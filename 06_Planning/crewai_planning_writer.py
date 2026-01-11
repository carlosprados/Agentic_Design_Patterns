import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

def run_planning_crew(topic: str = "The importance of Reinforcement Learning in AI"):
    """
    Demonstrates a CrewAI setup where an agent plans and then writes an article.
    """
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set.")
        return

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # Define Agent
    planner_writer_agent = Agent(
        role='Article Planner and Writer',
        goal='Plan and then write a concise, engaging summary on a specified topic.',
        backstory=(
            'You are an expert technical writer and content strategist. '
            'Your strength lies in creating a clear, actionable plan before writing, '
            'ensuring the final summary is both informative and easy to digest.'
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Define Task
    high_level_task = Task(
        description=(
            f"1. Create a bullet-point plan for a summary on the topic: '{topic}'.\n"
            f"2. Write the summary based on your plan, keeping it around 200 words."
        ),
        expected_output=(
            "A final report containing two distinct sections:\n\n"
            "### Plan\n"
            "- A bulleted list outlining the main points of the summary.\n\n"
            "### Summary\n"
            "- A concise and well-structured summary of the topic."
        ),
        agent=planner_writer_agent,
    )

    # Create Crew
    crew = Crew(
        agents=[planner_writer_agent],
        tasks=[high_level_task],
        process=Process.sequential,
    )

    print(f"## Running the planning and writing task for topic: {topic} ##")
    result = crew.kickoff()

    print("\n--- Task Result ---")
    print(result)

if __name__ == "__main__":
    run_planning_crew()
