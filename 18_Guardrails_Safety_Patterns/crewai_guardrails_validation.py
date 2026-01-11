import logging
import json
import re
from typing import Tuple, Any
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# CrewAI imports
try:
    from crewai import Agent, Task, Crew, Process
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    print("Error: CrewAI components not found.")
    Agent = Task = Crew = Process = None

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- 1. Content Moderation (Input Guardrail) ---
def moderate_input(text: str) -> Tuple[bool, str]:
    """
    Checks for forbidden keywords using regex.
    """
    forbidden = ["violence", "hate", "illegal"]
    pattern = r'\b(' + '|'.join(re.escape(k) for k in forbidden) + r')\b'
    if re.search(pattern, text, re.IGNORECASE):
        return False, "Input rejected: contains forbidden terms."
    return True, "Input clean."

# --- 2. Structured Output & Validation (Output Guardrail) ---
class ResearchSummary(BaseModel):
    title: str = Field(description="Title of the research.")
    key_findings: list[str] = Field(description="3-5 key insights.")
    confidence_score: float = Field(description="Score between 0.0 and 1.0.")

def validate_research_output(output: str) -> Tuple[bool, Any]:
    """
    Validates LLM string output against a Pydantic model.
    """
    try:
        data = json.loads(output)
        summary = ResearchSummary.model_validate(data)
        
        # Additional logical checks
        if len(summary.key_findings) < 3:
            return False, "Validation failed: at least 3 findings required."
        if not (0.0 <= summary.confidence_score <= 1.0):
            return False, "Validation failed: score must be between 0 and 1."
            
        logging.info("Output guardrail PASSED.")
        return True, output
    except (json.JSONDecodeError, ValidationError) as e:
        logging.error(f"Output guardrail FAILED: {e}")
        return False, f"Invalid format: {e}"

# --- 3. CrewAI Setup ---
def setup_guarded_crew():
    if not Agent:
        return None

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    researcher = Agent(
        role='Analyst',
        goal='Summarize research into validated JSON.',
        backstory='Expert in structured data extraction.',
        verbose=True,
        llm=llm
    )

    research_task = Task(
        description="Summarize findings on 'renewable energy' into a valid JSON object.",
        expected_output="JSON with title, key_findings (3-5), and confidence_score.",
        agent=researcher,
        guardrail=validate_research_output, # CrewAI guardrail hook
        output_pydantic=ResearchSummary
    )

    return Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        llm=llm
    )

if __name__ == "__main__":
    print("--- CrewAI Guardrails & Validation Demo ---")
    user_input = "Please research renewable energy."
    is_safe, msg = moderate_input(user_input)
    
    if is_safe:
        print("Input passed moderation. Ready to kickoff crew...")
        # crew = setup_guarded_crew()
        # crew.kickoff()
    else:
        print(msg)
