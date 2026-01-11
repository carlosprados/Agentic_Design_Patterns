from typing import AsyncGenerator
import os
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.agents import LlmAgent, BaseAgent
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events import Event
except ImportError:
    print("Error: Google ADK components not found.")
    LlmAgent = BaseAgent = InvocationContext = Event = None

# Load environment variables
load_dotenv()

class CustomTaskExecutor(BaseAgent):
    """
    A specialized custom agent with non-LLM behavior.
    """
    name: str = "TaskExecutor"
    description: str = "Executes a predefined task."

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        print(f"[{self.name}] Executing custom logic...")
        yield Event(author=self.name, content="Task finished successfully.")

def setup_coordinator():
    if not LlmAgent:
        return None

    greeter = LlmAgent(
        name="Greeter",
        model="gemini-2.5-flash",
        instruction="You are a friendly greeter."
    )

    task_doer = CustomTaskExecutor()

    # Coordinator delegates to specialized agents
    coordinator = LlmAgent(
        name="Coordinator",
        model="gemini-2.5-flash",
        description="A coordinator that can greet users and execute tasks.",
        instruction="When asked to greet, delegate to the Greeter. When asked to perform a task, delegate to the TaskExecutor.",
        sub_agents=[greeter, task_doer]
    )
    
    return coordinator

def main():
    print("--- ADK Custom Agent Coordinator Example ---")
    coordinator = setup_coordinator()
    if coordinator:
        print("Coordinator and sub-agents (Greeter, TaskExecutor) initialized.")
    else:
        print("Configuration failed.")

if __name__ == "__main__":
    main()
