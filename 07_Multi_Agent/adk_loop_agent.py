from typing import AsyncGenerator
import os
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.agents import LoopAgent, LlmAgent, BaseAgent
    from google.adk.events import Event, EventActions
    from google.adk.agents.invocation_context import InvocationContext
except ImportError:
    print("Error: Google ADK components not found.")
    LoopAgent = LlmAgent = BaseAgent = Event = EventActions = InvocationContext = None

# Load environment variables
load_dotenv()

class ConditionChecker(BaseAgent):
    """
    A custom agent that checks for a 'completed' status in session state to stop a loop.
    """
    name: str = "ConditionChecker"
    description: str = "Checks if a process is complete and signals the loop to stop."

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        status = context.session.state.get("status", "pending")
        is_done = (status == "completed")

        if is_done:
            print(f"[{self.name}] Termination condition met.")
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            print(f"[{self.name}] Continuing loop...")
            yield Event(author=self.name, content="Condition not met, continuing loop.")

def setup_loop_agent():
    if not LoopAgent:
        return None

    process_step = LlmAgent(
        name="ProcessingStep",
        model="gemini-2.5-flash",
        instruction="You are a step in a longer process. Perform your task. If you are the final step, update session state by setting 'status' to 'completed'."
    )

    poller = LoopAgent(
        name="StatusPoller",
        max_iterations=10,
        sub_agents=[
            process_step,
            ConditionChecker()
        ]
    )
    
    return poller

def main():
    print("--- ADK Loop Agent Example ---")
    poller = setup_loop_agent()
    if poller:
        print("Loop agent 'StatusPoller' configured for up to 10 iterations.")
    else:
        print("Configuration failed.")

if __name__ == "__main__":
    main()
