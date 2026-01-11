from typing import AsyncGenerator
import os
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.agents import Agent, BaseAgent
    from google.adk.events import Event
    from google.adk.agents.invocation_context import InvocationContext
except ImportError:
    print("Error: Google ADK components not found.")
    Agent = BaseAgent = Event = InvocationContext = None

# Load environment variables
load_dotenv()

def setup_specialized_agents():
    if not Agent:
        return None, None

    # More powerful/expensive agent
    pro_agent = Agent(
        name="GeminiProAgent",
        model="gemini-1.5-pro",
        instruction="You are a highly capable agent for complex problem-solving."
    )

    # Faster/cheaper agent
    flash_agent = Agent(
        name="GeminiFlashAgent",
        model="gemini-2.5-flash",
        instruction="You are a quick assistant for straightforward questions."
    )
    
    return pro_agent, flash_agent

class ResourceRoutingAgent(BaseAgent):
    """
    A custom ADK agent that routes queries based on length as a proxy for complexity.
    """
    name: str = "QueryRouter"
    description: str = "Routes user queries based on resource needs."

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        # Simple heuristic: query length
        user_query = context.current_message.parts[0].text if context.current_message.parts else ""
        word_count = len(user_query.split())

        pro_agent, flash_agent = setup_specialized_agents()
        
        if word_count < 15:
            print(f"[{self.name}] Routing to FLASH agent (Words: {word_count})")
            # In a real ADK flow, you'd use self.transfer_to_agent or similar
            yield Event(author=self.name, content=f"Routing simple query to Flash agent...")
        else:
            print(f"[{self.name}] Routing to PRO agent (Words: {word_count})")
            yield Event(author=self.name, content=f"Routing complex query to Pro agent...")

if __name__ == "__main__":
    print("--- ADK Resource-Aware Routing Demo ---")
    router = ResourceRoutingAgent()
    print("Resource router initialized with simple word-count heuristic.")
