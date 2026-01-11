import os
from typing import Optional
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.agents import Agent
    from google.adk.callbacks import CallbackContext
    from google.adk.models.llm import LlmRequest
    from google.genai import types
except ImportError:
    print("Error: Google ADK components not found.")
    Agent = CallbackContext = LlmRequest = types = None

# Load environment variables
load_dotenv()

# Mock tools for demonstration
def troubleshoot_issue(issue: str) -> dict:
    """
    Simulated tool for initial technical troubleshooting.
    """
    print(f"TOOL: Troubleshooting issue: '{issue}'")
    return {"status": "success", "report": f"Troubleshooting completed for {issue}."}

def create_ticket(issue_type: str, details: str) -> dict:
    """
    Simulated tool for creating a support ticket.
    """
    print(f"TOOL: Creating ticket for {issue_type}. Details: {details}")
    return {"status": "success", "ticket_id": "TICKET456"}

def escalate_to_human(issue_type: str) -> dict:
    """
    Simulated tool for escalating to a human support agent.
    """
    print(f"TOOL: Escalating {issue_type} to human specialist.")
    return {"status": "success", "message": f"Assigned to {issue_type} specialist queue."}

def personalization_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmRequest]:
    """
    ADK Callback that injects customer personalization data into the LLM request.
    Useful for 'Human-in-the-Loop' context or just personalized AI support.
    """
    if not callback_context or not llm_request:
        return None

    # Retrieve customer info from session state
    customer_info = callback_context.state.get("customer_info", {})
    if not customer_info:
        return None

    customer_name = customer_info.get("name", "valued customer")
    customer_tier = customer_info.get("tier", "standard")
    recent_purchases = customer_info.get("recent_purchases", [])

    personalization_note = (
        f"\n[SYSTEM DATA: PERSONALIZATION]\n"
        f"Customer Name: {customer_name}\n"
        f"Customer Tier: {customer_tier}\n"
    )
    if recent_purchases:
        personalization_note += f"Recent Purchases: {', '.join(recent_purchases)}\n"

    # Inject into the request as a high-priority instruction or context
    if llm_request.contents:
        system_content = types.Content(
            role="system", 
            parts=[types.Part(text=personalization_note)]
        )
        llm_request.contents.insert(0, system_content)
    
    return None # Return None to signal the callback has modified the request in-place

def setup_support_agent():
    if not Agent:
        return None

    support_agent = Agent(
        name="technical_support_specialist",
        model="gemini-2.5-flash",
        instruction="""
        You are a empathetic technical support specialist.
        1. Greet the customer using their name if available in the personalization data.
        2. Reference their support history if present in state["customer_info"]["support_history"].
        3. Use troubleshoot_issue for initial diagnostics.
        4. If the issue is complex or the user expresses significant frustration, use escalate_to_human.
        5. Use create_ticket for logging issues that need follow-up.
        """,
        tools=[troubleshoot_issue, create_ticket, escalate_to_human]
    )
    
    # In a real app, you would register the callback with the agent or runner
    # agent.register_callback(personalization_callback)
    
    return support_agent

if __name__ == "__main__":
    print("--- ADK Human-in-the-Loop & Personalization Demo ---")
    agent = setup_support_agent()
    if agent:
        print("Technical support agent with personalization callback logic initialized.")
    else:
        print("Agent initialization failed.")
