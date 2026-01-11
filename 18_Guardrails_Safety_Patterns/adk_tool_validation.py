from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.agents import Agent
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.tool_context import ToolContext
except ImportError:
    print("Error: Google ADK components not found.")
    Agent = BaseTool = ToolContext = None

# Load environment variables
load_dotenv()

def validate_tool_params(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext
) -> Optional[Dict]:
    """
    Callback triggered before a tool executes to validate parameters.
    Returns a dictionary to block execution with an error, or None to proceed.
    """
    print(f"GUARDRAIL: Validating tool '{tool.name}' with args: {args}")

    # Example: Shared security check
    expected_user_id = tool_context.state.get("session_user_id")
    actual_user_id = args.get("user_id")

    if actual_user_id and expected_user_id and actual_user_id != expected_user_id:
        print(f"GUARDRAIL FAILED: User ID mismatch.")
        return {
            "status": "error",
            "error_message": "Unauthorized tool call: User ID does not match current session."
        }

    print(f"GUARDRAIL PASSED for tool '{tool.name}'.")
    return None

def setup_guarded_agent():
    if not Agent:
        return None

    return Agent(
        model='gemini-2.5-flash',
        name='secure_agent',
        instruction="You are a helpful assistant. Ensure you provide the correct user_id when calling tools.",
        before_tool_callback=validate_tool_params,
        tools=[] # Add actual tools here
    )

if __name__ == "__main__":
    print("--- ADK Tool Validation Guardrail Demo ---")
    agent = setup_guarded_agent()
    if agent:
        print("Agent with 'before_tool_callback' validation initialized.")
    else:
        print("Agent initialization failed.")
