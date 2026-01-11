import time
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.tools import ToolContext
    from google.adk.sessions import InMemorySessionService
except ImportError:
    print("Error: Google ADK components not found.")
    ToolContext = InMemorySessionService = None

# Load environment variables (mostly for consistency, not strictly needed for this mock)
load_dotenv()

def log_user_login(tool_context: ToolContext) -> dict:
    """
    Updates the session state upon a user login event using ToolContext.
    This demonstrates explicit state management within a tool.
    """
    if not tool_context:
        return {"status": "error", "message": "No context provided"}

    state = tool_context.state

    # Update state variables
    login_count = state.get("user:login_count", 0) + 1
    state["user:login_count"] = login_count
    state["task_status"] = "active"
    state["user:last_login_ts"] = time.time()
    state["temp:validation_needed"] = True

    print(f"[{time.ctime()}] State updated: login_count={login_count}")

    return {
        "status": "success",
        "message": f"User login tracked. Total logins: {login_count}."
    }

def main():
    print("--- ADK Explicit State Update Example ---")
    if not InMemorySessionService:
        print("Required ADK services not available.")
        return

    # Setup session
    session_service = InMemorySessionService()
    app_name, user_id, session_id = "state_app_tool", "user3", "session3"
    session = session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state={"user:login_count": 0, "task_status": "idle"}
    )
    print(f"Initial state: {session.state}")

    # Mocking ToolContext for demonstration
    # In a real app, this is handled by the ADK Runner
    from unittest.mock import MagicMock
    mock_context = MagicMock(spec=ToolContext)
    mock_context.state = session.state

    # Execute tool
    log_user_login(mock_context)

    # Verify updated state
    print(f"State after tool execution: {session.state}")

if __name__ == "__main__":
    main()
