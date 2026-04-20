"""
Tool Validation Pattern using LangGraph.

Demonstrates pre-execution parameter validation: a validation node checks
tool arguments against security rules before allowing execution.
"""

import sys
from pathlib import Path
from typing import TypedDict, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from shared.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

load_dotenv()

# Simulated session context
CURRENT_USER_ID = "user_123"


class ValidationState(TypedDict):
    action: str
    target_user_id: str
    params: str
    is_valid: bool
    validation_error: str
    result: str


def validate_parameters(state: ValidationState) -> dict:
    """Validates tool parameters against security rules."""
    print("--- NODE: validate_parameters ---")

    # Rule 1: User can only access their own data
    if state["target_user_id"] != CURRENT_USER_ID:
        error = (f"Access denied: user '{CURRENT_USER_ID}' cannot access "
                 f"data for user '{state['target_user_id']}'.")
        print(f"  BLOCKED: {error}")
        return {"is_valid": False, "validation_error": error}

    # Rule 2: Action must be in allowed list
    allowed_actions = {"read_profile", "update_email", "view_orders"}
    if state["action"] not in allowed_actions:
        error = f"Action '{state['action']}' not permitted. Allowed: {allowed_actions}"
        print(f"  BLOCKED: {error}")
        return {"is_valid": False, "validation_error": error}

    print("  Validation passed.")
    return {"is_valid": True, "validation_error": ""}


def execute_tool(state: ValidationState) -> dict:
    """Executes the validated tool action."""
    print("--- NODE: execute_tool ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Simulate executing this action and return a realistic response:\n"
         "Action: {action}\n"
         "User: {target_user_id}\n"
         "Params: {params}"),
        ("user", "Execute the action.")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "action": state["action"],
        "target_user_id": state["target_user_id"],
        "params": state["params"],
    })
    return {"result": result}


def reject_request(state: ValidationState) -> dict:
    """Returns the validation error to the user."""
    print("--- NODE: reject_request ---")
    return {"result": f"Request rejected: {state['validation_error']}"}


def route_after_validation(state: ValidationState) -> Literal["execute_tool", "reject_request"]:
    if state["is_valid"]:
        return "execute_tool"
    return "reject_request"


def build_validation_graph():
    builder = StateGraph(ValidationState)

    builder.add_node("validate_parameters", validate_parameters)
    builder.add_node("execute_tool", execute_tool)
    builder.add_node("reject_request", reject_request)

    builder.add_edge(START, "validate_parameters")
    builder.add_conditional_edges(
        "validate_parameters", route_after_validation,
        {"execute_tool": "execute_tool", "reject_request": "reject_request"}
    )
    builder.add_edge("execute_tool", END)
    builder.add_edge("reject_request", END)

    return builder.compile()


def main():
    print("--- LangGraph Tool Validation Example ---")
    graph = build_validation_graph()

    # Test 1: Valid request
    print("\n=== Test 1: Valid request ===")
    result = graph.invoke({
        "action": "read_profile",
        "target_user_id": "user_123",
        "params": "include_preferences=true",
    })
    print(f"Result: {result['result'][:300]}")

    # Test 2: Unauthorized user access
    print("\n=== Test 2: Unauthorized access ===")
    result = graph.invoke({
        "action": "read_profile",
        "target_user_id": "user_456",
        "params": "",
    })
    print(f"Result: {result['result']}")

    # Test 3: Forbidden action
    print("\n=== Test 3: Forbidden action ===")
    result = graph.invoke({
        "action": "delete_account",
        "target_user_id": "user_123",
        "params": "",
    })
    print(f"Result: {result['result']}")


if __name__ == "__main__":
    main()
