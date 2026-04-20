"""
State Management Pattern using LangGraph.

Demonstrates explicit state manipulation within graph nodes: tools that
update state, and nodes that store their output under specific state keys.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from shared.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

load_dotenv()


class AppState(TypedDict):
    user_name: str
    login_count: int
    last_login: str
    task_status: str
    greeting: str
    summary: str


def login_tracker(state: AppState) -> dict:
    """Simulates a login event by updating state with tracking data."""
    print("--- NODE: login_tracker ---")
    login_count = state.get("login_count", 0) + 1
    last_login = datetime.now(timezone.utc).isoformat()
    print(f"  Login #{login_count} at {last_login}")
    return {
        "login_count": login_count,
        "last_login": last_login,
        "task_status": "logged_in",
    }


def greeter(state: AppState) -> dict:
    """Generates a personalized greeting based on login state."""
    print("--- NODE: greeter ---")
    llm = get_llm(temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Generate a brief, personalized greeting for a user. Context:\n"
         "- Name: {user_name}\n"
         "- Visit number: {login_count}\n"
         "- Last login: {last_login}\n"
         "Be warm but concise."),
        ("user", "Greet me!")
    ])
    chain = prompt | llm | StrOutputParser()
    greeting = chain.invoke({
        "user_name": state["user_name"],
        "login_count": state["login_count"],
        "last_login": state["last_login"],
    })
    return {"greeting": greeting}


def session_summarizer(state: AppState) -> dict:
    """Reads all state and produces a session summary."""
    print("--- NODE: session_summarizer ---")
    summary = (
        f"Session Summary:\n"
        f"  User: {state['user_name']}\n"
        f"  Login count: {state['login_count']}\n"
        f"  Last login: {state['last_login']}\n"
        f"  Status: {state['task_status']}\n"
        f"  Greeting: {state['greeting'][:100]}"
    )
    return {"summary": summary, "task_status": "session_active"}


def build_state_management_graph():
    builder = StateGraph(AppState)
    builder.add_node("login_tracker", login_tracker)
    builder.add_node("greeter", greeter)
    builder.add_node("session_summarizer", session_summarizer)

    builder.add_edge(START, "login_tracker")
    builder.add_edge("login_tracker", "greeter")
    builder.add_edge("greeter", "session_summarizer")
    builder.add_edge("session_summarizer", END)

    return builder.compile()


def main():
    print("--- LangGraph State Management Example ---")
    graph = build_state_management_graph()

    result = graph.invoke({
        "user_name": "Charlie",
        "login_count": 41,
        "last_login": "2025-01-15T10:30:00Z",
        "task_status": "offline",
        "greeting": "",
        "summary": "",
    })

    print(f"\n=== GREETING ===")
    print(result["greeting"])
    print(f"\n=== SESSION SUMMARY ===")
    print(result["summary"])


if __name__ == "__main__":
    main()
