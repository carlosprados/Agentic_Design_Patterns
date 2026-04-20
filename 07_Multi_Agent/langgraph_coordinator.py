"""
Custom Coordinator Pattern using LangGraph.

Demonstrates a coordinator node that classifies requests and routes them
to specialized sub-nodes: a Greeter for greetings and a TaskExecutor
for task-related requests (with custom non-LLM logic).
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


class CoordinatorState(TypedDict):
    request: str
    route: str
    response: str


def coordinator(state: CoordinatorState) -> dict:
    """Classifies the request and decides routing."""
    print("--- NODE: coordinator ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Classify the user's message:\n"
         "- If it's a greeting (hello, hi, hey, good morning, etc.), output 'greeter'.\n"
         "- If it's a task or question, output 'task_executor'.\n"
         "ONLY output one word: 'greeter' or 'task_executor'."),
        ("user", "{request}")
    ])
    chain = prompt | llm | StrOutputParser()
    route = chain.invoke({"request": state["request"]}).strip().lower()
    print(f"  Route: {route}")
    return {"route": route}


def greeter(state: CoordinatorState) -> dict:
    """Handles greeting messages with a friendly response."""
    print("--- NODE: greeter ---")
    llm = get_llm(temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly assistant. Respond warmly to the greeting."),
        ("user", "{request}")
    ])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"request": state["request"]})
    return {"response": response}


def task_executor(state: CoordinatorState) -> dict:
    """Handles task requests with custom logic + LLM processing."""
    print("--- NODE: task_executor ---")
    # Custom non-LLM pre-processing (simulating task validation)
    request = state["request"]
    word_count = len(request.split())
    complexity = "complex" if word_count > 10 else "simple"
    print(f"  Task complexity: {complexity} ({word_count} words)")

    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a task execution agent. The task has been classified as {complexity}. "
         "Provide a clear, actionable response."),
        ("user", "{request}")
    ])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"request": request, "complexity": complexity})
    return {"response": response}


def route_request(state: CoordinatorState) -> Literal["greeter", "task_executor"]:
    if state["route"] == "greeter":
        return "greeter"
    return "task_executor"


def build_coordinator_graph():
    builder = StateGraph(CoordinatorState)
    builder.add_node("coordinator", coordinator)
    builder.add_node("greeter", greeter)
    builder.add_node("task_executor", task_executor)

    builder.add_edge(START, "coordinator")
    builder.add_conditional_edges(
        "coordinator", route_request,
        {"greeter": "greeter", "task_executor": "task_executor"}
    )
    builder.add_edge("greeter", END)
    builder.add_edge("task_executor", END)

    return builder.compile()


def main():
    print("--- LangGraph Coordinator Example ---")
    graph = build_coordinator_graph()

    requests = [
        "Hello! How are you today?",
        "Summarize the key differences between REST and GraphQL APIs.",
        "Good morning!",
    ]

    for req in requests:
        print(f"\nUser: {req}")
        result = graph.invoke({"request": req})
        print(f"Response: {result['response'][:300]}")


if __name__ == "__main__":
    main()
