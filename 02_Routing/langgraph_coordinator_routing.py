"""
Coordinator Routing Pattern using LangGraph StateGraph.

Demonstrates a coordinator agent that classifies user intent and routes
to specialized handler nodes via conditional edges.
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


# --- State Definition ---
class RouterState(TypedDict):
    request: str
    route: str
    response: str


# --- Node Implementations ---
def classify_intent(state: RouterState) -> dict:
    """Coordinator node: classifies user intent into a route."""
    print("--- NODE: classify_intent ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Analyze the user's request and determine which specialist should handle it.\n"
         "- If related to booking flights or hotels, output 'booker'.\n"
         "- For general information questions, output 'info'.\n"
         "ONLY output one word: 'booker' or 'info'."),
        ("user", "{request}")
    ])
    chain = prompt | llm | StrOutputParser()
    route = chain.invoke({"request": state["request"]}).strip().lower()
    print(f"  Classified as: {route}")
    return {"route": route}


def booking_node(state: RouterState) -> dict:
    """Specialist node: handles booking-related requests."""
    print("--- NODE: booking_node ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a travel booking assistant. Help the user with their booking request."),
        ("user", "{request}")
    ])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"request": state["request"]})
    return {"response": response}


def info_node(state: RouterState) -> dict:
    """Specialist node: handles general information requests."""
    print("--- NODE: info_node ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledgeable assistant. Answer the user's question concisely."),
        ("user", "{request}")
    ])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"request": state["request"]})
    return {"response": response}


# --- Edge Logic ---
def route_by_intent(state: RouterState) -> Literal["booking_node", "info_node"]:
    """Routes to the appropriate specialist based on classified intent."""
    if state["route"] == "booker":
        return "booking_node"
    return "info_node"


# --- Graph Construction ---
def build_coordinator_graph():
    """Builds and compiles the coordinator routing graph."""
    builder = StateGraph(RouterState)

    builder.add_node("classify_intent", classify_intent)
    builder.add_node("booking_node", booking_node)
    builder.add_node("info_node", info_node)

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {"booking_node": "booking_node", "info_node": "info_node"}
    )
    builder.add_edge("booking_node", END)
    builder.add_edge("info_node", END)

    return builder.compile()


def main():
    print("--- LangGraph Coordinator Routing Example ---")
    graph = build_coordinator_graph()

    requests = [
        "Book me a flight to London next Friday.",
        "What is the capital of Italy?",
        "I need a hotel in Tokyo for 3 nights.",
    ]

    for req in requests:
        print(f"\nUser: {req}")
        result = graph.invoke({"request": req})
        print(f"Route: {result['route']}")
        print(f"Assistant: {result['response'][:200]}...")


if __name__ == "__main__":
    main()
