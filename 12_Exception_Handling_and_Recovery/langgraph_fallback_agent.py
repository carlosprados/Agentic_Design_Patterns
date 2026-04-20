"""
Fallback Agent Pattern using LangGraph.

Demonstrates graceful degradation: a primary handler attempts a precise
operation; if it fails, a fallback handler provides an alternative response.
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


class FallbackState(TypedDict):
    query: str
    primary_failed: bool
    location_result: str
    response: str


def primary_handler(state: FallbackState) -> dict:
    """Attempts precise location lookup (simulates failure for certain queries)."""
    print("--- NODE: primary_handler ---")

    # Simulate a service that only knows specific locations
    known_locations = {"paris": "Paris, France: 48.8566N, 2.3522E - The City of Light",
                       "tokyo": "Tokyo, Japan: 35.6762N, 139.6503E - The world's largest metropolis"}

    query_lower = state["query"].lower()
    for key, info in known_locations.items():
        if key in query_lower:
            print("  Primary lookup succeeded.")
            return {"primary_failed": False, "location_result": info}

    print("  Primary lookup failed. Triggering fallback.")
    return {"primary_failed": True, "location_result": ""}


def fallback_handler(state: FallbackState) -> dict:
    """Provides a general response when the primary handler fails."""
    print("--- NODE: fallback_handler ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "The precise location service is unavailable for this query. "
         "Provide general information about the location from your knowledge."),
        ("user", "{query}")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"query": state["query"]})
    return {"location_result": result}


def response_formatter(state: FallbackState) -> dict:
    """Formats the final response regardless of which handler produced it."""
    print("--- NODE: response_formatter ---")
    source = "fallback service" if state["primary_failed"] else "primary service"
    response = f"[Source: {source}]\n{state['location_result']}"
    return {"response": response}


def route_after_primary(state: FallbackState) -> Literal["fallback_handler", "response_formatter"]:
    if state["primary_failed"]:
        return "fallback_handler"
    return "response_formatter"


def build_fallback_graph():
    builder = StateGraph(FallbackState)

    builder.add_node("primary_handler", primary_handler)
    builder.add_node("fallback_handler", fallback_handler)
    builder.add_node("response_formatter", response_formatter)

    builder.add_edge(START, "primary_handler")
    builder.add_conditional_edges(
        "primary_handler", route_after_primary,
        {"fallback_handler": "fallback_handler", "response_formatter": "response_formatter"}
    )
    builder.add_edge("fallback_handler", "response_formatter")
    builder.add_edge("response_formatter", END)

    return builder.compile()


def main():
    print("--- LangGraph Fallback Agent Example ---")
    graph = build_fallback_graph()

    queries = [
        "Tell me about Paris",
        "What can you tell me about Reykjavik, Iceland?",
        "Information about Tokyo",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        result = graph.invoke({"query": query, "primary_failed": False})
        print(result["response"][:300])


if __name__ == "__main__":
    main()
