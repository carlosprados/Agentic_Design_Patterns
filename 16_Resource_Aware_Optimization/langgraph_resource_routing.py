"""
Resource-Aware Routing Pattern using LangGraph.

Demonstrates routing queries to different model configurations based on
complexity analysis, optimizing for cost and latency.
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


class ResourceState(TypedDict):
    query: str
    complexity: str
    word_count: int
    response: str
    model_used: str


def analyze_complexity(state: ResourceState) -> dict:
    """Analyzes query complexity using a simple heuristic (word count)."""
    print("--- NODE: analyze_complexity ---")
    word_count = len(state["query"].split())
    complexity = "complex" if word_count > 15 else "simple"
    print(f"  Words: {word_count}, Complexity: {complexity}")
    return {"complexity": complexity, "word_count": word_count}


def fast_model_handler(state: ResourceState) -> dict:
    """Handles simple queries with a fast, cost-efficient model configuration."""
    print("--- NODE: fast_model_handler (low temperature, concise) ---")
    llm = get_llm(temperature=0, max_output_tokens=256)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise assistant. Answer briefly and directly."),
        ("user", "{query}")
    ])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"query": state["query"]})
    return {"response": response, "model_used": "gemini-2.5-flash (fast/concise)"}


def powerful_model_handler(state: ResourceState) -> dict:
    """Handles complex queries with more processing power."""
    print("--- NODE: powerful_model_handler (higher temperature, detailed) ---")
    llm = get_llm(temperature=0.3, max_output_tokens=1024)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a thorough assistant. Provide a detailed, well-structured response "
         "with analysis and reasoning."),
        ("user", "{query}")
    ])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"query": state["query"]})
    return {"response": response, "model_used": "gemini-2.5-flash (powerful/detailed)"}


def route_by_complexity(state: ResourceState) -> Literal["fast_model_handler", "powerful_model_handler"]:
    if state["complexity"] == "simple":
        return "fast_model_handler"
    return "powerful_model_handler"


def build_resource_routing_graph():
    builder = StateGraph(ResourceState)

    builder.add_node("analyze_complexity", analyze_complexity)
    builder.add_node("fast_model_handler", fast_model_handler)
    builder.add_node("powerful_model_handler", powerful_model_handler)

    builder.add_edge(START, "analyze_complexity")
    builder.add_conditional_edges(
        "analyze_complexity", route_by_complexity,
        {"fast_model_handler": "fast_model_handler", "powerful_model_handler": "powerful_model_handler"}
    )
    builder.add_edge("fast_model_handler", END)
    builder.add_edge("powerful_model_handler", END)

    return builder.compile()


def main():
    print("--- LangGraph Resource-Aware Routing Example ---")
    graph = build_resource_routing_graph()

    queries = [
        "What is Python?",
        "Explain the key architectural differences between microservices and monolithic "
        "applications, including their trade-offs for scalability, team organization, "
        "deployment complexity, and data consistency.",
    ]

    for query in queries:
        print(f"\nQuery: {query[:80]}...")
        result = graph.invoke({"query": query})
        print(f"Model used: {result['model_used']}")
        print(f"Response: {result['response'][:300]}")


if __name__ == "__main__":
    main()
