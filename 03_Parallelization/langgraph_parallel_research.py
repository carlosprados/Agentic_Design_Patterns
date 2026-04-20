"""
Parallel Research Pattern using LangGraph StateGraph.

Demonstrates fan-out/fan-in: multiple researcher nodes execute in parallel,
then a synthesis node aggregates all findings.
"""

import operator
import sys
from pathlib import Path
from typing import TypedDict, Annotated, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from shared.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

load_dotenv()


# --- State Definition ---
class ResearchState(TypedDict):
    topic: str
    findings: Annotated[List[str], operator.add]
    synthesis: str


# --- Node Implementations ---
def research_renewable_energy(state: ResearchState) -> dict:
    """Researcher 1: investigates renewable energy aspects of the topic."""
    print("--- NODE: research_renewable_energy ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a renewable energy researcher. Provide a brief (2-3 sentence) "
                   "analysis of the renewable energy implications of the given topic."),
        ("user", "{topic}")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"topic": state["topic"]})
    return {"findings": [f"[Renewable Energy] {result}"]}


def research_electric_vehicles(state: ResearchState) -> dict:
    """Researcher 2: investigates electric vehicle aspects of the topic."""
    print("--- NODE: research_electric_vehicles ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an electric vehicle industry analyst. Provide a brief (2-3 sentence) "
                   "analysis of the EV implications of the given topic."),
        ("user", "{topic}")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"topic": state["topic"]})
    return {"findings": [f"[Electric Vehicles] {result}"]}


def research_carbon_capture(state: ResearchState) -> dict:
    """Researcher 3: investigates carbon capture aspects of the topic."""
    print("--- NODE: research_carbon_capture ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a carbon capture technology specialist. Provide a brief (2-3 sentence) "
                   "analysis of the carbon capture implications of the given topic."),
        ("user", "{topic}")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"topic": state["topic"]})
    return {"findings": [f"[Carbon Capture] {result}"]}


def synthesize(state: ResearchState) -> dict:
    """Synthesis node: aggregates findings from all researchers into a report."""
    print("--- NODE: synthesize ---")
    llm = get_llm(temperature=0)
    findings_text = "\n\n".join(state["findings"])
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior research analyst. Synthesize the following research findings "
         "into a cohesive 1-paragraph executive summary.\n\nFindings:\n{findings}"),
        ("user", "Create the synthesis for the topic: {topic}")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"topic": state["topic"], "findings": findings_text})
    return {"synthesis": result}


# --- Graph Construction ---
def build_parallel_research_graph():
    """Builds a graph with 3 parallel researcher branches merging into synthesis."""
    builder = StateGraph(ResearchState)

    builder.add_node("research_renewable_energy", research_renewable_energy)
    builder.add_node("research_electric_vehicles", research_electric_vehicles)
    builder.add_node("research_carbon_capture", research_carbon_capture)
    builder.add_node("synthesize", synthesize)

    # Fan-out: START → all three researchers in parallel
    builder.add_edge(START, "research_renewable_energy")
    builder.add_edge(START, "research_electric_vehicles")
    builder.add_edge(START, "research_carbon_capture")

    # Fan-in: all researchers → synthesize
    builder.add_edge("research_renewable_energy", "synthesize")
    builder.add_edge("research_electric_vehicles", "synthesize")
    builder.add_edge("research_carbon_capture", "synthesize")

    builder.add_edge("synthesize", END)

    return builder.compile()


def main():
    print("--- LangGraph Parallel Research Example ---")
    graph = build_parallel_research_graph()

    result = graph.invoke({
        "topic": "The impact of government subsidies on clean technology adoption",
        "findings": [],
    })

    print("\n=== INDIVIDUAL FINDINGS ===")
    for finding in result["findings"]:
        print(f"\n{finding}")

    print("\n=== SYNTHESIS ===")
    print(result["synthesis"])


if __name__ == "__main__":
    main()
