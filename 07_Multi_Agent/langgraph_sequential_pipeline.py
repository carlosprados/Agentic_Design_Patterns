"""
Sequential Pipeline Pattern using LangGraph.

Demonstrates a two-stage pipeline where a data fetcher node passes
its output to a processor node via shared graph state.
"""

import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from shared.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

load_dotenv()


class PipelineState(TypedDict):
    topic: str
    raw_data: str
    analysis: str


def fetch_data(state: PipelineState) -> dict:
    """Step 1: Fetches raw information about the topic."""
    print("--- NODE: fetch_data ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data gathering agent. Provide a factual summary of key "
                   "data points about the given topic. Be specific with numbers and facts."),
        ("user", "Gather data about: {topic}")
    ])
    chain = prompt | llm | StrOutputParser()
    data = chain.invoke({"topic": state["topic"]})
    return {"raw_data": data}


def process_data(state: PipelineState) -> dict:
    """Step 2: Analyzes the raw data from step 1."""
    print("--- NODE: process_data ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a data analyst. Analyze the following raw data and provide:\n"
         "1. Key trends identified\n"
         "2. Notable patterns\n"
         "3. Brief conclusions\n\n"
         "Raw data:\n{raw_data}"),
        ("user", "Analyze the data about: {topic}")
    ])
    chain = prompt | llm | StrOutputParser()
    analysis = chain.invoke({"topic": state["topic"], "raw_data": state["raw_data"]})
    return {"analysis": analysis}


def build_sequential_pipeline():
    builder = StateGraph(PipelineState)
    builder.add_node("fetch_data", fetch_data)
    builder.add_node("process_data", process_data)
    builder.add_edge(START, "fetch_data")
    builder.add_edge("fetch_data", "process_data")
    builder.add_edge("process_data", END)
    return builder.compile()


def main():
    print("--- LangGraph Sequential Pipeline Example ---")
    graph = build_sequential_pipeline()
    result = graph.invoke({"topic": "Global electric vehicle adoption in 2025"})

    print("\n=== RAW DATA ===")
    print(result["raw_data"][:500])
    print("\n=== ANALYSIS ===")
    print(result["analysis"][:500])


if __name__ == "__main__":
    main()
