"""
Reflection Pipeline Pattern using LangGraph StateGraph.

Demonstrates a two-stage reflection pipeline: a DraftWriter generates
content and a FactChecker reviews it, connected via graph state.
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


# --- State Definition ---
class ReflectionState(TypedDict):
    topic: str
    draft_text: str
    review_output: str


# --- Node Implementations ---
def draft_writer(state: ReflectionState) -> dict:
    """Generates an initial draft on the given topic."""
    print("--- NODE: draft_writer ---")
    llm = get_llm(temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a technical writer. Write a concise (3-4 paragraph) article "
         "on the given topic. Focus on accuracy and clarity."),
        ("user", "Write about: {topic}")
    ])
    chain = prompt | llm | StrOutputParser()
    draft = chain.invoke({"topic": state["topic"]})
    return {"draft_text": draft}


def fact_checker(state: ReflectionState) -> dict:
    """Reviews the draft for factual accuracy and suggests improvements."""
    print("--- NODE: fact_checker ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a fact-checker and editor. Review the following draft for:\n"
         "1. Factual accuracy\n"
         "2. Logical consistency\n"
         "3. Missing important points\n"
         "4. Suggestions for improvement\n\n"
         "Provide a structured review with specific feedback."),
        ("user", "Draft to review:\n\n{draft_text}")
    ])
    chain = prompt | llm | StrOutputParser()
    review = chain.invoke({"draft_text": state["draft_text"]})
    return {"review_output": review}


# --- Graph Construction ---
def build_reflection_pipeline():
    """Builds a sequential draft → review pipeline."""
    builder = StateGraph(ReflectionState)

    builder.add_node("draft_writer", draft_writer)
    builder.add_node("fact_checker", fact_checker)

    builder.add_edge(START, "draft_writer")
    builder.add_edge("draft_writer", "fact_checker")
    builder.add_edge("fact_checker", END)

    return builder.compile()


def main():
    print("--- LangGraph Reflection Pipeline Example ---")
    graph = build_reflection_pipeline()

    result = graph.invoke({
        "topic": "The role of quantum computing in breaking modern encryption"
    })

    print("\n=== DRAFT ===")
    print(result["draft_text"])
    print("\n=== REVIEW ===")
    print(result["review_output"])


if __name__ == "__main__":
    main()
