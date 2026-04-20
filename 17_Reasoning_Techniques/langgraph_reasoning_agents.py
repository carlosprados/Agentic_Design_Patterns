"""
Multi-Agent Reasoning Pattern using LangGraph.

Demonstrates a coordinator agent that routes to specialized sub-agents:
a Search Agent for information retrieval and a Code Agent for computation,
to solve complex reasoning problems.
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


class ReasoningState(TypedDict):
    question: str
    agent_type: str
    search_result: str
    code_result: str
    final_answer: str


def classify_question(state: ReasoningState) -> dict:
    """Determines whether the question needs search or computation."""
    print("--- NODE: classify_question ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Classify the question type:\n"
         "- 'search' if it requires factual knowledge or information retrieval\n"
         "- 'code' if it requires calculation, data processing, or code generation\n"
         "Output ONLY one word: 'search' or 'code'."),
        ("user", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    agent_type = chain.invoke({"question": state["question"]}).strip().lower()
    print(f"  Classified as: {agent_type}")
    return {"agent_type": agent_type}


def search_agent(state: ReasoningState) -> dict:
    """Information retrieval specialist."""
    print("--- NODE: search_agent ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an information retrieval specialist. Provide a factual, "
         "well-sourced answer to the question. Include specific details."),
        ("user", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": state["question"]})
    return {"search_result": result}


def code_agent(state: ReasoningState) -> dict:
    """Computation and code specialist."""
    print("--- NODE: code_agent ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a computation specialist. Solve the problem step by step. "
         "Show your reasoning and provide the final answer clearly."),
        ("user", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": state["question"]})
    return {"code_result": result}


def synthesize_answer(state: ReasoningState) -> dict:
    """Produces the final answer from whichever agent responded."""
    print("--- NODE: synthesize_answer ---")
    if state.get("search_result"):
        return {"final_answer": state["search_result"]}
    return {"final_answer": state["code_result"]}


def route_to_agent(state: ReasoningState) -> Literal["search_agent", "code_agent"]:
    if state["agent_type"] == "code":
        return "code_agent"
    return "search_agent"


def build_reasoning_graph():
    builder = StateGraph(ReasoningState)

    builder.add_node("classify_question", classify_question)
    builder.add_node("search_agent", search_agent)
    builder.add_node("code_agent", code_agent)
    builder.add_node("synthesize_answer", synthesize_answer)

    builder.add_edge(START, "classify_question")
    builder.add_conditional_edges(
        "classify_question", route_to_agent,
        {"search_agent": "search_agent", "code_agent": "code_agent"}
    )
    builder.add_edge("search_agent", "synthesize_answer")
    builder.add_edge("code_agent", "synthesize_answer")
    builder.add_edge("synthesize_answer", END)

    return builder.compile()


def main():
    print("--- LangGraph Reasoning Agents Example ---")
    graph = build_reasoning_graph()

    questions = [
        "What are the main differences between TCP and UDP protocols?",
        "Calculate the compound interest on $10,000 at 5% annual rate over 10 years.",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        result = graph.invoke({"question": question})
        print(f"Agent: {result['agent_type']}")
        print(f"A: {result['final_answer'][:400]}")


if __name__ == "__main__":
    main()
