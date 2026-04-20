"""
Loop Agent Pattern using LangGraph.

Demonstrates an iterative loop where a processing node works on a task
and a checker node evaluates whether to continue or terminate.
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

MAX_ITERATIONS = 3


class LoopState(TypedDict):
    task: str
    current_draft: str
    iteration: int
    status: str


def processing_step(state: LoopState) -> dict:
    """Processes or refines the task output."""
    iteration = state.get("iteration", 0) + 1
    print(f"--- NODE: processing_step (iteration {iteration}) ---")

    llm = get_llm(temperature=0.7)

    if state.get("current_draft"):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a writer. Improve the following draft. Make it more concise, "
             "clear, and impactful. Only output the improved version.\n\n"
             "Current draft:\n{current_draft}"),
            ("user", "Improve this text about: {task}")
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a writer. Write a brief paragraph on the given topic."),
            ("user", "{task}")
        ])

    chain = prompt | llm | StrOutputParser()
    draft = chain.invoke({"task": state["task"], "current_draft": state.get("current_draft", "")})
    return {"current_draft": draft, "iteration": iteration}


def condition_checker(state: LoopState) -> dict:
    """Evaluates draft quality and decides whether it's complete."""
    print(f"--- NODE: condition_checker (iteration {state['iteration']}) ---")

    if state["iteration"] >= MAX_ITERATIONS:
        print("  Max iterations reached. Marking complete.")
        return {"status": "completed"}

    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Evaluate if the following text is high quality (clear, concise, informative). "
         "Output ONLY 'yes' if it meets all criteria, or 'no' if it needs improvement.\n\n"
         "Text:\n{current_draft}"),
        ("user", "Is this text good enough?")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"current_draft": state["current_draft"]}).strip().lower()

    if "yes" in result:
        print("  Quality check passed. Marking complete.")
        return {"status": "completed"}

    print("  Needs improvement. Continuing loop.")
    return {"status": "in_progress"}


def should_continue(state: LoopState) -> Literal["processing_step", "__end__"]:
    if state["status"] == "completed":
        return "__end__"
    return "processing_step"


def build_loop_graph():
    builder = StateGraph(LoopState)
    builder.add_node("processing_step", processing_step)
    builder.add_node("condition_checker", condition_checker)

    builder.add_edge(START, "processing_step")
    builder.add_edge("processing_step", "condition_checker")
    builder.add_conditional_edges("condition_checker", should_continue)

    return builder.compile()


def main():
    print("--- LangGraph Loop Agent Example ---")
    graph = build_loop_graph()

    result = graph.invoke({
        "task": "Explain how neural networks learn through backpropagation",
        "current_draft": "",
        "iteration": 0,
        "status": "in_progress",
    })

    print(f"\n=== FINAL (after {result['iteration']} iterations) ===")
    print(result["current_draft"])


if __name__ == "__main__":
    main()
