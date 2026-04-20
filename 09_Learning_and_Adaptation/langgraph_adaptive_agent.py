"""
Adaptive Agent Pattern using LangGraph.

Demonstrates an agent that evaluates its own output quality and
iteratively adapts its approach until the result meets a quality threshold.
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

MAX_ADAPTATIONS = 3


class AdaptiveState(TypedDict):
    task: str
    output: str
    score: float
    feedback: str
    iteration: int
    strategy: str


def performer(state: AdaptiveState) -> dict:
    """Generates output based on the current strategy and any feedback."""
    iteration = state.get("iteration", 0) + 1
    print(f"--- NODE: performer (iteration {iteration}) ---")

    llm = get_llm(temperature=0.7)

    strategy = state.get("strategy", "default")
    feedback = state.get("feedback", "")

    system_msg = (
        f"You are an adaptive agent. Current strategy: {strategy}.\n"
        "Produce the best possible output for the given task."
    )
    if feedback:
        system_msg += f"\n\nPrevious feedback to incorporate:\n{feedback}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", "{task}")
    ])
    chain = prompt | llm | StrOutputParser()
    output = chain.invoke({"task": state["task"]})
    return {"output": output, "iteration": iteration}


def evaluator(state: AdaptiveState) -> dict:
    """Evaluates the output quality and provides improvement feedback."""
    print(f"--- NODE: evaluator (iteration {state['iteration']}) ---")

    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Evaluate the following output on a scale of 1-10 for quality, accuracy, "
         "and completeness. Output EXACTLY in this format:\n"
         "SCORE: <number>\n"
         "FEEDBACK: <specific improvement suggestions>\n"
         "STRATEGY: <recommended approach for next attempt>"),
        ("user", "Task: {task}\n\nOutput to evaluate:\n{output}")
    ])
    chain = prompt | llm | StrOutputParser()
    evaluation = chain.invoke({"task": state["task"], "output": state["output"]})

    # Parse score from evaluation
    score = 5.0
    feedback = evaluation
    strategy = "refined"
    for line in evaluation.split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = float(line.split(":")[1].strip().split("/")[0])
            except (ValueError, IndexError):
                pass
        elif line.startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()
        elif line.startswith("STRATEGY:"):
            strategy = line.split(":", 1)[1].strip()

    print(f"  Score: {score}/10")
    return {"score": score, "feedback": feedback, "strategy": strategy}


def should_adapt(state: AdaptiveState) -> Literal["performer", "__end__"]:
    """Decides whether to adapt again or accept the current output."""
    if state["score"] >= 8.0:
        print(f"  Quality threshold met ({state['score']}/10). Accepting output.")
        return "__end__"
    if state["iteration"] >= MAX_ADAPTATIONS:
        print(f"  Max adaptations reached. Accepting best output.")
        return "__end__"
    print(f"  Score {state['score']}/10 below threshold. Adapting...")
    return "performer"


def build_adaptive_graph():
    builder = StateGraph(AdaptiveState)
    builder.add_node("performer", performer)
    builder.add_node("evaluator", evaluator)

    builder.add_edge(START, "performer")
    builder.add_edge("performer", "evaluator")
    builder.add_conditional_edges("evaluator", should_adapt)

    return builder.compile()


def main():
    print("--- LangGraph Adaptive Agent Example ---")
    graph = build_adaptive_graph()

    result = graph.invoke({
        "task": "Write a haiku about machine learning that is technically accurate and poetic.",
        "output": "",
        "score": 0.0,
        "feedback": "",
        "iteration": 0,
        "strategy": "default",
    })

    print(f"\n=== FINAL OUTPUT (score: {result['score']}/10, iterations: {result['iteration']}) ===")
    print(result["output"])


if __name__ == "__main__":
    main()
