"""
Exploration and Discovery Pattern using LangGraph.

Demonstrates a multi-agent research laboratory workflow:
PostDoc reviews literature and formulates plans, Reviewers provide
peer review, and Professor synthesizes the final report.
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


class DiscoveryState(TypedDict):
    research_topic: str
    literature_review: str
    experimental_plan: str
    reviews: Annotated[List[str], operator.add]
    final_report: str


def literature_review(state: DiscoveryState) -> dict:
    """PostDoc agent: conducts initial literature review."""
    print("--- NODE: literature_review (PostDoc) ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a postdoctoral researcher. Conduct a brief literature review on "
         "the given topic. Identify 3-4 key papers/findings and gaps in current research."),
        ("user", "Literature review for: {research_topic}")
    ])
    chain = prompt | llm | StrOutputParser()
    review = chain.invoke({"research_topic": state["research_topic"]})
    return {"literature_review": review}


def formulate_plan(state: DiscoveryState) -> dict:
    """PostDoc agent: formulates an experimental plan based on literature."""
    print("--- NODE: formulate_plan (PostDoc) ---")
    llm = get_llm(temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a postdoctoral researcher. Based on the literature review, "
         "formulate a concrete experimental plan with:\n"
         "1. Hypothesis\n"
         "2. Methodology\n"
         "3. Expected outcomes\n"
         "4. Potential challenges\n\n"
         "Literature review:\n{literature_review}"),
        ("user", "Create experimental plan for: {research_topic}")
    ])
    chain = prompt | llm | StrOutputParser()
    plan = chain.invoke({
        "research_topic": state["research_topic"],
        "literature_review": state["literature_review"],
    })
    return {"experimental_plan": plan}


def reviewer_experimental(state: DiscoveryState) -> dict:
    """Reviewer 1: harsh focus on experimental rigor."""
    print("--- NODE: reviewer_experimental ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a harsh peer reviewer focused on experimental methodology. "
         "Critique the experimental plan for methodological flaws, missing controls, "
         "and statistical validity concerns. Be specific.\n\n"
         "Plan:\n{experimental_plan}"),
        ("user", "Review this research plan.")
    ])
    chain = prompt | llm | StrOutputParser()
    review = chain.invoke({"experimental_plan": state["experimental_plan"]})
    return {"reviews": [f"[Reviewer 1 - Experimental Rigor]\n{review}"]}


def reviewer_impact(state: DiscoveryState) -> dict:
    """Reviewer 2: focus on impact and significance."""
    print("--- NODE: reviewer_impact ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a peer reviewer focused on research impact and significance. "
         "Evaluate whether this research would make a meaningful contribution "
         "to the field. Assess novelty and practical implications.\n\n"
         "Plan:\n{experimental_plan}"),
        ("user", "Review this research plan for impact.")
    ])
    chain = prompt | llm | StrOutputParser()
    review = chain.invoke({"experimental_plan": state["experimental_plan"]})
    return {"reviews": [f"[Reviewer 2 - Impact & Significance]\n{review}"]}


def reviewer_novelty(state: DiscoveryState) -> dict:
    """Reviewer 3: focus on novelty and originality."""
    print("--- NODE: reviewer_novelty ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a peer reviewer focused on novelty and originality. "
         "Assess whether the approach is truly novel or incremental. "
         "Suggest how to differentiate from existing work.\n\n"
         "Plan:\n{experimental_plan}"),
        ("user", "Review this research plan for novelty.")
    ])
    chain = prompt | llm | StrOutputParser()
    review = chain.invoke({"experimental_plan": state["experimental_plan"]})
    return {"reviews": [f"[Reviewer 3 - Novelty]\n{review}"]}


def professor_synthesis(state: DiscoveryState) -> dict:
    """Professor agent: synthesizes everything into a final assessment."""
    print("--- NODE: professor_synthesis (Professor) ---")
    llm = get_llm(temperature=0.3)
    reviews_text = "\n\n".join(state["reviews"])
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior professor and PI. Synthesize the peer reviews into a "
         "final assessment with:\n"
         "1. Overall recommendation (accept/revise/reject)\n"
         "2. Key strengths\n"
         "3. Critical issues to address\n"
         "4. Suggested next steps\n\n"
         "Literature review:\n{literature_review}\n\n"
         "Experimental plan:\n{experimental_plan}\n\n"
         "Peer reviews:\n{reviews}"),
        ("user", "Provide final assessment for: {research_topic}")
    ])
    chain = prompt | llm | StrOutputParser()
    report = chain.invoke({
        "research_topic": state["research_topic"],
        "literature_review": state["literature_review"],
        "experimental_plan": state["experimental_plan"],
        "reviews": reviews_text,
    })
    return {"final_report": report}


def build_discovery_graph():
    builder = StateGraph(DiscoveryState)

    builder.add_node("literature_review", literature_review)
    builder.add_node("formulate_plan", formulate_plan)
    builder.add_node("reviewer_experimental", reviewer_experimental)
    builder.add_node("reviewer_impact", reviewer_impact)
    builder.add_node("reviewer_novelty", reviewer_novelty)
    builder.add_node("professor_synthesis", professor_synthesis)

    # Sequential: literature → plan
    builder.add_edge(START, "literature_review")
    builder.add_edge("literature_review", "formulate_plan")

    # Parallel: all three reviewers
    builder.add_edge("formulate_plan", "reviewer_experimental")
    builder.add_edge("formulate_plan", "reviewer_impact")
    builder.add_edge("formulate_plan", "reviewer_novelty")

    # Fan-in: all reviewers → professor
    builder.add_edge("reviewer_experimental", "professor_synthesis")
    builder.add_edge("reviewer_impact", "professor_synthesis")
    builder.add_edge("reviewer_novelty", "professor_synthesis")

    builder.add_edge("professor_synthesis", END)

    return builder.compile()


def main():
    print("--- LangGraph Discovery Workflow Example ---")
    graph = build_discovery_graph()

    result = graph.invoke({
        "research_topic": "Using large language models for automated scientific hypothesis generation",
        "reviews": [],
    })

    print("\n=== LITERATURE REVIEW ===")
    print(result["literature_review"][:400])
    print("\n=== EXPERIMENTAL PLAN ===")
    print(result["experimental_plan"][:400])
    print("\n=== PEER REVIEWS ===")
    for review in result["reviews"]:
        print(f"\n{review[:300]}")
    print("\n=== PROFESSOR'S FINAL REPORT ===")
    print(result["final_report"][:500])


if __name__ == "__main__":
    main()
