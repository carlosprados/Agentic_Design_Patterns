"""
Human-in-the-Loop Pattern using LangGraph.

Demonstrates personalization injection via state, agent troubleshooting,
and a human approval step using LangGraph's interrupt mechanism.
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
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


class SupportState(TypedDict):
    customer_name: str
    customer_tier: str
    issue: str
    diagnosis: str
    needs_escalation: bool
    human_approved: bool
    resolution: str


def personalize_and_diagnose(state: SupportState) -> dict:
    """Troubleshoots the issue using customer context from state."""
    print("--- NODE: personalize_and_diagnose ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a support agent. The customer context is:\n"
         "- Name: {customer_name}\n"
         "- Tier: {customer_tier}\n\n"
         "Diagnose the issue and determine if it needs human escalation.\n"
         "If the issue is complex, sensitive, or the customer is frustrated, "
         "recommend escalation.\n\n"
         "Output format:\n"
         "DIAGNOSIS: <your diagnosis>\n"
         "ESCALATE: <yes or no>"),
        ("user", "Customer issue: {issue}")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "customer_name": state["customer_name"],
        "customer_tier": state["customer_tier"],
        "issue": state["issue"],
    })

    needs_escalation = "ESCALATE: yes" in result.lower() or "escalate: yes" in result.lower()
    diagnosis = result.split("DIAGNOSIS:")[-1].split("ESCALATE:")[0].strip() if "DIAGNOSIS:" in result else result

    print(f"  Needs escalation: {needs_escalation}")
    return {"diagnosis": diagnosis, "needs_escalation": needs_escalation}


def request_human_approval(state: SupportState) -> dict:
    """Pauses execution for human review before escalation."""
    print("--- NODE: request_human_approval ---")
    print(f"  AWAITING HUMAN APPROVAL for escalation of: {state['customer_name']}")
    print(f"  Diagnosis: {state['diagnosis'][:200]}")
    # In production, this would use LangGraph's interrupt() for real human input.
    # For demo purposes, we auto-approve.
    print("  [Auto-approved for demo]")
    return {"human_approved": True}


def resolve_directly(state: SupportState) -> dict:
    """Resolves the issue directly without human intervention."""
    print("--- NODE: resolve_directly ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a support agent. Provide a friendly resolution to the customer. "
         "Address them by name: {customer_name}. Their tier: {customer_tier}.\n"
         "Diagnosis: {diagnosis}"),
        ("user", "Resolve: {issue}")
    ])
    chain = prompt | llm | StrOutputParser()
    resolution = chain.invoke({
        "customer_name": state["customer_name"],
        "customer_tier": state["customer_tier"],
        "diagnosis": state["diagnosis"],
        "issue": state["issue"],
    })
    return {"resolution": resolution}


def escalate_to_human(state: SupportState) -> dict:
    """Escalates to a human agent after approval."""
    print("--- NODE: escalate_to_human ---")
    resolution = (
        f"Dear {state['customer_name']}, as a {state['customer_tier']} customer, "
        f"your issue has been escalated to a senior specialist. "
        f"Diagnosis summary: {state['diagnosis'][:150]}... "
        f"A human agent will follow up shortly."
    )
    return {"resolution": resolution}


def route_after_diagnosis(state: SupportState) -> Literal["request_human_approval", "resolve_directly"]:
    if state["needs_escalation"]:
        return "request_human_approval"
    return "resolve_directly"


def build_support_graph():
    builder = StateGraph(SupportState)

    builder.add_node("personalize_and_diagnose", personalize_and_diagnose)
    builder.add_node("request_human_approval", request_human_approval)
    builder.add_node("resolve_directly", resolve_directly)
    builder.add_node("escalate_to_human", escalate_to_human)

    builder.add_edge(START, "personalize_and_diagnose")
    builder.add_conditional_edges(
        "personalize_and_diagnose", route_after_diagnosis,
        {"request_human_approval": "request_human_approval", "resolve_directly": "resolve_directly"}
    )
    builder.add_edge("request_human_approval", "escalate_to_human")
    builder.add_edge("resolve_directly", END)
    builder.add_edge("escalate_to_human", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


def main():
    print("--- LangGraph Human-in-the-Loop Example ---")
    graph = build_support_graph()

    # Case 1: Simple issue — resolved directly
    print("\n=== Case 1: Simple Issue ===")
    result = graph.invoke({
        "customer_name": "Alice",
        "customer_tier": "Standard",
        "issue": "I forgot my password and need to reset it.",
        "needs_escalation": False,
        "human_approved": False,
    }, {"configurable": {"thread_id": "case-1"}})
    print(f"Resolution: {result['resolution'][:300]}")

    # Case 2: Complex issue — needs escalation
    print("\n=== Case 2: Complex Issue (Escalation) ===")
    result = graph.invoke({
        "customer_name": "Bob",
        "customer_tier": "Enterprise",
        "issue": "I've been charged incorrectly for 3 months and I'm extremely frustrated. "
                 "Your billing system has a serious bug and I want a full refund.",
        "needs_escalation": False,
        "human_approved": False,
    }, {"configurable": {"thread_id": "case-2"}})
    print(f"Resolution: {result['resolution'][:300]}")


if __name__ == "__main__":
    main()
