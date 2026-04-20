"""
Inter-Agent Communication Pattern using LangGraph.

Demonstrates two independent sub-graphs (CalendarAgent and TaskManager)
communicating through a coordinator graph that passes structured messages
between them via shared state.
"""

import sys
from pathlib import Path
from typing import TypedDict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from shared.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

load_dotenv()


# --- Sub-graph 1: Calendar Agent ---
class CalendarState(TypedDict):
    request: str
    calendar_response: str


def calendar_handler(state: CalendarState) -> dict:
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a calendar management assistant. Handle scheduling requests. "
                   "Respond with what actions you would take (create/modify/check events)."),
        ("user", "{request}")
    ])
    chain = prompt | llm | StrOutputParser()
    return {"calendar_response": chain.invoke({"request": state["request"]})}


def build_calendar_subgraph():
    builder = StateGraph(CalendarState)
    builder.add_node("calendar_handler", calendar_handler)
    builder.add_edge(START, "calendar_handler")
    builder.add_edge("calendar_handler", END)
    return builder.compile()


# --- Sub-graph 2: Task Manager Agent ---
class TaskState(TypedDict):
    request: str
    task_response: str


def task_handler(state: TaskState) -> dict:
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a task management assistant. Handle task creation, "
                   "prioritization, and status tracking requests."),
        ("user", "{request}")
    ])
    chain = prompt | llm | StrOutputParser()
    return {"task_response": chain.invoke({"request": state["request"]})}


def build_task_subgraph():
    builder = StateGraph(TaskState)
    builder.add_node("task_handler", task_handler)
    builder.add_edge(START, "task_handler")
    builder.add_edge("task_handler", END)
    return builder.compile()


# --- Coordinator: routes and passes messages between sub-graphs ---
calendar_graph = build_calendar_subgraph()
task_graph = build_task_subgraph()


class CoordinatorState(TypedDict):
    user_request: str
    calendar_result: str
    task_result: str
    final_response: str


def dispatch_to_calendar(state: CoordinatorState) -> dict:
    """Sends the request to the Calendar sub-graph."""
    print("--- NODE: dispatch_to_calendar ---")
    result = calendar_graph.invoke({"request": state["user_request"]})
    return {"calendar_result": result["calendar_response"]}


def dispatch_to_tasks(state: CoordinatorState) -> dict:
    """Sends the request to the Task Manager sub-graph."""
    print("--- NODE: dispatch_to_tasks ---")
    result = task_graph.invoke({"request": state["user_request"]})
    return {"task_result": result["task_response"]}


def synthesize_responses(state: CoordinatorState) -> dict:
    """Combines responses from both sub-agents into a final answer."""
    print("--- NODE: synthesize_responses ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a coordinator. Synthesize the following agent responses into a "
         "unified, actionable answer for the user.\n\n"
         "Calendar Agent said:\n{calendar_result}\n\n"
         "Task Manager said:\n{task_result}"),
        ("user", "Original request: {user_request}")
    ])
    chain = prompt | llm | StrOutputParser()
    final = chain.invoke({
        "calendar_result": state["calendar_result"],
        "task_result": state["task_result"],
        "user_request": state["user_request"],
    })
    return {"final_response": final}


def build_coordinator_graph():
    builder = StateGraph(CoordinatorState)

    builder.add_node("dispatch_to_calendar", dispatch_to_calendar)
    builder.add_node("dispatch_to_tasks", dispatch_to_tasks)
    builder.add_node("synthesize_responses", synthesize_responses)

    # Both sub-agents run in parallel
    builder.add_edge(START, "dispatch_to_calendar")
    builder.add_edge(START, "dispatch_to_tasks")

    # Both feed into synthesis
    builder.add_edge("dispatch_to_calendar", "synthesize_responses")
    builder.add_edge("dispatch_to_tasks", "synthesize_responses")

    builder.add_edge("synthesize_responses", END)

    return builder.compile()


def main():
    print("--- LangGraph Inter-Agent Communication Example ---")
    graph = build_coordinator_graph()

    result = graph.invoke({
        "user_request": "I have a project deadline next Friday. Schedule a 2-hour focus block "
                        "each day this week and create tasks for the deliverables.",
        "calendar_result": "",
        "task_result": "",
        "final_response": "",
    })

    print("\n=== CALENDAR AGENT ===")
    print(result["calendar_result"][:300])
    print("\n=== TASK MANAGER ===")
    print(result["task_result"][:300])
    print("\n=== COORDINATOR SYNTHESIS ===")
    print(result["final_response"][:400])


if __name__ == "__main__":
    main()
