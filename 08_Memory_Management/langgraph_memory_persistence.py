"""
Memory Persistence Pattern using LangGraph.

Demonstrates LangGraph's built-in checkpointing with MemorySaver for
conversation persistence across multiple turns and session resumption.
"""

import operator
import sys
from pathlib import Path
from typing import TypedDict, Annotated

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from shared.llm import get_llm
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


def chat_node(state: ConversationState) -> dict:
    """Processes the conversation with full message history."""
    print("--- NODE: chat ---")
    llm = get_llm(temperature=0.7)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def build_persistent_chat():
    """Builds a chat graph with memory persistence via checkpointing."""
    builder = StateGraph(ConversationState)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)

    # MemorySaver provides in-memory checkpointing (use SqliteSaver for disk persistence)
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


def main():
    print("--- LangGraph Memory Persistence Example ---")
    graph = build_persistent_chat()

    # Thread ID groups messages into a single conversation session
    config = {"configurable": {"thread_id": "session-001"}}

    # Turn 1
    print("\n=== Turn 1 ===")
    result = graph.invoke(
        {"messages": [HumanMessage(content="Hi! My name is Charlie and I'm a CTO.")]},
        config
    )
    print(f"AI: {result['messages'][-1].content[:300]}")

    # Turn 2 — the model remembers Turn 1 via checkpointed state
    print("\n=== Turn 2 ===")
    result = graph.invoke(
        {"messages": [HumanMessage(content="What's my name and role?")]},
        config
    )
    print(f"AI: {result['messages'][-1].content[:300]}")

    # Turn 3 — different thread = fresh session
    print("\n=== Turn 3 (different session) ===")
    config2 = {"configurable": {"thread_id": "session-002"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content="Do you know my name?")]},
        config2
    )
    print(f"AI: {result['messages'][-1].content[:300]}")


if __name__ == "__main__":
    main()
