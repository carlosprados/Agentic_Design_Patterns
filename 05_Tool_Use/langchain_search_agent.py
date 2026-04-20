"""
Search Agent Pattern using LangChain.

Demonstrates an agent that uses a search tool to answer questions
by retrieving information from external sources.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from shared.llm import get_llm
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()


# Simulated search tool (replace with GoogleSearchAPIWrapper for production)
KNOWLEDGE_BASE = {
    "python": "Python is a high-level programming language created by Guido van Rossum in 1991. "
              "It emphasizes code readability and supports multiple paradigms.",
    "langchain": "LangChain is a framework for developing applications powered by LLMs. "
                 "It provides tools for chains, agents, and retrieval-augmented generation.",
    "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
                 "It extends LangChain with graph-based workflow orchestration.",
    "gemini": "Gemini is Google's multimodal AI model family. Gemini 2.5 Flash is optimized "
              "for speed and cost efficiency while maintaining strong performance.",
}


@tool
def search_information(query: str) -> str:
    """Searches a knowledge base for information about a topic.
    Returns relevant information if found, or indicates no results.
    """
    query_lower = query.lower()
    results = []
    for key, value in KNOWLEDGE_BASE.items():
        if key in query_lower:
            results.append(value)
    if results:
        return " ".join(results)
    return f"No specific information found for: {query}. Try a more specific query."


def main():
    print("--- LangChain Search Agent Example ---")

    llm = get_llm(temperature=0)
    agent = create_react_agent(llm, [search_information])

    queries = [
        "What is LangGraph and how does it relate to LangChain?",
        "Tell me about the Python programming language.",
        "What is Gemini?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        final_message = result["messages"][-1]
        print(f"Answer: {final_message.content[:300]}")


if __name__ == "__main__":
    main()
