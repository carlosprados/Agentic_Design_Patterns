"""
Multi-Agent Blog Creation Pattern using LangGraph.

Demonstrates a two-agent workflow: a Researcher produces findings,
then a Writer uses those findings to create a blog post.
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


class BlogState(TypedDict):
    topic: str
    research: str
    blog_post: str


def researcher(state: BlogState) -> dict:
    """Research agent: gathers information and key points about the topic."""
    print("--- NODE: researcher ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a thorough researcher. Investigate the given topic and provide:\n"
         "1. Key facts and statistics\n"
         "2. Current trends\n"
         "3. Expert opinions or notable viewpoints\n"
         "4. Potential impact and future outlook\n"
         "Be specific and factual."),
        ("user", "Research: {topic}")
    ])
    chain = prompt | llm | StrOutputParser()
    research = chain.invoke({"topic": state["topic"]})
    return {"research": research}


def writer(state: BlogState) -> dict:
    """Writer agent: creates a blog post based on the research findings."""
    print("--- NODE: writer ---")
    llm = get_llm(temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a skilled blog writer. Write an engaging blog post based on the "
         "following research. Include a catchy title, introduction, body sections, "
         "and conclusion. Make it informative but accessible.\n\n"
         "Research findings:\n{research}"),
        ("user", "Write a blog post about: {topic}")
    ])
    chain = prompt | llm | StrOutputParser()
    blog = chain.invoke({"topic": state["topic"], "research": state["research"]})
    return {"blog_post": blog}


def build_blog_graph():
    builder = StateGraph(BlogState)
    builder.add_node("researcher", researcher)
    builder.add_node("writer", writer)
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", END)
    return builder.compile()


def main():
    print("--- LangGraph Multi-Agent Blog Example ---")
    graph = build_blog_graph()

    result = graph.invoke({"topic": "The rise of AI coding assistants and their impact on developers"})

    print("\n=== RESEARCH ===")
    print(result["research"][:500])
    print("\n=== BLOG POST ===")
    print(result["blog_post"][:800])


if __name__ == "__main__":
    main()
