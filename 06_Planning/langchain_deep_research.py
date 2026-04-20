"""
Deep Research Pattern using LangChain.

Demonstrates a multi-step research pipeline that generates search queries,
simulates retrieval, and synthesizes findings into a comprehensive report.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from shared.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


def setup_deep_research_chain():
    """Builds a research chain that plans queries, gathers info, and synthesizes."""
    llm = get_llm(temperature=0)

    # Stage 1: Generate research queries
    query_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research assistant. Given a research topic, generate 3-5 specific "
         "search queries that would help gather comprehensive information. "
         "Output only the queries, one per line."),
        ("user", "Research topic: {topic}")
    ])

    # Stage 2: Synthesize into a report
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior research analyst. Based on the research topic and the search "
         "queries that were planned, provide a comprehensive research report. Include:\n"
         "1. Executive Summary\n"
         "2. Key Findings (organized by query area)\n"
         "3. Conclusions and Recommendations\n\n"
         "Research queries planned:\n{queries}"),
        ("user", "Write the research report for: {topic}")
    ])

    query_chain = query_prompt | llm | StrOutputParser()

    full_chain = (
        RunnablePassthrough.assign(queries=query_chain)
        | synthesis_prompt
        | llm
        | StrOutputParser()
    )

    return query_chain, full_chain


def main():
    print("--- LangChain Deep Research Example ---")

    query_chain, full_chain = setup_deep_research_chain()

    topic = "The current state and future of quantum computing applications in cryptography"

    print("\n=== RESEARCH QUERIES ===")
    queries = query_chain.invoke({"topic": topic})
    print(queries)

    print("\n=== RESEARCH REPORT ===")
    report = full_chain.invoke({"topic": topic})
    print(report)


if __name__ == "__main__":
    main()
