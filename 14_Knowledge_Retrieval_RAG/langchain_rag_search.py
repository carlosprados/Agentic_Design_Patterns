"""
Search-Based RAG Pattern using LangChain.

Demonstrates retrieval-augmented generation where a simulated search
provides context that grounds the LLM's response.
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

# Simulated search results (replace with GoogleSearchAPIWrapper for production)
SEARCH_DB = {
    "langchain": [
        "LangChain v1.0 released with stable API and improved performance.",
        "LangGraph enables stateful multi-actor LLM applications.",
        "LangChain supports 100+ LLM providers and retrieval integrations.",
    ],
    "rag": [
        "RAG combines retrieval and generation for grounded AI responses.",
        "Vector databases like Weaviate and Pinecone power RAG pipelines.",
        "RAG reduces hallucination by grounding LLM output in retrieved facts.",
    ],
    "ai agents": [
        "AI agents use tools and reasoning to accomplish complex tasks.",
        "Multi-agent systems enable collaboration between specialized agents.",
        "Agent frameworks include LangGraph, CrewAI, and Google ADK.",
    ],
}


def search_documents(query: str) -> str:
    """Simulates document retrieval from a search index."""
    query_lower = query.lower()
    results = []
    for key, docs in SEARCH_DB.items():
        if key in query_lower:
            results.extend(docs)
    if not results:
        # Return all docs as fallback
        for docs in SEARCH_DB.values():
            results.extend(docs)
    return "\n".join(f"- {doc}" for doc in results[:5])


def setup_rag_chain():
    """Builds a RAG chain: retrieve documents → generate grounded answer."""
    llm = get_llm(temperature=0)

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a knowledgeable assistant. Answer the user's question based ONLY on "
         "the provided context. If the context doesn't contain enough information, say so.\n\n"
         "Context:\n{context}"),
        ("user", "{question}")
    ])

    chain = (
        RunnablePassthrough.assign(context=lambda x: search_documents(x["question"]))
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main():
    print("--- LangChain Search-Based RAG Example ---")
    chain = setup_rag_chain()

    questions = [
        "What is RAG and how does it reduce hallucinations?",
        "Tell me about LangChain and LangGraph.",
        "How do AI agents work?",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        context = search_documents(question)
        print(f"Retrieved context:\n{context}")
        answer = chain.invoke({"question": question})
        print(f"A: {answer[:300]}")


if __name__ == "__main__":
    main()
