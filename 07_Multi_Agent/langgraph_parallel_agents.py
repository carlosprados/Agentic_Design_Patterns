"""
Parallel Agents Pattern using LangGraph.

Demonstrates two agents (weather + news) running in parallel branches,
with their outputs merged into shared state.
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


class ParallelState(TypedDict):
    city: str
    results: Annotated[List[str], operator.add]


def weather_agent(state: ParallelState) -> dict:
    """Fetches simulated weather information for the city."""
    print("--- NODE: weather_agent ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a weather service. Provide a brief current weather report "
                   "for the given city (make a realistic forecast)."),
        ("user", "Weather for: {city}")
    ])
    chain = prompt | llm | StrOutputParser()
    weather = chain.invoke({"city": state["city"]})
    return {"results": [f"[WEATHER] {weather}"]}


def news_agent(state: ParallelState) -> dict:
    """Fetches simulated news headlines for the city."""
    print("--- NODE: news_agent ---")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a news service. Provide 3 brief recent headline-style news "
                   "items relevant to the given city."),
        ("user", "News for: {city}")
    ])
    chain = prompt | llm | StrOutputParser()
    news = chain.invoke({"city": state["city"]})
    return {"results": [f"[NEWS] {news}"]}


def build_parallel_graph():
    builder = StateGraph(ParallelState)
    builder.add_node("weather_agent", weather_agent)
    builder.add_node("news_agent", news_agent)

    # Fan-out: both agents start from START
    builder.add_edge(START, "weather_agent")
    builder.add_edge(START, "news_agent")

    # Fan-in: both end at END
    builder.add_edge("weather_agent", END)
    builder.add_edge("news_agent", END)

    return builder.compile()


def main():
    print("--- LangGraph Parallel Agents Example ---")
    graph = build_parallel_graph()
    result = graph.invoke({"city": "Tokyo", "results": []})

    for item in result["results"]:
        print(f"\n{item[:300]}")


if __name__ == "__main__":
    main()
