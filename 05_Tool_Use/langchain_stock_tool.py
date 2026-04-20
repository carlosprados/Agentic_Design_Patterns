"""
Custom Tool Pattern using LangChain.

Demonstrates defining a custom tool with @tool decorator and using it
with a ReAct agent to answer financial questions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from shared.llm import get_llm
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()


# Simulated stock data
STOCK_PRICES = {
    "AAPL": 178.50,
    "GOOGL": 141.25,
    "MSFT": 415.80,
    "AMZN": 185.60,
    "TSLA": 245.30,
    "NVDA": 875.40,
}


@tool
def get_stock_price(ticker: str) -> str:
    """Looks up the current stock price for a given ticker symbol.
    Pass the ticker symbol in uppercase (e.g., AAPL, GOOGL, MSFT).
    """
    ticker = ticker.upper().strip()
    if ticker in STOCK_PRICES:
        return f"The current price of {ticker} is ${STOCK_PRICES[ticker]:.2f}"
    return f"Ticker '{ticker}' not found. Available: {', '.join(STOCK_PRICES.keys())}"


def main():
    print("--- LangChain Stock Tool Example ---")

    llm = get_llm(temperature=0)
    agent = create_react_agent(llm, [get_stock_price])

    queries = [
        "What's the current price of Apple stock?",
        "Compare the stock prices of Google and Microsoft.",
        "Which is more expensive, Tesla or NVIDIA stock?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        final_message = result["messages"][-1]
        print(f"Answer: {final_message.content[:300]}")


if __name__ == "__main__":
    main()
