"""
Code Execution Tool Pattern using LangChain.

Demonstrates an agent that uses a Python REPL tool to solve
mathematical and computational problems by writing and executing code.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from shared.llm import get_llm
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()


@tool
def python_calculator(expression: str) -> str:
    """Evaluates a Python mathematical expression and returns the result.
    Use this for any calculation. Pass a valid Python expression as a string.
    Examples: '2**10', 'sum(range(1, 101))', 'import math; math.sqrt(144)'
    """
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


def main():
    print("--- LangChain Code Execution Example ---")

    llm = get_llm(temperature=0)
    agent = create_react_agent(llm, [python_calculator])

    queries = [
        "What is the sum of all prime numbers below 50?",
        "Calculate 2 to the power of 20.",
        "What is the factorial of 12?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        final_message = result["messages"][-1]
        print(f"Answer: {final_message.content}")


if __name__ == "__main__":
    main()
