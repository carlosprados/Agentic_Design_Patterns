"""
MCP Client Pattern using LangChain.

Demonstrates connecting to an MCP server (FastMCP) via HTTP and using
the exposed tools through a LangChain agent. Requires the FastMCP server
from fastmcp_server.py to be running on localhost:8000.

Usage:
    1. Start the server: uv run 10_Model_Context_Protocol_MCP/fastmcp_server.py
    2. Run this client: uv run 10_Model_Context_Protocol_MCP/langchain_mcp_client.py
"""

import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from shared.llm import get_llm
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

MCP_SERVER_URL = "http://localhost:8000"


@tool
def greet_via_mcp(name: str) -> str:
    """Calls the MCP server's greet tool to get a personalized greeting.
    Pass a person's name to receive a greeting from the remote MCP service.
    """
    try:
        response = requests.post(
            f"{MCP_SERVER_URL}/call-tool",
            json={"name": "greet", "arguments": {"name": name}},
            timeout=5,
        )
        if response.ok:
            data = response.json()
            # Extract text content from MCP response
            if isinstance(data, dict) and "content" in data:
                for item in data["content"]:
                    if item.get("type") == "text":
                        return item["text"]
            return str(data)
        return f"MCP server returned status {response.status_code}"
    except requests.ConnectionError:
        return "Error: MCP server not reachable at localhost:8000. Start it with: uv run 10_Model_Context_Protocol_MCP/fastmcp_server.py"
    except Exception as e:
        return f"Error calling MCP server: {e}"


def main():
    print("--- LangChain MCP Client Example ---")
    print("Note: Requires fastmcp_server.py running on localhost:8000\n")

    llm = get_llm(temperature=0)
    agent = create_react_agent(llm, [greet_via_mcp])

    queries = [
        "Greet Charlie using the MCP service.",
        "Can you get a greeting for Antonio from the greeting service?",
    ]

    for query in queries:
        print(f"Query: {query}")
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        final_message = result["messages"][-1]
        print(f"Answer: {final_message.content}\n")


if __name__ == "__main__":
    main()
