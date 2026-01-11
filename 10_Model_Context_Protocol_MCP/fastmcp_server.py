from fastmcp import FastMCP, tool
import os

# Initialize the FastMCP server
mcp_server = FastMCP("GreetingServer")

@tool()
def greet(name: str) -> str:
    """
    Generates a personalized greeting.

    Args:
        name: The name of the person to greet.
    """
    return f"Hello, {name}! Nice to meet you."

if __name__ == "__main__":
    print("Starting FastMCP server on http://localhost:8000")
    print("This server exposes a 'greet' tool.")
    # Run the server
    mcp_server.run()
