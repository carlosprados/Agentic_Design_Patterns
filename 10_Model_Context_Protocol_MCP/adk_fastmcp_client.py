import os
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.agents import LlmAgent
    from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, HttpServerParameters
except ImportError:
    print("Error: Google ADK or MCP components not found.")
    LlmAgent = MCPToolset = HttpServerParameters = None

# Load environment variables
load_dotenv()

# Define the FastMCP server's address (ensure fastmcp_server.py is running)
FASTMCP_SERVER_URL = "http://localhost:8000"

def setup_mcp_client_agent():
    if not LlmAgent:
        return None

    return LlmAgent(
        model='gemini-2.5-flash',
        name='fastmcp_greeter_agent',
        instruction='You are a friendly assistant that can greet people by their name. Use the "greet" tool.',
        tools=[
            MCPToolset(
                connection_params=HttpServerParameters(
                    url=FASTMCP_SERVER_URL,
                ),
                tool_filter=['greet']
            )
        ],
    )

if __name__ == "__main__":
    print("--- ADK FastMCP Client Agent ---")
    agent = setup_mcp_client_agent()
    if agent:
        print(f"Agent '{agent.name}' initialized to consume MCP server at {FASTMCP_SERVER_URL}")
    else:
        print("Agent initialization failed.")
