import os
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.agents import LlmAgent
    from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
except ImportError:
    print("Error: Google ADK or MCP components not found.")
    LlmAgent = MCPToolset = StdioServerParameters = None

# Load environment variables
load_dotenv()

# Target folder for MCP filesystem
TARGET_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_managed_files")
os.makedirs(TARGET_FOLDER_PATH, exist_ok=True)

def setup_mcp_filesystem_agent():
    if not LlmAgent:
        return None

    return LlmAgent(
        model='gemini-2.5-flash',
        name='filesystem_assistant_agent',
        instruction=(
            'Help the user manage their files. You can list files, read files, and write files. '
            f'You are operating in the following directory: {TARGET_FOLDER_PATH}'
        ),
        tools=[
            MCPToolset(
                connection_params=StdioServerParameters(
                    command='npx',
                    args=[
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        TARGET_FOLDER_PATH,
                    ],
                ),
            )
        ],
    )

if __name__ == "__main__":
    print("--- ADK MCP Filesystem Agent ---")
    agent = setup_mcp_filesystem_agent()
    if agent:
        print(f"Agent '{agent.name}' initialized with filesystem MCP access to: {TARGET_FOLDER_PATH}")
    else:
        print("Agent initialization failed.")
