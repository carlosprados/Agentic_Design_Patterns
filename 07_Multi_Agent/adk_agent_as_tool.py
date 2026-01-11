import os
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.agents import LlmAgent
    from google.adk.tools import agent_tool
except ImportError:
    print("Error: Google ADK components not found.")
    LlmAgent = agent_tool = None

# Load environment variables
load_dotenv()

def generate_image(prompt: str) -> dict:
    """
    Simulated tool for generating an image based on a textual prompt.
    """
    print(f"TOOL: Generating image for prompt: '{prompt}'")
    mock_image_bytes = b"mock_image_data_for_a_cat_wearing_a_hat"
    return {
        "status": "success",
        "image_bytes": mock_image_bytes,
        "mime_type": "image/png"
    }

def setup_agent_as_tool():
    if not LlmAgent or not agent_tool:
        return None

    # Image Generator sub-agent
    image_generator_agent = LlmAgent(
        name="ImageGen",
        model="gemini-2.5-flash",
        description="Generates an image based on a detailed text prompt.",
        instruction=(
            "You are an image generation specialist. Your task is to take the user's request "
            "and use the `generate_image` tool to create the image. "
            "The user's entire request should be used as the 'prompt' argument for the tool. "
            "After the tool returns the image bytes, you MUST output the image."
        ),
        tools=[generate_image]
    )

    # Wrap as a tool for the parent agent
    image_tool = agent_tool.AgentTool(
        agent=image_generator_agent,
        description="Use this tool to generate an image. The input should be a descriptive prompt of the desired image."
    )

    # Parent Artist Agent
    artist_agent = LlmAgent(
        name="Artist",
        model="gemini-2.5-flash",
        instruction=(
            "You are a creative artist. First, invent a creative and descriptive prompt for an image. "
            "Then, use the `ImageGen` tool to generate the image using your prompt."
        ),
        tools=[image_tool]
    )
    
    return artist_agent

def main():
    print("--- ADK Agent-as-a-Tool Example ---")
    artist = setup_agent_as_tool()
    if artist:
        print("Artist agent with 'ImageGen' tool configured.")
    else:
        print("Configuration failed.")

if __name__ == "__main__":
    main()
