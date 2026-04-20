"""
Agent-as-Tool Pattern using LangGraph.

Demonstrates wrapping a compiled sub-graph as a tool that a parent
agent can invoke, enabling hierarchical agent composition.
"""

import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from shared.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

load_dotenv()


# --- Sub-agent: Image Description Generator (simulated) ---
class ImageGenState(TypedDict):
    prompt: str
    description: str


def generate_image_description(state: ImageGenState) -> dict:
    """Generates a detailed image description from a prompt."""
    llm = get_llm(temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an image generation AI. Given a prompt, describe in vivid detail "
         "what the generated image would look like. Include colors, composition, "
         "lighting, and style."),
        ("user", "{prompt}")
    ])
    chain = prompt | llm | StrOutputParser()
    description = chain.invoke({"prompt": state["prompt"]})
    return {"description": description}


def build_image_subgraph():
    builder = StateGraph(ImageGenState)
    builder.add_node("generate", generate_image_description)
    builder.add_edge(START, "generate")
    builder.add_edge("generate", END)
    return builder.compile()


# Compile the sub-graph once
image_graph = build_image_subgraph()


# --- Wrap the sub-graph as a tool for the parent agent ---
@tool
def generate_image(prompt: str) -> str:
    """Generates a detailed image description based on a creative prompt.
    Use this tool when the user asks for an image or visual content.
    """
    result = image_graph.invoke({"prompt": prompt})
    return result["description"]


def main():
    print("--- LangGraph Agent-as-Tool Example ---")

    llm = get_llm(temperature=0)
    parent_agent = create_react_agent(llm, [generate_image])

    queries = [
        "Create an image of a futuristic city at sunset with flying cars.",
        "Generate a picture of a cat sitting on a stack of programming books.",
    ]

    for query in queries:
        print(f"\nUser: {query}")
        result = parent_agent.invoke({"messages": [{"role": "user", "content": query}]})
        final_message = result["messages"][-1]
        print(f"Artist: {final_message.content[:400]}")


if __name__ == "__main__":
    main()
