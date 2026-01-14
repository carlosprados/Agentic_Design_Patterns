import os
import asyncio
from typing import List
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool as langchain_tool
from langchain.agents import create_agent

# Load environment variables
load_dotenv()

@langchain_tool
def search_information(query: str) -> str:
   """
   Provides factual information on a given topic. Use this tool to find answers to phrases
   like 'capital of France' or 'weather in London?'.
   """
   print(f"\n--- 🛠️ Tool Called: search_information with query: '{query}' ---")
   # Clean the query for better matching
   clean_query = query.lower().strip("?.! ")
   simulated_results = {
       "weather in london": "The weather in London is currently cloudy with a temperature of 15°C.",
       "capital of france": "The capital of France is Paris.",
       "population of earth": "The estimated population of Earth is around 8 billion people.",
       "tallest mountain": "Mount Everest is the tallest mountain above sea level.",
       "default": f"Simulated search result for '{query}': No specific information found."
   }
   result = simulated_results.get(clean_query, simulated_results["default"])
   print(f"--- TOOL RESULT: {result} ---")
   return result

def setup_tool_agent():
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    except Exception as e:
        print(f"🛑 Error initializing language model: {e}")
        return None

    tools = [search_information]
    
    # Modern create_agent API handles prompt and tools automatically
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant.",
        debug=False # Set to True for verbose graph execution
    )

async def run_agent_with_tool(graph, query: str):
   print(f"\n--- 🏃 Running Agent with Query: '{query}' ---")
   try:
       # Modern API uses "messages" key for input
       response = await graph.ainvoke({"messages": [{"role": "user", "content": query}]})
       print("\n--- ✅ Final Agent Response ---")
       # Final response is the last message in the state
       final_message = response["messages"][-1]
       print(final_message.content)
   except Exception as e:
       print(f"\n🛑 An error occurred: {e}")

async def main():
   graph = setup_tool_agent()
   if graph:
       tasks = [
           run_agent_with_tool(graph, "What is the capital of France?"),
           run_agent_with_tool(graph, "What's the weather like in London?"),
           run_agent_with_tool(graph, "Tell me something about dogs.")
       ]
       await asyncio.gather(*tasks)

if __name__ == "__main__":
   asyncio.run(main())
