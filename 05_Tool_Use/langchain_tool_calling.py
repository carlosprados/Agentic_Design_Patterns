import os
import asyncio
from typing import List
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool as langchain_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Load environment variables
load_dotenv()

@langchain_tool
def search_information(query: str) -> str:
   """
   Provides factual information on a given topic. Use this tool to find answers to phrases
   like 'capital of France' or 'weather in London?'.
   """
   print(f"\n--- 🛠️ Tool Called: search_information with query: '{query}' ---")
   simulated_results = {
       "weather in london": "The weather in London is currently cloudy with a temperature of 15°C.",
       "capital of france": "The capital of France is Paris.",
       "population of earth": "The estimated population of Earth is around 8 billion people.",
       "tallest mountain": "Mount Everest is the tallest mountain above sea level.",
       "default": f"Simulated search result for '{query}': No specific information found."
   }
   result = simulated_results.get(query.lower(), simulated_results["default"])
   print(f"--- TOOL RESULT: {result} ---")
   return result

def setup_tool_agent():
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    except Exception as e:
        print(f"🛑 Error initializing language model: {e}")
        return None

    tools = [search_information]
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    return AgentExecutor(agent=agent, verbose=True, tools=tools)

async def run_agent_with_tool(executor, query: str):
   print(f"\n--- 🏃 Running Agent with Query: '{query}' ---")
   try:
       response = await executor.ainvoke({"input": query})
       print("\n--- ✅ Final Agent Response ---")
       print(response["output"])
   except Exception as e:
       print(f"\n🛑 An error occurred: {e}")

async def main():
   executor = setup_tool_agent()
   if executor:
       tasks = [
           run_agent_with_tool(executor, "What is the capital of France?"),
           run_agent_with_tool(executor, "What's the weather like in London?"),
           run_agent_with_tool(executor, "Tell me something about dogs.")
       ]
       await asyncio.gather(*tasks)

if __name__ == "__main__":
   asyncio.run(main())
