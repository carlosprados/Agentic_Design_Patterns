import os
import asyncio
from typing import List
from dotenv import load_dotenv
import logging

# Google ADK imports
try:
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.code_executors import BuiltInCodeExecutor
    from google.genai import types
except ImportError:
    print("Error: Google ADK components not found.")
    LlmAgent = Runner = InMemorySessionService = BuiltInCodeExecutor = types = None

# Load environment variables
load_dotenv()

# Application Constants
APP_NAME = "calculator"
USER_ID = "user1234"
SESSION_ID = "session_code_exec_async"

def setup_code_agent():
    if not LlmAgent:
        return None
        
    return LlmAgent(
       name="calculator_agent",
       model="gemini-2.5-flash",
       code_executor=BuiltInCodeExecutor(),
       instruction="""You are a calculator agent.
       When given a mathematical expression, write and execute Python code to calculate the result.
       Return only the final numerical result as plain text, without markdown or code blocks.
       """,
       description="Executes Python code to perform calculations.",
    )

async def call_agent_async(agent, query):
    if not Runner:
        return

    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    # In ADK 1.22.0, explicit session creation is required before running if providing a session_id
    await runner.session_service.create_session(
        app_name=runner.app_name, 
        user_id=USER_ID, 
        session_id=SESSION_ID
    )

    content = types.Content(role='user', parts=[types.Part(text=query)])
    print(f"\n--- Running Query: {query} ---")
    
    try:
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            if event.content and event.content.parts and event.is_final_response():
                for part in event.content.parts:
                    if part.executable_code:
                        print(f"  Debug: Agent generated code:\n```python\n{part.executable_code.code}\n```")
                    elif part.code_execution_result:
                        print(f"  Debug: Code Execution Result: {part.code_execution_result.outcome} - Output:\n{part.code_execution_result.output}")
                    elif part.text and not part.text.isspace():
                        print(f"  Text: '{part.text.strip()}'")

                text_parts = [part.text for part in event.content.parts if part.text]
                final_result = "".join(text_parts)
                print(f"==> Final Agent Response: {final_result}")

    except Exception as e:
        print(f"ERROR during agent run: {e}")

async def main():
    agent = setup_code_agent()
    if agent:
        await call_agent_async(agent, "Calculate the value of (5 + 7) * 3")
        await call_agent_async(agent, "What is 10 factorial?")

if __name__ == "__main__":
    asyncio.run(main())
