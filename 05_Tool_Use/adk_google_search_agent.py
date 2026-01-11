import asyncio
import os
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools import google_search
    from google.genai import types
except ImportError:
    print("Error: Google ADK components not found.")
    Agent = Runner = InMemorySessionService = google_search = types = None

# Load environment variables
load_dotenv()

APP_NAME = "Google Search_agent"
USER_ID = "user1234"
SESSION_ID = "1234"

def setup_search_agent():
    if not Agent:
        return None
        
    return Agent(
       name="basic_search_agent",
       model="gemini-2.5-flash",
       description="Agent to answer questions using Google Search.",
       instruction="I can answer your questions by searching the internet. Just ask me anything!",
       tools=[google_search] 
    )

async def call_agent(query):
   agent = setup_search_agent()
   if not agent or not Runner:
       print("Search agent setup failed.")
       return

   session_service = InMemorySessionService()
   runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

   content = types.Content(role='user', parts=[types.Part(text=query)])
   
   print(f"\n--- Running Search Query: {query} ---")
   # Note: The original used runner.run (sync) in an async context. 
   # We use the events iterator here.
   events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

   for event in events:
       if event.is_final_response():
           final_response = event.content.parts[0].text
           print("Agent Response: ", final_response)

if __name__ == "__main__":
   asyncio.run(call_agent("what's the latest ai news?"))
