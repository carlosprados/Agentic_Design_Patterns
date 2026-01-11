import os
import asyncio
from typing import List, Optional, Dict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain imports
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import Tool
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.memory import ConversationBufferMemory
except ImportError:
    print("Error: Required LangChain components not found.")
    ChatPromptTemplate = Tool = ChatGoogleGenerativeAI = AgentExecutor = create_react_agent = ConversationBufferMemory = None

# Load environment variables
load_dotenv()

# --- 1. Task Management System ---
class Task(BaseModel):
    """Represents a single task in the system."""
    id: str
    description: str
    priority: Optional[str] = None  # P0, P1, P2
    assigned_to: Optional[str] = None # Name of the worker

class TaskManager:
    """Manages tasks in memory."""
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.next_id = 1

    def create(self, description: str) -> Task:
        task_id = f"TASK-{self.next_id:03d}"
        task = Task(id=task_id, description=description)
        self.tasks[task_id] = task
        self.next_id += 1
        return task

    def update(self, task_id: str, priority=None, assigned_to=None) -> Optional[Task]:
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if priority: task.priority = priority
            if assigned_to: task.assigned_to = assigned_to
            return task
        return None

    def list_tasks(self) -> str:
        if not self.tasks: return "No tasks found."
        return "\n".join([f"[{t.id}] {t.description} | Pri: {t.priority or 'N/A'} | User: {t.assigned_to or 'N/A'}" for t in self.tasks.values()])

task_manager = TaskManager()

# --- 2. Tool Definitions ---
def create_task_tool(description: str) -> str:
    """Creates a new task and returns its ID."""
    t = task_manager.create(description)
    return f"Created task {t.id}."

def prioritize_task_tool(task_id: str, priority: str) -> str:
    """Sets priority for a task (P0, P1, P2)."""
    t = task_manager.update(task_id, priority=priority)
    return f"Priority {priority} set for {task_id}." if t else "Task not found."

def assign_task_tool(task_id: str, worker: str) -> str:
    """Assigns a task to a worker."""
    t = task_manager.update(task_id, assigned_to=worker)
    return f"Assigned {task_id} to {worker}." if t else "Task not found."

# --- 3. Agent Setup ---
def setup_pm_agent():
    if not all([ChatGoogleGenerativeAI, AgentExecutor]):
        return None

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    tools = [
        Tool(name="create_task", func=create_task_tool, description="Create a task. Input: description"),
        Tool(name="prioritize", func=prioritize_task_tool, description="Set priority. Input: 'TASK-ID, PRIORITY'"),
        Tool(name="assign", func=assign_task_tool, description="Assign worker. Input: 'TASK-ID, WORKER'"),
        Tool(name="list", func=task_manager.list_tasks, description="List all tasks.")
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Project Manager agent. Create tasks, then assign priority and workers. Default to P1 and 'Worker A' if unspecified."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    )

if __name__ == "__main__":
    print("--- Prioritization & Task Management Agent Demo ---")
    executor = setup_pm_agent()
    if executor:
        print("PM Agent ready.")
        # asyncio.run(executor.ainvoke({"input": "Create a task to fix the database bug, it's urgent!"}))
    else:
        print("Agent setup failed.")
