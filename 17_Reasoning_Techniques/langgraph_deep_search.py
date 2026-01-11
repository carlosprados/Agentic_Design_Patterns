from typing import List, TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- 1. State Definition ---
class OverallState(TypedDict):
    query: str
    search_queries: List[str]
    research_results: List[str]
    reflection: str
    final_answer: str

# --- 2. Node Implementations ---
def generate_query(state: OverallState):
    print("--- NODE: generate_query ---")
    # Simulation: LLM generates search queries
    return {"search_queries": [f"search for {state['query']}"]}

def web_research(state: OverallState):
    print("--- NODE: web_research ---")
    # Simulation: Tools perform search
    return {"research_results": ["Research findings for " + q for q in state['search_queries']]}

def reflection(state: OverallState):
    print("--- NODE: reflection ---")
    # Simulation: LLM reflects on findings
    return {"reflection": "The research is partially complete."}

def finalize_answer(state: OverallState):
    print("--- NODE: finalize_answer ---")
    return {"final_answer": "Final synthesized answer based on research and reflection."}

# --- 3. Edge Logic ---
def continue_to_web_research(state: OverallState):
    # Logic to decide if we need more research or move to reflection
    return "web_research"

def evaluate_research(state: OverallState):
    # Logic to decide if research is sufficient
    # Return "web_research" to loop or "finalize_answer" to finish
    if "partially" in state['reflection']:
        return "finalize_answer" # Simplified for demo
    return "web_research"

# --- 4. Graph Construction ---
def build_deep_search_graph():
    builder = StateGraph(OverallState)

    builder.add_node("generate_query", generate_query)
    builder.add_node("web_research", web_research)
    builder.add_node("reflection", reflection)
    builder.add_node("finalize_answer", finalize_answer)

    builder.add_edge(START, "generate_query")
    
    # Using simple edge instead of conditional for basic demo if appropriate, 
    # but following notebook's conditional structure:
    builder.add_conditional_edges("generate_query", lambda s: "web_research")
    
    builder.add_edge("web_research", "reflection")
    
    builder.add_conditional_edges(
        "reflection", 
        evaluate_research, 
        {"web_research": "web_research", "finalize_answer": "finalize_answer"}
    )
    
    builder.add_edge("finalize_answer", END)
    
    return builder.compile()

if __name__ == "__main__":
    print("--- LangGraph Deep Search Reasoning Flow ---")
    graph = build_deep_search_graph()
    result = graph.invoke({"query": "Impact of AI on code maintainability"})
    print("\nFinal Result:", result['final_answer'])
