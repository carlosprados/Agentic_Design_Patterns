import os
import requests
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set.")
        return None
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def classify_prompt(llm, prompt: str) -> str:
    """
    Classifies the prompt to choose the most cost-effective/appropriate model/tool.
    """
    system_prompt = (
        "You are a classifier that analyzes user prompts and returns one of three categories ONLY:\n\n"
        "- simple\n"
        "- reasoning\n"
        "- internet_search\n\n"
        "Respond ONLY with the category name."
    )

    try:
        response = llm.invoke([
            ("system", system_prompt),
            ("user", prompt)
        ])
        classification = response.content.strip().lower()
        if classification not in ["simple", "reasoning", "internet_search"]:
            classification = "simple"
        return classification
    except Exception as e:
        print(f"Classification failed: {e}")
        return "simple"

def perform_google_search(query: str) -> list:
    """
    Performs a Google Custom Search.
    """
    api_key = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    
    if not api_key or not cse_id:
        print("Warning: Google Search credentials not set. Returning empty results.")
        return []

    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cse_id, "q": query, "num": 1}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()
        return results.get("items", [])
    except Exception as e:
        print(f"Search failed: {e}")
        return []

def generate_optimized_response(prompt: str, classification: str, search_results=None):
    """
    Generates a response using the model chosen by the classifier.
    """
    if classification == "simple":
        model = "gemini-2.5-flash"
        full_content = prompt
    elif classification == "reasoning":
        model = "gemini-1.5-pro" 
        full_content = f"Think deeply and provide a detailed response: {prompt}"
    elif classification == "internet_search":
        model = "gemini-2.5-flash"
        context = ""
        if search_results:
            context = "\n".join([f"Title: {r.get('title')}\nSnippet: {r.get('snippet')}" for r in search_results])
        full_content = f"Use these search results to answer precisely:\n{context}\n\nQuery: {prompt}"
    else:
        model = "gemini-2.5-flash"
        full_content = prompt

    print(f"Using model: {model} for classification: {classification}")
    
    llm = ChatGoogleGenerativeAI(model=model)
    response = llm.invoke(full_content)
    return response.content

def run_resource_aware_demo(test_prompt: str):
    llm = get_llm()
    if not llm: return

    print(f"\n--- Processing Prompt: '{test_prompt}' ---")
    
    classification = classify_prompt(llm, test_prompt)
    
    search_results = None
    if classification == "internet_search":
        search_results = perform_google_search(test_prompt)
        
    final_answer = generate_optimized_response(test_prompt, classification, search_results)
    print("\nFinal Response:")
    print(final_answer)

if __name__ == "__main__":
    run_resource_aware_demo("What is the capital of Australia?")
    # run_resource_aware_demo("Who won the Super Bowl last year?")
