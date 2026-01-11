import os
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda

# Load environment variables
load_dotenv()

def booking_handler(request: str) -> str:
    """Simulates the Booking Agent handling a request."""
    print("\n--- DELEGATING TO BOOKING HANDLER ---")
    return f"Booking Handler processed request: '{request}'. Result: Simulated booking action."

def info_handler(request: str) -> str:
    """Simulates the Info Agent handling a request."""
    print("\n--- DELEGATING TO INFO HANDLER ---")
    return f"Info Handler processed request: '{request}'. Result: Simulated information retrieval."

def unclear_handler(request: str) -> str:
    """Handles requests that couldn't be delegated."""
    print("\n--- HANDLING UNCLEAR REQUEST ---")
    return f"Coordinator could not delegate request: '{request}'. Please clarify."

def setup_langgraph_router():
    """
    Sets up a routing chain using LangChain Expression Language (LCEL).
    """
    try:
        # Use a descriptive model name if needed, or fallback
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None

    # Coordinator Router Prompt
    coordinator_router_prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze the user's request and determine which specialist handler should process it.
         - If the request is related to booking flights or hotels, output 'booker'.
         - For all other general information questions, output 'info'.
         - If the request is unclear or doesn't fit either category, output 'unclear'.
         ONLY output one word: 'booker', 'info', or 'unclear'."""),
        ("user", "{request}")
    ])

    router_chain = coordinator_router_prompt | llm | StrOutputParser()

    # Define the delegation branches
    def route_to_handler(inputs):
        decision = inputs['decision'].strip().lower()
        request_text = inputs['request']
        
        if decision == 'booker':
            return booking_handler(request_text)
        elif decision == 'info':
            return info_handler(request_text)
        else:
            return unclear_handler(request_text)

    # Combine into a single chain
    full_chain = (
        {"decision": router_chain, "request": RunnablePassthrough()}
        | RunnableLambda(route_to_handler)
    )
    
    return full_chain

def main():
    print("--- LangGraph/LCEL Routing Example ---")
    
    chain = setup_langgraph_router()
    if not chain:
        print("Chain setup failed.")
        return

    # Tests
    requests = [
        "Book me a flight to London.",
        "What is the capital of Italy?",
        "Tell me about quantum physics."
    ]

    for req in requests:
        print(f"\nUser: {req}")
        result = chain.invoke(req)
        print(f"Assistant: {result}")

if __name__ == "__main__":
    main()
