import os
import asyncio
from typing import Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

# Load environment variables
load_dotenv()

def setup_parallel_chain():
    """
    Sets up a parallel execution chain using LangChain's RunnableParallel (LCEL).
    """
    try:
        # Configuration
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None

    # --- Define Independent Chains ---
    summarize_chain: Runnable = (
        ChatPromptTemplate.from_messages([
            ("system", "Summarize the following topic concisely:"),
            ("user", "{topic}")
        ])
        | llm
        | StrOutputParser()
    )

    questions_chain: Runnable = (
        ChatPromptTemplate.from_messages([
            ("system", "Generate three interesting questions about the following topic:"),
            ("user", "{topic}")
        ])
        | llm
        | StrOutputParser()
    )

    terms_chain: Runnable = (
        ChatPromptTemplate.from_messages([
            ("system", "Identify 5-10 key terms from the following topic, separated by commas:"),
            ("user", "{topic}")
        ])
        | llm
        | StrOutputParser()
    )

    # --- Build the Parallel + Synthesis Chain ---
    # 1. Define parallel tasks
    map_chain = RunnableParallel(
        {
            "summary": summarize_chain,
            "questions": questions_chain,
            "key_terms": terms_chain,
            "topic": RunnablePassthrough(),
        }
    )

    # 2. Synthesis prompt
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", """Based on the following information:
         Summary: {summary}
         Related Questions: {questions}
         Key Terms: {key_terms}
         Synthesize a comprehensive answer."""),
        ("user", "Original topic: {topic}")
    ])

    # 3. Full chain
    full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()
    
    return full_parallel_chain

async def run_parallel_example(topic: str):
    """
    Asynchronously invokes the parallel processing chain.
    """
    chain = setup_parallel_chain()
    if not chain:
        print("Parallel chain setup failed.")
        return

    print(f"\n--- Running Parallel LangChain Example for Topic: '{topic}' ---")
    try:
        response = await chain.ainvoke({"topic": topic})
        print("\n--- Final Response ---")
        print(response)
    except Exception as e:
        print(f"\nAn error occurred during chain execution: {e}")

if __name__ == "__main__":
    test_topic = "The history of space exploration"
    asyncio.run(run_parallel_example(test_topic))
