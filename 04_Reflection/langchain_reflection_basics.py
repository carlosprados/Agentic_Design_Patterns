import os
import sys
import asyncio
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

def setup_reflection_chain():
    """
    Sets up a basic reflection chain with Generation, Critique, and Refinement stages.
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None

    # 1. Initial Generation
    generation_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "Write a short, simple product description for a new smart coffee mug."),
            ("user", "{product_details}")
        ])
        | llm
        | StrOutputParser()
    )

    # 2. Critique
    critique_chain = (
        ChatPromptTemplate.from_messages([
            ("system", """Critique the following product description based on clarity, conciseness, and appeal.
            Provide specific suggestions for improvement."""),
            ("user", "Product Description to Critique:\n{initial_description}")
        ])
        | llm
        | StrOutputParser()
    )

    # 3. Refinement
    refinement_chain = (
        ChatPromptTemplate.from_messages([
            ("system", """Based on the original product details and the following critique,
            rewrite the product description to be more effective.

            Original Product Details: {product_details}
            Critique: {critique}

            Refined Product Description:"""),
            ("user", "")
        ])
        | llm
        | StrOutputParser()
    )

    # Full Reflection Chain
    full_reflection_chain = (
        RunnablePassthrough.assign(
            initial_description=generation_chain
        )
        | RunnablePassthrough.assign(
            critique=critique_chain
        )
        | refinement_chain
    )
    
    return full_reflection_chain

async def run_reflection_example(product_details: str):
    """
    Runs the LangChain reflection example.
    """
    chain = setup_reflection_chain()
    if not chain:
        print("Reflection chain setup failed.")
        return

    print(f"\n--- Running Reflection Example for Product: '{product_details}' ---")
    try:
        final_refined_description = await chain.ainvoke(
            {"product_details": product_details}
        )
        print("\n--- Final Refined Product Description ---")
        print(final_refined_description)
    except Exception as e:
        print(f"\nAn error occurred during chain execution: {e}")

if __name__ == "__main__":
    test_product_details = "A mug that keeps coffee hot and can be controlled by a smartphone app."
    asyncio.run(run_reflection_example(test_product_details))
