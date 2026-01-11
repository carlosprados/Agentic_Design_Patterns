from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_prompt_chaining_example():
    """
    Demonstrates basic prompt chaining using LangChain Expression Language (LCEL).
    Extracts technical specifications from text and transforms them into JSON.
    """
    # Initialize the Language Model
    # Ensure GOOGLE_API_KEY is set in your .env file
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # --- Prompt 1: Extract Information ---
    prompt_extract = ChatPromptTemplate.from_template(
        "Extract the technical specifications from the following text:\n\n{text_input}"
    )

    # --- Prompt 2: Transform to JSON ---
    prompt_transform = ChatPromptTemplate.from_template(
        "Transform the following specifications into a JSON object with 'cpu', 'memory', and 'storage' as keys:\n\n{specifications}"
    )

    # --- Build the Chain using LCEL ---
    # The StrOutputParser() converts the LLM's message output to a simple string.
    extraction_chain = prompt_extract | llm | StrOutputParser()

    # The full chain passes the output of the extraction chain into the 'specifications'
    # variable for the transformation prompt.
    full_chain = (
        {"specifications": extraction_chain}
        | prompt_transform
        | llm
        | StrOutputParser()
    )

    # --- Run the Chain ---
    input_text = "The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD."

    print("\n--- Running Extraction and Transformation Chain ---")
    final_result = full_chain.invoke({"text_input": input_text})

    print("\n--- Final JSON Output ---")
    print(final_result)

if __name__ == "__main__":
    run_prompt_chaining_example()
