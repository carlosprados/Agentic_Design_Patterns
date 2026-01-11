import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

def run_reflection_loop(task: str, max_iterations: int = 3):
    """
    Demonstrates a multi-step AI reflection loop to progressively improve a Python function.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found.")
        return

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    current_code = ""
    message_history = [HumanMessage(content=task)]

    for i in range(max_iterations):
        print(f"\n{'='*25} REFLECTION LOOP: ITERATION {i + 1} {'='*25}")

        # --- 1. GENERATE / REFINE STAGE ---
        if i == 0:
            print("\n>>> STAGE 1: GENERATING initial code...")
            response = llm.invoke(message_history)
            current_code = response.content
        else:
            print("\n>>> STAGE 1: REFINING code based on previous critique...")
            message_history.append(HumanMessage(content="Please refine the code using the critiques provided."))
            response = llm.invoke(message_history)
            current_code = response.content

        print(f"\n--- Generated Code (v{i+1}) ---\n{current_code}")
        message_history.append(response)

        # --- 2. REFLECT STAGE ---
        print("\n>>> STAGE 2: REFLECTING on the generated code...")
        reflector_prompt = [
            SystemMessage(content="""
                You are a senior software engineer and an expert in Python.
                Your role is to perform a meticulous code review.
                Critically evaluate the provided Python code based on the original task requirements.
                Look for bugs, style issues, missing edge cases, and areas for improvement.
                If the code is perfect and meets all requirements, respond with the single phrase 'CODE_IS_PERFECT'.
                Otherwise, provide a bulleted list of your critiques.
            """),
            HumanMessage(content=f"Original Task:\n{task}\n\nCode to Review:\n{current_code}")
        ]

        critique_response = llm.invoke(reflector_prompt)
        critique = critique_response.content

        # --- 3. STOPPING CONDITION ---
        if "CODE_IS_PERFECT" in critique:
            print("\n--- Critique ---\nNo further critiques found. The code is satisfactory.")
            break

        print(f"\n--- Critique ---\n{critique}")
        message_history.append(HumanMessage(content=f"Critique of the previous code:\n{critique}"))

    print(f"\n{'='*30} FINAL RESULT {'='*30}")
    print("\nFinal refined code after the reflection process:\n")
    print(current_code)

if __name__ == "__main__":
    task_prompt = """
    Your task is to create a Python function named `calculate_factorial`.
    This function should do the following:
    1.  Accept a single integer `n` as input.
    2.  Calculate its factorial (n!).
    3.  Include a clear docstring explaining what the function does.
    4.  Handle edge cases: The factorial of 0 is 1.
    5.  Handle invalid input: Raise a ValueError if the input is a negative number.
    """
    run_reflection_loop(task_prompt)
