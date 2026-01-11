import os
import random
import re
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# LangChain imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    print("Error: 'langchain_google_genai' not found. Agent will not run.")
    ChatGoogleGenerativeAI = None

# Load environment variables
_ = load_dotenv(find_dotenv())

def generate_prompt(use_case: str, goals: list[str], previous_code: str = "", feedback: str = "") -> str:
    """
    Constructs the prompt for the AI coding agent.
    """
    base_prompt = f"""
You are an AI coding agent. Your job is to write Python code based on the following use case:

Use Case: {use_case}

Your goals are:
{chr(10).join(f"- {g.strip()}" for g in goals)}
"""
    if previous_code:
        base_prompt += f"\nPreviously generated code:\n{previous_code}"
    if feedback:
        base_prompt += f"\nFeedback on previous version:\n{feedback}\n"

    base_prompt += "\nPlease return only the revised Python code. Do not include comments or explanations outside the code."
    return base_prompt

def get_code_feedback(llm, code: str, goals: list[str]) -> str:
    """
    Evaluates the code against the specified goals using the LLM.
    """
    feedback_prompt = f"""
You are a Python code reviewer. A code snippet is shown below. Based on the following goals:

{chr(10).join(f"- {g.strip()}" for g in goals)}

Please critique this code and identify if the goals are met. Mention if improvements are needed for clarity, simplicity, correctness, edge case handling, or test coverage.

Code:
{code}
"""
    return llm.invoke(feedback_prompt).content.strip()

def goals_met(llm, feedback_text: str, goals: list[str]) -> bool:
    """
    Uses the LLM to judge whether the goals have been met based on the feedback.
    """
    review_prompt = f"""
You are an AI reviewer.

Here are the goals:
{chr(10).join(f"- {g.strip()}" for g in goals)}

Here is the feedback on the code:
\"\"\"
{feedback_text}
\"\"\"

Based on the feedback above, have the goals been met?

Respond with only one word: True or False.
"""
    response = llm.invoke(review_prompt).content.strip().lower()
    return "true" in response

def clean_code_block(code: str) -> str:
    """
    Removes markdown code blocks if present.
    """
    code = code.strip()
    # Remove markdown formatting if the LLM included it
    code = re.sub(r"^```python\n", "", code, flags=re.MULTILINE)
    code = re.sub(r"^```\n", "", code, flags=re.MULTILINE)
    code = re.sub(r"\n```$", "", code, flags=re.MULTILINE)
    return code.strip()

def save_code_to_file(llm, code: str, use_case: str) -> str:
    """
    Generates a filename and saves the code.
    """
    summary_prompt = (
        f"Summarize the following use case into a single lowercase word or phrase, "
        f"no more than 10 characters, suitable for a Python filename:\n\n{use_case}"
    )
    raw_summary = llm.invoke(summary_prompt).content.strip()
    short_name = re.sub(r"[^a-zA-Z0-9_]", "", raw_summary.replace(" ", "_").lower())[:10]

    random_suffix = str(random.randint(1000, 9999))
    filename = f"{short_name}_{random_suffix}.py"
    filepath = Path.cwd() / filename

    with open(filepath, "w") as f:
        f.write(code)

    return str(filepath)

def run_code_iteration_agent(use_case: str, goals_list: list[str], max_iterations: int = 5):
    """
    Main logic for the goal-setting and monitoring agent.
    """
    if not ChatGoogleGenerativeAI:
        return

    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        print("ERROR: GOOGLE_API_KEY not set.")
        return

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    print(f"\n🎯 Use Case: {use_case}")
    print("🎯 Goals:")
    for g in goals_list:
        print(f"  - {g}")

    previous_code = ""
    feedback_text = ""

    for i in range(max_iterations):
        print(f"\n=== 🔁 Iteration {i + 1} of {max_iterations} ===")
        
        prompt = generate_prompt(use_case, goals_list, previous_code, feedback_text)
        print("🚧 Generating code...")
        
        raw_response = llm.invoke(prompt).content
        code = clean_code_block(raw_response)
        
        print(f"\n🧾 Generated Code Snippet (First 10 lines):\n" + "\n".join(code.splitlines()[:10]) + "\n...")

        print("\n📤 Evaluating code...")
        feedback_text = get_code_feedback(llm, code, goals_list)
        print(f"Feedback Summary: {feedback_text[:100]}...")

        if goals_met(llm, feedback_text, goals_list):
            print("\n✅ Goals met! Stopping iteration.")
            break
        
        print("🛠️ Goals not met. Refining...")
        previous_code = code

    final_header = f"# Implementation of: {use_case}\n\n"
    save_path = save_code_to_file(llm, final_header + code, use_case)
    print(f"\n✅ Final code saved to: {save_path}")

if __name__ == "__main__":
    use_case = "Write code to find BinaryGap of a given positive integer"
    goals = [
        "Simple to understand",
        "Functionally correct",
        "Handles edge cases (like zero or negative)",
        "Includes example usage"
    ]
    run_code_iteration_agent(use_case, goals)
