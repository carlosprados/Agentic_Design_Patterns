SELF_CORRECTION_PROMPT = """
You are a highly critical and detail-oriented Self-Correction Agent. Your task is to review a previously generated piece of content against its original requirements and identify areas for improvement.

Process:
1.  **Understand Original Requirements:** What was the original intent and constraints?
2.  **Analyze Current Content:** Read the provided content carefully.
3.  **Identify Discrepancies:** Look for accuracy issues, completeness gaps, and clarity problems.
4.  **Propose Specific Improvements:** Propose concrete solutions for each weakness.
5.  **Generate Revised Content:** Rewrite the content incorporating all changes.

Original Prompt: "{original_prompt}"
Current Content: "{current_content}"

Please provide your internal 'Correction Thoughts' followed by 'Revised Content'.
"""

def print_self_correction_example():
    original = "Write a short social media post (max 150 chars) for 'GreenTech Gadgets'."
    current = "We have new products. They are green and techy. Buy GreenTech Gadgets now!"
    formatted = SELF_CORRECTION_PROMPT.format(original_prompt=original, current_content=current)
    print("--- Self-Correction Reasoning Prompt ---")
    print(formatted)

if __name__ == "__main__":
    print_self_correction_example()
