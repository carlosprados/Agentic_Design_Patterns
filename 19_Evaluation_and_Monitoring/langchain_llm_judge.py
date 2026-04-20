"""
LLM-as-a-Judge Pattern using LangChain.

Demonstrates using an LLM to evaluate agent outputs against a structured
rubric, producing scored assessments with reasoning.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from shared.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

EVALUATION_RUBRIC = """
Evaluation Criteria (score each 1-5):
1. **Clarity & Precision**: Is the response clear, unambiguous, and well-structured?
2. **Neutrality & Bias**: Is the response balanced and free from bias?
3. **Relevance**: Does the response directly address the question asked?
4. **Completeness**: Does the response cover all important aspects of the topic?
"""


class LLMJudge:
    """Evaluates text outputs against a rubric using LLM-based scoring."""

    def __init__(self):
        self.llm = get_llm(temperature=0)

    def evaluate(self, question: str, response: str) -> dict:
        """Evaluates a response and returns structured scores."""
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert evaluator. Assess the following response against the rubric.\n\n"
             "Rubric:\n{rubric}\n\n"
             "Output your evaluation as valid JSON with these fields:\n"
             '{{"overall_score": <1-5>, "clarity": <1-5>, "neutrality": <1-5>, '
             '"relevance": <1-5>, "completeness": <1-5>, '
             '"rationale": "<brief explanation>", '
             '"recommended_action": "<accept/revise/reject>"}}'),
            ("user",
             "Question: {question}\n\nResponse to evaluate:\n{response}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        raw = chain.invoke({
            "rubric": EVALUATION_RUBRIC,
            "question": question,
            "response": response,
        })

        # Parse JSON from LLM output
        try:
            # Strip markdown code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[1:-1])
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"raw_evaluation": raw, "parse_error": True}


def main():
    print("--- LangChain LLM-as-a-Judge Example ---")
    judge = LLMJudge()

    # Sample question and response to evaluate
    question = "What are the main benefits and risks of artificial general intelligence?"
    response_to_evaluate = (
        "Artificial General Intelligence (AGI) could revolutionize science, medicine, "
        "and technology by solving complex problems humans cannot. However, risks include "
        "loss of human control, economic disruption, and potential misuse. The development "
        "of AGI requires careful safety research, international cooperation, and robust "
        "governance frameworks to ensure beneficial outcomes."
    )

    print(f"\nQuestion: {question}")
    print(f"\nResponse: {response_to_evaluate}")

    evaluation = judge.evaluate(question, response_to_evaluate)

    print("\n=== EVALUATION ===")
    if evaluation.get("parse_error"):
        print(f"Raw: {evaluation.get('raw_evaluation', '')[:500]}")
    else:
        print(f"  Overall Score: {evaluation.get('overall_score', 'N/A')}/5")
        print(f"  Clarity: {evaluation.get('clarity', 'N/A')}/5")
        print(f"  Neutrality: {evaluation.get('neutrality', 'N/A')}/5")
        print(f"  Relevance: {evaluation.get('relevance', 'N/A')}/5")
        print(f"  Completeness: {evaluation.get('completeness', 'N/A')}/5")
        print(f"  Rationale: {evaluation.get('rationale', 'N/A')}")
        print(f"  Action: {evaluation.get('recommended_action', 'N/A')}")


if __name__ == "__main__":
    main()
