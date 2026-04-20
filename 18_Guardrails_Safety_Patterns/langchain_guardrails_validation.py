"""
Guardrails Validation Pattern using LangChain.

Demonstrates multi-layer guardrails: input moderation (regex-based) and
output validation (Pydantic schema enforcement).
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from shared.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- Input Guardrail: Content Moderation ---
FORBIDDEN_PATTERNS = re.compile(
    r"\b(violence|hate\s*speech|illegal\s*activity|how\s+to\s+hack)\b",
    re.IGNORECASE
)


def moderate_input(text: str) -> tuple[bool, str]:
    """Checks input for forbidden content. Returns (is_safe, reason)."""
    match = FORBIDDEN_PATTERNS.search(text)
    if match:
        return False, f"Input blocked: forbidden content detected ('{match.group()}')"
    return True, "Input passed moderation."


# --- Output Guardrail: Pydantic Schema Validation ---
class ResearchSummary(BaseModel):
    """Validated output schema for research summaries."""
    title: str = Field(min_length=5, description="Title of the research summary")
    key_findings: list[str] = Field(min_length=2, description="List of key findings")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")

    @field_validator("key_findings")
    @classmethod
    def validate_findings(cls, v):
        if len(v) < 2:
            raise ValueError("At least 2 key findings required.")
        return v


def validate_research_output(raw_output: str) -> tuple[bool, str | ResearchSummary]:
    """Attempts to parse LLM output into the validated schema."""
    try:
        # Extract structured data from LLM output
        lines = raw_output.strip().split("\n")
        title = ""
        findings = []
        score = 0.5

        for line in lines:
            line = line.strip()
            if line.startswith("Title:"):
                title = line.split(":", 1)[1].strip()
            elif line.startswith("- ") or line.startswith("* "):
                findings.append(line.lstrip("- *").strip())
            elif line.startswith("Confidence:"):
                try:
                    score = float(line.split(":")[1].strip())
                except ValueError:
                    pass

        if not title:
            title = lines[0] if lines else "Untitled"
        if not findings:
            findings = [l.strip() for l in lines[1:] if l.strip()][:3]

        summary = ResearchSummary(
            title=title,
            key_findings=findings,
            confidence_score=score,
        )
        return True, summary
    except Exception as e:
        return False, f"Output validation failed: {e}"


def main():
    print("--- LangChain Guardrails Validation Example ---")

    # Test input guardrail
    print("\n=== Input Moderation ===")
    test_inputs = [
        "Research the latest trends in renewable energy",
        "Tell me about violence in video games and how to hack systems",
        "What are the best practices for AI safety?",
    ]
    for text in test_inputs:
        is_safe, reason = moderate_input(text)
        status = "PASS" if is_safe else "BLOCK"
        print(f"  [{status}] '{text[:60]}...' -> {reason}")

    # Test output guardrail with real LLM
    print("\n=== Output Validation ===")
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research assistant. Provide your response in this exact format:\n"
         "Title: <title>\n"
         "- <finding 1>\n"
         "- <finding 2>\n"
         "- <finding 3>\n"
         "Confidence: <score between 0 and 1>"),
        ("user", "Summarize the current state of AI safety research.")
    ])
    chain = prompt | llm | StrOutputParser()
    raw_output = chain.invoke({})

    print(f"\nRaw LLM output:\n{raw_output[:300]}")

    is_valid, result = validate_research_output(raw_output)
    if is_valid:
        print(f"\nValidated output:")
        print(f"  Title: {result.title}")
        print(f"  Findings: {len(result.key_findings)} items")
        print(f"  Confidence: {result.confidence_score}")
    else:
        print(f"\nValidation failed: {result}")


if __name__ == "__main__":
    main()
