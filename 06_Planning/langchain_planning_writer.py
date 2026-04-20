"""
Planning Pattern using LangChain LCEL.

Demonstrates a two-stage pipeline: first the agent creates an article plan,
then uses that plan as context to write the full article.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from shared.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


def setup_planning_chain():
    """Builds a two-stage LCEL chain: plan generation → article writing."""
    llm = get_llm(temperature=0.7)

    # Stage 1: Generate the plan
    plan_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert article planner. Create a structured outline for an article "
         "on the given topic. Include:\n"
         "1. Title\n"
         "2. Key sections (3-4)\n"
         "3. Main points for each section\n"
         "4. Target audience"),
        ("user", "Create an article plan for: {topic}")
    ])

    # Stage 2: Write using the plan
    write_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a skilled technical writer. Write a concise article based on the "
         "following plan. Keep it focused and practical.\n\nPlan:\n{plan}"),
        ("user", "Write the article about: {topic}")
    ])

    plan_chain = plan_prompt | llm | StrOutputParser()

    full_chain = (
        RunnablePassthrough.assign(plan=plan_chain)
        | write_prompt
        | llm
        | StrOutputParser()
    )

    return plan_chain, full_chain


def main():
    print("--- LangChain Planning Writer Example ---")

    plan_chain, full_chain = setup_planning_chain()

    topic = "How AI agents are transforming software development workflows"

    # Show the plan first
    print("\n=== PLAN ===")
    plan = plan_chain.invoke({"topic": topic})
    print(plan)

    # Generate the article using the plan
    print("\n=== ARTICLE ===")
    article = full_chain.invoke({"topic": topic})
    print(article)


if __name__ == "__main__":
    main()
