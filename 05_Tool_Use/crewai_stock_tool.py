import os
import logging
from crewai import Agent, Task, Crew
from crewai.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@tool("Stock Price Lookup Tool")
def get_stock_price(ticker: str) -> float:
    """
    Fetches the latest simulated stock price for a given stock ticker symbol.
    Returns the price as a float. Raises a ValueError if the ticker is not found.
    """
    logging.info(f"Tool Call: get_stock_price for ticker '{ticker}'")
    simulated_prices = {
        "AAPL": 178.15,
        "GOOGL": 1750.30,
        "MSFT": 425.50,
    }
    price = simulated_prices.get(ticker.upper())

    if price is not None:
        return price
    else:
        raise ValueError(f"Simulated price for ticker '{ticker.upper()}' not found.")

def run_financial_crew():
    """
    Sets up and runs a CrewAI agent with a stock price lookup tool.
    """
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: The GOOGLE_API_KEY environment variable is not set.")
        return

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # Define the Agent
    financial_analyst_agent = Agent(
        role='Senior Financial Analyst',
        goal='Analyze stock data using provided tools and report key prices.',
        backstory="You are an experienced financial analyst adept at using data sources to find stock information. You provide clear, direct answers.",
        verbose=True,
        tools=[get_stock_price],
        allow_delegation=False,
        llm=llm
    )

    # Define the Task
    analyze_aapl_task = Task(
        description=(
            "What is the current simulated stock price for Apple (ticker: AAPL)? "
            "Use the 'Stock Price Lookup Tool' to find it. "
            "If the ticker is not found, you must report that you were unable to retrieve the price."
        ),
        expected_output=(
            "A single, clear sentence stating the simulated stock price for AAPL. "
            "For example: 'The simulated stock price for AAPL is $178.15.' "
            "If the price cannot be found, state that clearly."
        ),
        agent=financial_analyst_agent,
    )

    # Formulate the Crew
    financial_crew = Crew(
        agents=[financial_analyst_agent],
        tasks=[analyze_aapl_task],
        verbose=True
    )

    print("\n## Starting the Financial Crew...")
    result = financial_crew.kickoff()
    print("\n## Crew execution finished.")
    print("\nFinal Result:\n", result)

if __name__ == "__main__":
    run_financial_crew()
