import pandas as pd
import yfinance as yf
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_community.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
# Initialize LLM
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)

# ------------------------
# Stock Analysis Tools
# ------------------------


def get_basic_stock_info(ticker: str) -> str:
    """Retrieves basic information about a single stock."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        basic_info = {
            "Name": info.get("longName", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Market Cap": (
                f"${info.get('marketCap', 'N/A'):,}" if info.get("marketCap") else "N/A"
            ),
            "Current Price": (
                f"${info.get('currentPrice', 'N/A'):.2f}"
                if info.get("currentPrice")
                else "N/A"
            ),
            "52 Week High": (
                f"${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}"
                if info.get("fiftyTwoWeekHigh")
                else "N/A"
            ),
            "52 Week Low": (
                f"${info.get('fiftyTwoWeekLow', 'N/A'):.2f}"
                if info.get("fiftyTwoWeekLow")
                else "N/A"
            ),
            "Average Volume": (
                f"{info.get('averageVolume', 'N/A'):,}"
                if info.get("averageVolume")
                else "N/A"
            ),
        }

        return "\n".join([f"{k}: {v}" for k, v in basic_info.items()])

    except Exception as e:
        return f"Error fetching stock info: {str(e)}"


# ------------------------
# Analysis Chains
# ------------------------


def create_analysis_chains():
    research_prompt = PromptTemplate(
        input_variables=["ticker", "stock_data"],
        template="""
        Analyze the following stock ticker: {ticker}
        
        Stock Data:
        {stock_data}
        
        Provide a comprehensive overview including:
        1. Company description
        2. Key financial metrics
        3. Recent performance
        4. Industry position
        """,
    )

    analysis_prompt = PromptTemplate(
        input_variables=["ticker", "research_report"],
        template="""
        Based on this research report for {ticker}:
        {research_report}
        
        Perform detailed analysis covering:
        1. Valuation assessment
        2. Growth prospects
        3. Risk factors
        4. Competitive advantages
        """,
    )

    recommendation_prompt = PromptTemplate(
        input_variables=["ticker", "research_report", "analysis_report"],
        template="""
        For stock {ticker}, synthesize this information:
        
        RESEARCH SUMMARY:
        {research_report}
        
        ANALYSIS FINDINGS:
        {analysis_report}
        
        Generate an investment recommendation covering:
        1. Investment thesis
        2. Price targets
        3. Risk/reward assessment
        4. Suggested position size
        """,
    )

    return {
        "research": LLMChain(
            llm=llm, prompt=research_prompt, output_key="research_report"
        ),
        "analysis": LLMChain(
            llm=llm, prompt=analysis_prompt, output_key="analysis_report"
        ),
        "recommendation": LLMChain(
            llm=llm, prompt=recommendation_prompt, output_key="recommendation"
        ),
    }


# ------------------------
# Main Application
# ------------------------


def analyze_stock(ticker: str):
    """Run complete stock analysis workflow"""
    print(f"\n🔍 Analyzing {ticker}...\n")

    # Get basic data
    stock_data = get_basic_stock_info(ticker)
    print("📊 Basic Information:")
    print(stock_data)

    # Setup analysis chains
    chains = create_analysis_chains()

    # Run analysis pipeline
    research_result = chains["research"].run(
        {"ticker": ticker, "stock_data": stock_data}
    )

    print("\n📝 Research Report:")
    print(research_result)

    analysis_result = chains["analysis"].run(
        {"ticker": ticker, "research_report": research_result}
    )

    print("\n📈 Detailed Analysis:")
    print(analysis_result)

    recommendation = chains["recommendation"].run(
        {
            "ticker": ticker,
            "research_report": research_result,
            "analysis_report": analysis_result,
        }
    )

    print("\n💡 Investment Recommendation:")
    print(recommendation)


def main():
    print("📈 Stock Analysis CLI Tool")
    print("-------------------------")

    while True:
        ticker = input("\nEnter stock ticker (or 'quit' to exit): ").strip().upper()

        if ticker.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if not ticker:
            print("Please enter a valid ticker symbol")
            continue

        try:
            analyze_stock(ticker)
        except Exception as e:
            print(f"Error analyzing stock: {str(e)}")


if __name__ == "__main__":
    main()
