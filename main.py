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
import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

# Load environment variables
load_dotenv()

# Initialize LLM with error handling
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not found")

llm = ChatGroq(
    model_name="llama3-70b-8192",
    groq_api_key=GROQ_API_KEY,
    temperature=0.3,  # Lower temperature for more factual responses
)

# ------------------------
# Professional Analysis Tools
# ------------------------


def get_comprehensive_stock_data(ticker: str) -> Tuple[Dict, pd.DataFrame]:
    """
    Retrieves comprehensive stock data including:
    - Company fundamentals
    - Historical prices
    - Analyst estimates
    Returns tuple of (info_dict, historical_data_df)
    """
    try:
        stock = yf.Ticker(ticker)

        # Get all available info
        info = stock.info

        # Get historical data for different timeframes
        hist_1y = stock.history(period="1y")
        hist_5y = stock.history(period="5y")

        # Get analyst recommendations
        recommendations = stock.recommendations

        # Get institutional holders
        institutional = stock.institutional_holders

        # Get financial statements
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        cash_flow = stock.cashflow

        return {
            "info": info,
            "recommendations": recommendations,
            "institutional_holders": institutional,
            "balance_sheet": balance_sheet,
            "income_statement": income_stmt,
            "cash_flow": cash_flow,
        }, hist_1y

    except Exception as e:
        raise Exception(f"Failed to fetch data for {ticker}: {str(e)}")


def calculate_technical_indicators(data: pd.DataFrame) -> Dict:
    """
    Calculates professional technical indicators:
    - Moving averages
    - RSI
    - MACD
    - Bollinger Bands
    - Volume analysis
    """
    closes = data["Close"]
    volumes = data["Volume"]

    # Moving Averages
    sma_50 = closes.rolling(window=50).mean()
    sma_200 = closes.rolling(window=200).mean()

    # RSI
    delta = closes.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = closes.ewm(span=12, adjust=False).mean()
    ema_26 = closes.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    rolling_std = closes.rolling(window=20).std()
    upper_band = sma_50 + (rolling_std * 2)
    lower_band = sma_50 - (rolling_std * 2)

    return {
        "sma_50": sma_50.iloc[-1],
        "sma_200": sma_200.iloc[-1],
        "rsi": rsi.iloc[-1],
        "macd": macd.iloc[-1],
        "signal": signal.iloc[-1],
        "upper_band": upper_band.iloc[-1],
        "lower_band": lower_band.iloc[-1],
        "volume_avg": volumes.mean(),
    }


def generate_valuation_metrics(info: Dict) -> Dict:
    """
    Calculates comprehensive valuation metrics:
    - Traditional ratios (P/E, P/S, P/B, etc.)
    - Growth metrics
    - Profitability metrics
    - Financial health ratios
    """
    return {
        "valuation": {
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "price_to_book": info.get("priceToBook"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
        },
        "growth": {
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "ebitda_growth": info.get("ebitdaGrowth"),
        },
        "profitability": {
            "gross_margins": info.get("grossMargins"),
            "operating_margins": info.get("operatingMargins"),
            "profit_margins": info.get("profitMargins"),
            "return_on_equity": info.get("returnOnEquity"),
            "return_on_assets": info.get("returnOnAssets"),
        },
        "financial_health": {
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "debt_to_equity": info.get("debtToEquity"),
            "interest_coverage": (
                info.get("earningsBeforeInterestTaxes") / info.get("interestExpense")
                if info.get("interestExpense")
                else None
            ),
        },
    }


# ------------------------
# Professional Analysis Chains
# ------------------------


def create_professional_chains():
    """Creates institutional-grade analysis chains"""

    fundamental_prompt = PromptTemplate(
        input_variables=["ticker", "info", "valuation_metrics"],
        template="""
        Perform comprehensive fundamental analysis for {ticker}:
        
        Company Information:
        {info}
        
        Valuation Metrics:
        {valuation_metrics}
        
        Your analysis must include:
        1. Business Model Analysis (competitive advantages, moat)
        2. Financial Health Assessment (liquidity, solvency)
        3. Growth Prospects (historical and projected)
        4. Valuation Assessment (relative and absolute)
        5. Industry Position and Competitive Landscape
        
        Format as a professional research report with clear sections.
        """,
    )

    technical_prompt = PromptTemplate(
        input_variables=["ticker", "technical_indicators", "price_data"],
        template="""
        Perform technical analysis for {ticker}:
        
        Technical Indicators:
        {technical_indicators}
        
        Price Data Summary:
        {price_data}
        
        Analyze:
        1. Trend Analysis (short, medium, long-term)
        2. Key Support/Resistance Levels
        3. Momentum Indicators Interpretation
        4. Volume Analysis
        5. Chart Patterns
        
        Provide specific price levels for entry/exit points.
        """,
    )

    recommendation_prompt = PromptTemplate(
        input_variables=["ticker", "fundamental_analysis", "technical_analysis"],
        template="""
        Generate institutional investment recommendation for {ticker}:
        
        Fundamental Analysis:
        {fundamental_analysis}
        
        Technical Analysis:
        {technical_analysis}
        
        Include:
        1. Investment Thesis (3-5 key points)
        2. Price Targets (conservative/base/aggressive)
        3. Risk Assessment (systematic/unsystematic risks)
        4. Position Sizing Guidance
        5. Monitoring Criteria
        
        Format for a professional investment committee.
        """,
    )

    return {
        "fundamental": LLMChain(
            llm=llm, prompt=fundamental_prompt, output_key="fundamental_analysis"
        ),
        "technical": LLMChain(
            llm=llm, prompt=technical_prompt, output_key="technical_analysis"
        ),
        "recommendation": LLMChain(
            llm=llm, prompt=recommendation_prompt, output_key="recommendation"
        ),
    }


# ------------------------
# Main Application
# ------------------------


def professional_analysis(ticker: str):
    """Run institutional-grade analysis workflow"""
    print(f"\n🏦 Initiating Professional Analysis for {ticker}...")
    start_time = time.time()

    try:
        # Step 1: Data Collection
        print("\n📊 Collecting comprehensive data...")
        info_dict, hist_data = get_comprehensive_stock_data(ticker)

        # Step 2: Technical Analysis
        print("📈 Calculating technical indicators...")
        technical_indicators = calculate_technical_indicators(hist_data)

        # Step 3: Fundamental Analysis
        print("💼 Analyzing fundamentals...")
        valuation_metrics = generate_valuation_metrics(info_dict["info"])

        # Step 4: Run Analysis Chains
        chains = create_professional_chains()

        print("\n🔍 Conducting fundamental analysis...")
        fundamental_result = chains["fundamental"].run(
            {
                "ticker": ticker,
                "info": str(info_dict["info"]),
                "valuation_metrics": str(valuation_metrics),
            }
        )

        print("\n📊 Performing technical analysis...")
        technical_result = chains["technical"].run(
            {
                "ticker": ticker,
                "technical_indicators": str(technical_indicators),
                "price_data": str(hist_data.describe()),
            }
        )

        # Step 5: Generate Recommendation
        print("\n💡 Formulating recommendation...")
        recommendation = chains["recommendation"].run(
            {
                "ticker": ticker,
                "fundamental_analysis": fundamental_result,
                "technical_analysis": technical_result,
            }
        )

        # Display Results
        print("\n" + "=" * 80)
        print(f"🏛️  INSTITUTIONAL RESEARCH REPORT: {ticker}")
        print("=" * 80)
        print("\n⭐ FUNDAMENTAL ANALYSIS:")
        print(fundamental_result)
        print("\n📈 TECHNICAL ANALYSIS:")
        print(technical_result)
        print("\n💎 INVESTMENT RECOMMENDATION:")
        print(recommendation)
        print("\n" + "=" * 80)
        print(f"⏱️  Analysis completed in {time.time()-start_time:.2f} seconds")

    except Exception as e:
        print(f"\n❌ Professional analysis failed: {str(e)}")


def main():
    print(
        """
    #############################################
    #  INSTITUTIONAL STOCK ANALYSIS PLATFORM    #
    #  ---------------------------------------- #
    #  Provides professional-grade analysis for #
    #  fund managers and serious investors     #
    #############################################
    """
    )

    while True:
        try:
            ticker = input("\nEnter stock ticker (or 'exit' to quit): ").strip().upper()

            if ticker.lower() in ["exit", "quit"]:
                print("\nTerminating analysis session...")
                break

            if not ticker:
                print("Please enter a valid ticker symbol")
                continue

            professional_analysis(ticker)

        except KeyboardInterrupt:
            print("\nSession terminated by user")
            break
        except Exception as e:
            print(f"\n⚠️  Error: {str(e)}")


if __name__ == "__main__":
    main()
