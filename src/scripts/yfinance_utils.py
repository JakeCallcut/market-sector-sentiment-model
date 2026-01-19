import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

#CONFIG VARIABLES
START_DATE = "2020-01-01"
END_DATE = "2021-01-01"
OUTPUT_PATH = "../../data/market/raw/"
DEBUG = True
SAVE_TO_FILE = True
VISUALISE = True

TICKERS = [
    "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU", "GLD", "USO", "SPY"
]

#Returns close prices for tickers in date range
def get_adj_close():

    #get data from yfinance
    df = yf.download(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    
    #keep only the adjusted closes column
    closes = df["Adj Close"]

    if DEBUG:
        print(closes)

    if SAVE_TO_FILE:
        closes.to_csv(f"{OUTPUT_PATH}closes_daily.csv", index_label="date")
        print("saved closes")

    if VISUALISE:
        #plot a lightweight graph of prices
        sample_cols = closes.columns[:3]  # just benchmark, gold, and oil
        closes[sample_cols].plot(figsize=(8, 4), linewidth=1)

        plt.title("Daily Adjusted Close (Sample)")
        plt.xlabel("Date")
        plt.ylabel("Adjusted Close")
        plt.axhline(0, linewidth=0.8)

        plt.tight_layout()
        plt.show()

    return closes

#Returns returns (change in price) for tickers in date range
def get_returns():

    #read closes file
    closes = pd.read_csv(
        f"{OUTPUT_PATH}closes_daily.csv",
        parse_dates=["date"]
    ).set_index("date")

    #calculate change and ignore first day
    returns = closes.pct_change()
    returns = returns.dropna(how="all")

    if DEBUG:
        print(returns)
    
    if SAVE_TO_FILE:
        returns.to_csv(f"{OUTPUT_PATH}returns_daily.csv", index_label="date")
        print("saved returns")

    if VISUALISE:
        #plot a lightweight graph of returns
        sample_cols = returns.columns[:3]  # just benchmark, gold, and oil
        returns[sample_cols].plot(figsize=(8, 4), linewidth=1)

        plt.title("Daily Returns (Sample)")
        plt.xlabel("Date")
        plt.ylabel("Return")
        plt.axhline(0, linewidth=0.8)

        plt.tight_layout()
        plt.show()

    return returns


get_adj_close()
get_returns()