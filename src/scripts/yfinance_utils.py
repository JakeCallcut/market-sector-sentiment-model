import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

#CONFIG VARIABLES
START_DATE = "2019-01-01"
END_DATE = "2019-12-31"
PROCESSED_PATH = "../../data/processed/processed_returns.csv"
DEBUG = True
SAVE_TO_FILE = True

TICKERS = [
    "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU", "GLD", "USO", "SPY"
]
TAU = 0.002

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
        closes.to_csv(PROCESSED_PATH, index_label="date")
        print("saved closes")

    return closes

#Returns returns (change in price) for tickers in date range
def get_returns():

    #read closes file
    closes = pd.read_csv(
        PROCESSED_PATH,
        parse_dates=["date"]
    ).set_index("date")

    #calculate change and ignore first day
    returns = closes.pct_change()
    returns = returns.dropna(how="all")

    if DEBUG:
        print(returns)
    
    if SAVE_TO_FILE:
        returns.to_csv(PROCESSED_PATH, index_label="date")
        print("saved returns")

    return returns

def label_returns():
    #read returns file into dataframe
    returns = pd.read_csv(
        PROCESSED_PATH,
        parse_dates=["date"]
    ).set_index("date")

    #create labels dataframe
    labels = returns.copy()

    #when data point is above tau, assign 1 label
    #when data point is above tau or below -tau, assign 0 label
    #when data point is below -tau, assign -1 label
    labels[returns > TAU] = 1
    labels[(returns >= -TAU) & (returns <= TAU)] = 0
    labels[returns < -TAU] = -1

    if DEBUG:
        print(labels)

    if SAVE_TO_FILE:
        labels.to_csv(PROCESSED_PATH, index_label="date")

    return labels

#RUN FUNCTIONS
get_adj_close()
get_returns()
label_returns()