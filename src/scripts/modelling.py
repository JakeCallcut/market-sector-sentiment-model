import pandas as pd

DEBUG = True
RETURNS_PATH = "../../data/market/processed/labelled_returns.csv"
SCORES_PATH = "../../data/twitter/processed/agg_scores/all_agg_scores.csv"
MODEL_PATH = "../../data/modelling_table.csv"

def create_modelling_table():
    market_df = pd.read_csv(RETURNS_PATH)
    twitter_df = pd.read_csv(SCORES_PATH)

    twitter_df["market_date"] = pd.to_datetime(twitter_df["market_date"], errors="coerce")
    market_df["market_date"] = pd.to_datetime(market_df["market_date"], errors="coerce")

    twitter_df = twitter_df[twitter_df["market_date"].notna()].copy()
    market_df = market_df[market_df["market_date"].notna()].copy()

    model_df = pd.merge(
        twitter_df,
        market_df,
        on="market_date",
        how="inner"
    ).sort_values("market_date")

    model_df.to_csv(MODEL_PATH, index=False)

create_modelling_table()