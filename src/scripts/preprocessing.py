import pandas as pd

#read tweets and returns
tweet_df = pd.read_csv("../../data/processed/clean_tweets.csv")
returns_df = pd.read_csv("../../data/processed/processed_returns.csv")

#ensure the date is a datetime type
returns_df["date"] = pd.to_datetime(returns_df["date"]).dt.normalize()

#just keep the timestamp and sentiment
tweet_df = tweet_df[["timestamp", "fb_score"]].copy()

#change to just keep date
tweet_df["timestamp"] = pd.to_datetime(tweet_df["timestamp"])
tweet_df["date"] = tweet_df["timestamp"].dt.normalize()

#aggregate average score by date
daily_mean = tweet_df.groupby("date", as_index=False)["fb_score"].mean().rename(columns={"fb_score": "mean_fb_score"})

#join the two datasets on date
df = returns_df.merge(daily_mean, on="date", how="inner")

#write to csv
df.to_csv("../../data/results/modelling_table.csv", index=False)