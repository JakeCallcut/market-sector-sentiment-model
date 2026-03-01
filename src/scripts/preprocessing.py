import pandas as pd
import numpy as np

#config vars
TAU_RETURN = 0.002  #0.2%

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

# helper: compute weighted daily score using positive/negative averages and counts
# based on formula defined in dissertation document
def compute_weighted_daily_score(tweet_df, tau=TAU_RETURN):

	df = tweet_df.copy()
	# boolean indicators for positive / negative
	df["_is_pos"] = df["fb_score"] > tau
	df["_is_neg"] = df["fb_score"] < -tau

	# aggregate counts and means per date
	agg = df.groupby("date").agg(
		N_pos=("_is_pos", "sum"),
		N_neg=("_is_neg", "sum"),
		sbar_pos=("fb_score", lambda x: x[x > tau].mean()),
		sbar_neg=("fb_score", lambda x: x[x < -tau].mean()),
		mean_all=("fb_score", "mean"),
	).reset_index()

	# safe multiplication: replace NaN means with 0
	numerator = agg["sbar_pos"].fillna(0) * agg["N_pos"] + agg["sbar_neg"].fillna(0) * agg["N_neg"]
	denom = agg["N_pos"] + agg["N_neg"]

	# where denom == 0, fallback to arithmetic mean for that date
	agg["mean_fb_score"] = np.where(denom > 0, numerator / denom, agg["mean_all"])

	return agg[["date", "mean_fb_score"]]


# compute weighted daily mean using existing fb_score values (no rescoring)
daily_mean = compute_weighted_daily_score(tweet_df, tau=TAU_RETURN)

# join the two datasets on date
df = returns_df.merge(daily_mean, on="date", how="inner")

#write to csv
df.to_csv("../../data/results/modelling_table.csv", index=False)