import pandas as pd
import numpy as np
import pandas_market_calendars as mcal

#config vars
TAU_RETURN = 0.002  #0.2%

#read tweets and returns
tweet_df = pd.read_csv("../../data/processed/clean_tweets.csv")
returns_df = pd.read_csv("../../data/processed/processed_returns.csv")

#ensure the date is a datetime type
returns_df["date"] = pd.to_datetime(returns_df["date"], utc=True).dt.normalize()

#just keep the timestamp and sentiment
tweet_df = tweet_df[["timestamp", "fb_score"]].copy()

#change to just keep date
tweet_df["date"] = pd.to_datetime(tweet_df["timestamp"], utc=True).dt.normalize()

# helper: compute weighted daily score using positive/negative averages and counts
# based on formula defined in dissertation document
# also returns tweet_volume (count of tweets per day)
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
		tweet_volume=("fb_score", "count"),  # count tweets per day
	).reset_index()

	# safe multiplication: replace NaN means with 0
	numerator = agg["sbar_pos"].fillna(0) * agg["N_pos"] + agg["sbar_neg"].fillna(0) * agg["N_neg"]
	denom = agg["N_pos"] + agg["N_neg"]

	# where denom == 0, fallback to arithmetic mean for that date
	agg["mean_fb_score"] = np.where(denom > 0, numerator / denom, agg["mean_all"])

	return agg[["date", "mean_fb_score", "tweet_volume"]]


def validate_nyse_trading_dates(df, date_col="date"):

	#fail if there are invalid dates
	dates = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.normalize()
	if dates.isna().any():
		raise ValueError("Found invalid or missing dates in modelling table before market-date validation.")

	#get date range
	start = dates.min().strftime("%Y-%m-%d")
	end = dates.max().strftime("%Y-%m-%d")

	#get valid dates in range from pandas-market-calendars 
	nyse = mcal.get_calendar("NYSE")
	valid_days = nyse.valid_days(start_date=start, end_date=end)
	valid_dates = pd.DatetimeIndex(valid_days.tz_convert("UTC").tz_localize(None)).normalize()

	#get dates from modelling table
	date_values = pd.DatetimeIndex(dates.dt.tz_localize(None)).normalize()
	#get all of the invalid dates by finding the difference
	invalid_dates = sorted(date_values.difference(valid_dates))

	#fail if there are any
	if invalid_dates:
		invalid_str = ", ".join(d.strftime("%Y-%m-%d") for d in invalid_dates)
		raise ValueError(
			f"Found {len(invalid_dates)} non-NYSE market date(s) in modelling table: {invalid_str}"
		)


# compute weighted daily mean using existing fb_score values (no rescoring)
daily_mean = compute_weighted_daily_score(tweet_df, tau=TAU_RETURN)

# load VIX data
vix_df = pd.read_csv("../../data/processed/vix.csv")
vix_df["date"] = pd.to_datetime(vix_df["date"], utc=True).dt.normalize()

# join the datasets on date (returns + sentiment + VIX)
df = returns_df.merge(daily_mean, on="date", how="inner")
df = df.merge(vix_df, on="date", how="inner")

# fail if any non-trading dates reach the final modelling table
validate_nyse_trading_dates(df, date_col="date")

#write to csv
df.to_csv("../../data/results/modelling_table.csv", index=False)