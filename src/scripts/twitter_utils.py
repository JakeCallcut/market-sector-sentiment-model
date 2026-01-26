import pandas as pd
import re
import datetime
import finbert
import os

#config vars
DEBUG = True
IN_PATH = "../../data/twitter/raw/"
OUT_PATH = "../../data/twitter/processed/"
FINAL_OUT = "../../data/twitter/processed/agg_scores/all_agg_scores.csv"


#US Eastern Market open and close times
market_open = pd.to_datetime("09:30:00").time()
market_close = pd.to_datetime("16:00:00").time()

#Regular expressions for cleaning
URL_RE = re.compile(r"https?://\S+")
PUNCT_RE = re.compile(r"[^\w\s\$\%]")
MENTION_RE  = re.compile(r"(?<!\$)@\w+")
HASHTAG_RE  = re.compile(r"#(\w+)")
EMOJI_RE    = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")

#returns a cleaned version of the inputted string
def clean_tweet(tweet: str):
    #apply all regular expression rules to string sequentially and return
    if not isinstance(tweet, str):
        return ""
    output = URL_RE.sub(" ", tweet)
    output = MENTION_RE.sub(" ", output)
    output = HASHTAG_RE.sub(r"\1", output)
    output = PUNCT_RE.sub(" ", output)
    output = EMOJI_RE.sub(" ", output)
    output = output.lower()
    output = WHITESPACE_RE.sub(" ", output)
    output = output.strip()
    return output

if os.path.exists(FINAL_OUT):
    os.remove(FINAL_OUT)

#function to process and sentiment score all tweets, processing includes cleaning, adjusting market days, removing fields, and de-duping 
def process_tweets():
    #store all data from the year
    jan_data = pd.read_csv(f"{IN_PATH}FT_Jan_2020.csv")
    feb_data = pd.read_csv(f"{IN_PATH}FT_Feb_2020.csv")
    mar_data = pd.read_csv(f"{IN_PATH}FT_Mar_2020.csv")
    apr_data = pd.read_csv(f"{IN_PATH}FT_Apr_2020.csv")
    may_data = pd.read_csv(f"{IN_PATH}FT_May_2020.csv")
    jun_data = pd.read_csv(f"{IN_PATH}FT_Jun_2020.csv")
    jul_data = pd.read_csv(f"{IN_PATH}FT_Jul_2020.csv")
    aug_data = pd.read_csv(f"{IN_PATH}FT_Aug_2020.csv")
    sep_data = pd.read_csv(f"{IN_PATH}FT_Sep_2020.csv")
    oct_data = pd.read_csv(f"{IN_PATH}FT_Oct_2020.csv")
    nov_data = pd.read_csv(f"{IN_PATH}FT_Nov_2020.csv")
    dec_data = pd.read_csv(f"{IN_PATH}FT_Dec_2020.csv")

    year_data = [jan_data, feb_data, mar_data, apr_data, may_data, jun_data, jul_data, aug_data, sep_data, oct_data, nov_data, dec_data]

    clean_year_data = []

    i = 1

    #loop though each months data
    for month_df in year_data:
        
        #apply cleaning function
        month_df["clean_tweet"] = month_df["tweet"].apply(clean_tweet)

        #only keep timestamp adn text and rename them
        month_df = month_df[["created_at","clean_tweet"]].copy()
        month_df = month_df.rename(columns={"created_at": "timestamp", "clean_tweet": "text"})

        #convert timestamp to datetime and convert timezone to US eastern 
        month_df["timestamp"] = pd.to_datetime(month_df["timestamp"], utc=True, errors="coerce")
        month_df["timestamp"] = month_df["timestamp"].dt.tz_convert("America/New_York")

        #make market date (just the date without time)
        month_df["market_date"] = month_df["timestamp"].dt.floor("D")

        #collect all of the tweets where the timestamp is after market close and increment the market date
        after_close = month_df["timestamp"].dt.time > market_close
        month_df.loc[after_close, "market_date"] = month_df.loc[after_close, "market_date"] + pd.Timedelta(days=1)
        
        #remove irrelevant time from market date (will always be 00:00:00 due to florring)
        month_df["market_date"] = month_df["market_date"].dt.date

        #just keep market date and clean text
        month_df = month_df[["market_date", "text"]].copy()

        #dedupe and write to csv
        month_df = month_df.drop_duplicates(subset=["text"], keep="first")

        #use finbert model on all text and store in new file
        month_df["finbert_score"] = month_df["text"].apply(finbert.score_text)
        month_df.to_csv(f"{OUT_PATH}{i}_cleaned_scored.csv", index = False)

        #group and aggregate into days, taking the average sentiment score, and total tweet count
        score_df = month_df[['market_date', 'finbert_score']].copy()
        score_df = (score_df.groupby('market_date', as_index=False).agg(mean_sentiment=("finbert_score", "mean"), tweet_count=("finbert_score", "size"),)).sort_values("market_date")

        #save to individual csv
        score_df.to_csv(f"{OUT_PATH}agg_scores/{i}_agg_scores.csv", index = False)

        #append to large year table
        score_df.to_csv(
            FINAL_OUT,
            mode="a",
            index=False,
            header = not os.path.exists(FINAL_OUT)
        )

        i += 1

#process all months
process_tweets()

#read large year table
final_df = pd.read_csv(FINAL_OUT)

#force into date time and drop errors
final_df["market_date"] = pd.to_datetime(final_df["market_date"], errors="coerce")
final_df = final_df[final_df["market_date"].notna()].copy()

#aggregate and group again
final_df = (final_df.groupby("market_date", as_index = False)).agg(mean_sentiment=("mean_sentiment", "mean"),tweet_count=("tweet_count", "sum"),)

#sort values chronologically
final_df = final_df.sort_values("market_date")

final_df["market_date"] = final_df["market_date"].apply(lambda x: x.strftime('%Y-%m-%d'))

#save again
final_df.to_csv(FINAL_OUT, index=False)
