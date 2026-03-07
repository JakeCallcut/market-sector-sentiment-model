import pandas as pd
import re
import finbert as fb

IN_PATH = "hf://datasets/StephanAkkerman/stock-market-tweets-data/stock-market-tweets-data.csv"
OUT_PATH = "../../data/processed/clean_tweets.csv"

#cleaning tweets of all undesired characters and strings
def clean_tweet(tweet: str):
    #Regular expressions for all unecessary urls, punctuation, user tags, emojis, and whitespace
    URL_RE = re.compile(r"https?://\S+")
    PUNCT_RE = re.compile(r"[^\w\s\$\%]")
    MENTION_RE = re.compile(r"(?<!\$)@\w+")
    HASHTAG_RE = re.compile(r"#(\w+)")
    EMOJI_RE = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
    WHITESPACE_RE = re.compile(r"\s+") 

    #if the tweet is a string, apply the regexs
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

#read the csv from Hugging Face
df = pd.read_csv(IN_PATH, parse_dates=["created_at"])

#only take the date and text
df = df[["created_at", "text"]].copy()

#rename to fit naming scheme
df = df.rename(columns={"created_at": "timestamp"})

#drop empty rows
df = df.dropna().copy()

#cut tweets longer than 300 chars (technically not possible but just in case)
df["text"] = df["text"].str.slice(0, 300)

#take only rows in a window
df = df[df["timestamp"].between("2020-04-09", "2020-07-16")]

#Clean tweets
df["text"] = df["text"].apply(clean_tweet)

#score tweets
print("Scoring tweets...")
df["fb_score"] = fb.score_dataframe(df["text"])

df.to_csv(OUT_PATH, index=False)
print(df)