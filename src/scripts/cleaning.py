import pandas as pd
import re
import finbert as fb

IN_PATH = "../../data/raw/raw_tweets.csv"
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

#read the csv (skipping bad lines and using python engine to prevent exception on non-friendly char)
df = pd.read_csv(IN_PATH, on_bad_lines="skip", engine="python", parse_dates=["post_date"])

#only take the date and text
df = df[["post_date", "body"]].copy()

#rename to fit naming scheme
df = df.rename(columns={"post_date": "timestamp", "body": "text"})

#drop empty rows
df = df.dropna().copy()

#cut tweets longer than 300 chars (technically not possible but just in case)
df["text"] = df["text"].str.slice(0, 300)

#take only rows in a window
df = df[df["timestamp"].between("2019-01-01", "2019-06-31")]

#Clean tweets
df["text"] = df["text"].apply(clean_tweet)

#score tweets
print("Scoring tweets...")
df["fb_score"] = df["text"].apply(fb.score_text)

df.to_csv(OUT_PATH, index=False)
print(df)