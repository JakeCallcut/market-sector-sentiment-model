import pandas as pd
import re

DEBUG = True
IN_PATH = "../../data/twitter/raw/"
OUT_PATH = "../../data/twitter/processed/"

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

URL_RE = re.compile(r"https?://\S+")
PUNCT_RE = re.compile(r"[^\w\s\$\%]")
MENTION_RE  = re.compile(r"(?<!\$)@\w+")
HASHTAG_RE  = re.compile(r"#(\w+)")
EMOJI_RE    = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")

def clean_tweet(tweet: str):
    output = URL_RE.sub(" ", tweet)
    output = MENTION_RE.sub(" ", output)
    output = HASHTAG_RE.sub(r"\1", output)
    output = PUNCT_RE.sub(" ", output)
    output = EMOJI_RE.sub(" ", output)
    output = output.lower()
    output = WHITESPACE_RE.sub(" ", output)
    output = output.strip()
    return output

