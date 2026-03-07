# Config paths
MODEL_TABLE_PATH = "../../data/results/modelling_table.csv"

# sklearn requires non-negative classes, so map {-1, 0, 1} -> {0, 1, 2}
LABEL_MAP = {-1: 0, 0: 1, 1: 2}

# Feature columns used for prediction
FEATURE_COLS = ["mean_fb_score", "mean_fb_score_lag1", "mean_fb_score_lag2"]

#creating 1 day and 2 day lags
def make_lags(df, col, lags=(1, 2)):
    for l in lags:
        df[f"{col}_lag{l}"] = df[col].shift(l)
    return df
