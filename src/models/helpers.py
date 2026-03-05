# Config paths
MODEL_TABLE_PATH = "../../data/results/modelling_table.csv"

# sklearn requires non-negative classes, so map {-1, 0, 1} -> {0, 1, 2}
LABEL_MAP = {-1: 0, 0: 1, 1: 2}

# Feature columns used for prediction
FEATURE_COLS = ["mean_fb_score", "mean_fb_score_lag1", "mean_fb_score_lag2"]


def make_lags(df, col, lags=(1, 2)):
    """Add lagged columns to dataframe.
    
    Args:
        df: DataFrame to modify
        col: Column name to create lags from
        lags: Tuple of lag periods (default: 1 and 2 day lags)
    
    Returns:
        DataFrame with new lag columns added
    """
    for l in lags:
        df[f"{col}_lag{l}"] = df[col].shift(l)
    return df
