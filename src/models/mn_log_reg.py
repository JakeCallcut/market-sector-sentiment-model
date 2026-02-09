import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#config vars
MODEL_TABLE_PATH = "../../data/results/modelling_table.csv"

#get modelling table
df = pd.read_csv(MODEL_TABLE_PATH)

df["market_date"] = pd.to_datetime(df["market_date"], errors="coerce")
df = df[df["market_date"].notna()].copy()

#map up, neutral, down to 0, 1, 2 
label_map = {-1: 0, 0: 1, 1: 2}
df["target"] = df["SPY"].map(label_map)
