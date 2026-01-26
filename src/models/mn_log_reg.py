import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#config vars
MODEL_TABLE_PATH = "../../data/modelling_table.csv"
THRESHOLD_DATE = "2020-09-01"
FEATURES = ["mean_sentiment", "tweet_count"]

#get modelling table
df = pd.read_csv(MODEL_TABLE_PATH)

df["market_date"] = pd.to_datetime(df["market_date"], errors="coerce")
df = df[df["market_date"].notna()].copy()

#map up, neutral, down to 0, 1, 2 
label_map = {-1: 0, 0: 1, 1: 2}
df["target"] = df["SPY"].map(label_map)

#define target attribute and features
X = df[FEATURES]
y = df["target"]

#split into train and test by threshold date, avoiding bias
train = df[df["market_date"] < THRESHOLD_DATE]
test  = df[df["market_date"] >= THRESHOLD_DATE]

X_train = train[FEATURES]
y_train = train["target"]

X_test = test[FEATURES]
y_test = test["target"]

#scaling for MN logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced"
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
baseline = max(y_test.mean(), 1 - y_test.mean())

print(f"Model accuracy:    {accuracy:.3f}")
print(f"Baseline accuracy: {baseline:.3f}")
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, labels=[-1, 0, 1]))

print(y_test.value_counts())