import os
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

from helpers import make_lags, LABEL_MAP, FEATURE_COLS, MODEL_TABLE_PATH


# config vars
EVAL_PATH = "../../data/results/gb_evaluation.csv"


def get_gradient_boost(**kwargs):
	defaults = {
		"n_estimators": 100,
		"max_depth": 3,
		"random_state": 42,
	}
	defaults.update(kwargs)
	return GradientBoostingClassifier(**defaults)


def train_and_evaluate(df, ticker, n_splits=5):

	df_local = df.copy()
	# create lags for all features
	df_local = make_lags(df_local, "mean_fb_score", lags=(1, 2))
	df_local = make_lags(df_local, "tweet_volume", lags=(1,))

	#get only essential columns
	required_cols = FEATURE_COLS + [ticker]
	df_local = df_local.dropna(subset=required_cols).reset_index(drop=True)

	X = df_local[FEATURE_COLS].values
	y = df_local[ticker].map(LABEL_MAP).values

	#split for cross-validation
	tscv = TimeSeriesSplit(n_splits=n_splits)

	train_accs = []
	accs = []
	f1s = []
	y_true_all = []
	y_pred_all = []

	#perform cross validation
	for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
		X_train, X_test = X[train_idx], X[test_idx]
		y_train, y_test = y[train_idx], y[test_idx]

		pipeline = make_pipeline(StandardScaler(), get_gradient_boost())
		pipeline.fit(X_train, y_train)

		# compute train accuracy
		y_pred_train = pipeline.predict(X_train)
		train_acc = accuracy_score(y_train, y_pred_train)
		train_accs.append(train_acc)

		y_pred = pipeline.predict(X_test)

		acc = accuracy_score(y_test, y_pred)
		f1 = f1_score(y_test, y_pred, average="macro")

		accs.append(acc)
		f1s.append(f1)

		y_true_all.extend(y_test.tolist())
		y_pred_all.extend(y_pred.tolist())

	# aggregate metrics
	train_accuracy_mean = float(np.mean(train_accs))
	overall_acc = float(np.mean(accs))
	overall_f1 = float(np.mean(f1s))
	cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1, 2])

	# retrain on full data
	final_pipeline = make_pipeline(StandardScaler(), get_gradient_boost())
	final_pipeline.fit(X, y)

	#build result map storing accuracy, f1 and confusion matrix for plotting later
	result = {
		"ticker": ticker,
		"train_accuracy_mean": train_accuracy_mean,
		"accuracy_mean": overall_acc,
		"f1_macro_mean": overall_f1,
		"cm_00": int(cm[0, 0]),
		"cm_01": int(cm[0, 1]),
		"cm_02": int(cm[0, 2]),
		"cm_10": int(cm[1, 0]),
		"cm_11": int(cm[1, 1]),
		"cm_12": int(cm[1, 2]),
		"cm_20": int(cm[2, 0]),
		"cm_21": int(cm[2, 1]),
		"cm_22": int(cm[2, 2]),
		"n_samples": int(len(y)),
	}

	#show output and return result
	print(f"SECTOR={ticker}, ACCURACY={overall_acc:.2f}, Used {int(len(y))} samples")
	return result


def main():

	#get modelling table
	df = pd.read_csv(MODEL_TABLE_PATH)

	df["date"] = pd.to_datetime(df["date"], errors="coerce")
	df = df[df["date"].notna()].copy()
	df = df.sort_values("date").reset_index(drop=True)

	# identify tickers (all columns except features)
	ticker_cols = [c for c in df.columns if c not in ("date", "mean_fb_score", "tweet_volume", "VIX")]

	#run model for each ticker and store results
	results = []
	for ticker in ticker_cols:
		res = train_and_evaluate(df, ticker, n_splits=5)
		if res:
			results.append(res)

	#build and save evaluation table
	if results:
		os.makedirs(os.path.dirname(EVAL_PATH), exist_ok=True)
		pd.DataFrame(results).to_csv(EVAL_PATH, index=False)
		print(f"Evaluation saved to {EVAL_PATH}")


if __name__ == "__main__":
	main()
