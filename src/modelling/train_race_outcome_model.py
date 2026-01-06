import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

DATA_PATH = "data/processed/race_driver_level/race_driver_features.parquet"

df = pd.read_parquet(DATA_PATH)


df["win_flag"] = (df["finish_position"] == 1).astype(int)
df["podium_flag"] = (df["finish_position"] <= 3).astype(int)
df["points_flag"] = (df["finish_position"] <= 10).astype(int)

FEATURE_COLS = [
    "grid_position",
    "avg_lap_time",
    "best_lap_time",
    "lap_count",
    "stints_used",
    "used_soft",
    "used_medium",
    "used_hard"
]

X = df[FEATURE_COLS]
y = df["podium_flag"]
groups = df["race_name"]

gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

baseline_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

baseline_model.fit(X_train, y_train)

y_pred = baseline_model.predict(X_test)
y_prob = baseline_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("BM ROC-AUC:", roc_auc_score(y_test, y_prob))

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_prob = rf_model.predict_proba(X_test)[:, 1]
print("RF ROC-AUC:", roc_auc_score(y_test, rf_prob))


importance = pd.Series(
    rf_model.feature_importances_,
    index=FEATURE_COLS
).sort_values(ascending=False)

print(importance)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(
    rf_model,
    MODEL_DIR / "podium_model.pkl"
)
