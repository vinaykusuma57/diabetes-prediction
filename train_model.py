import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

CSV_PATH = "diabetesmain.csv"
MODEL_PATH = "model.joblib"
RANDOM_STATE = 42

def main():
    df = pd.read_csv(CSV_PATH)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    pipe.fit(X, y)
    joblib.dump(pipe, MODEL_PATH)
    print(f"✅ Model trained & saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
