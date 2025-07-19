from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from glob import glob

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "diabetesmain.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
PLOT_PATH  = BASE_DIR / "static" / "feature_importance.png"
COMPARE_DIR = BASE_DIR / "static" / "compare_plots"
COMPARE_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
df_ref = pd.read_csv(DATA_PATH)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# ---------- utility -------------------------------------------------
def make_feature_plot():
    if model is None or not hasattr(model.named_steps['clf'], 'feature_importances_'):
        return
    if PLOT_PATH.exists():
        return
    importances = model.named_steps["clf"].feature_importances_
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(FEATURES, importances)
    ax.set_xlabel("Relative Importance")
    ax.set_title("Which factors influence the prediction?")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, bbox_inches="tight")
    plt.close(fig)

def build_kde_plots(user_values):
    """Generate population KDE vs user value for each feature."""
    # Clear previous pngs
    for fp in COMPARE_DIR.glob("*.png"):
        fp.unlink()
    paths = []
    for feat, uval in zip(FEATURES, user_values):
        fig, ax = plt.subplots(figsize=(4,2))
        sns.kdeplot(df_ref[feat], ax=ax, fill=True, alpha=0.3)
        ax.axvline(uval, color="red", lw=2)
        ax.set_title(feat)
        ax.set_xlabel("")
        ax.set_ylabel("")
        fig.tight_layout()
        fname = f"{feat}.png"
        save_path = COMPARE_DIR / fname
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        paths.append(fname)
    return paths

make_feature_plot()

# ---------- routes --------------------------------------------------
@app.route("/")
def root():
    return render_template("welcome.html")
# adding form
@app.route("/form")
def form():
    return render_template("index.html", features=FEATURES)
# adding dash board route
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model not found. Train it first by running train_model.py.", 500
    try:
        values = [float(request.form[f]) for f in FEATURES]
    except ValueError:
        return "Invalid input. Please enter numerical values only.", 400

    proba = model.predict_proba([values])[0][1]
    outcome = "Positive" if proba >= 0.5 else "Negative"

    build_kde_plots(values)

    return render_template(
        "result.html",
        outcome=outcome,
        proba=f"{proba*100:0.1f}%"
    )

@app.route("/compare")
def compare():
    pngs = [os.path.basename(p) for p in glob(str(COMPARE_DIR / "*.png"))]
    return render_template("compare.html", pngs=pngs)

@app.route("/visualize")
def visualize():
    return render_template("visualize.html")

if __name__ == "__main__":
    app.run(debug=True)
