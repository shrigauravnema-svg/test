import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# -----------------------------
# Config
# -----------------------------
BASE_TRAIN_PATH = "train-2.csv"
TARGET_COL = "LOAN_DEFAULT"

DATE_COLS = ["DATE_OF_BIRTH", "DISBURSAL_DATE"]
AGE_TEXT_COLS = ["AVERAGE_ACCT_AGE", "CREDIT_HISTORY_LENGTH"]
DROP_ID_COLS = ["UNIQUEID"]  # ID-like columns, not useful for learning

_AGE_PAT = re.compile(r"(?:(\d+)\s*yrs?)?\s*(?:(\d+)\s*mon)?", re.IGNORECASE)


# -----------------------------
# Feature Engineering Helpers
# -----------------------------
def age_text_to_months(x):
    """Convert strings like '2yrs 3mon' -> 27 months. Robust to blanks/NaN."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if not s:
        return np.nan
    s = s.replace(" ", "")
    m = _AGE_PAT.fullmatch(s)
    if not m:
        return np.nan
    yrs = int(m.group(1)) if m.group(1) else 0
    mon = int(m.group(2)) if m.group(2) else 0
    return yrs * 12 + mon


def safe_parse_date(series, fmt=None):
    """Parse date with flexible parsing. Returns datetime64; invalid -> NaT."""
    # Your file looks like '01-01-1984' (DD-MM-YYYY), but keep robust:
    if fmt:
        return pd.to_datetime(series, format=fmt, errors="coerce")
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready feature table (keeps target if present)."""
    out = df.copy()

    # Drop ID-like columns if present
    for c in DROP_ID_COLS:
        if c in out.columns:
            out = out.drop(columns=[c])

    # Date features
    if "DATE_OF_BIRTH" in out.columns:
        dob = safe_parse_date(out["DATE_OF_BIRTH"])
        # age in days relative to "today" (fixed at runtime)
        today = pd.Timestamp(datetime.now().date())
        out["AGE_DAYS"] = (today - dob).dt.days
        out["DOB_YEAR"] = dob.dt.year
        out = out.drop(columns=["DATE_OF_BIRTH"])

    if "DISBURSAL_DATE" in out.columns:
        disb = safe_parse_date(out["DISBURSAL_DATE"])
        out["DISBURSAL_YEAR"] = disb.dt.year
        out["DISBURSAL_MONTH"] = disb.dt.month
        out["DISBURSAL_DAY"] = disb.dt.day
        out = out.drop(columns=["DISBURSAL_DATE"])

    # Age-text columns -> months
    for c in AGE_TEXT_COLS:
        if c in out.columns:
            out[c + "_MONTHS"] = out[c].apply(age_text_to_months)
            out = out.drop(columns=[c])

    return out


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build preprocessing + model pipeline."""
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_cols),
            ("cat", categorical_tf, categorical_cols)
        ],
        remainder="drop"
    )

    clf = LogisticRegression(max_iter=2000, n_jobs=None)

    return Pipeline(steps=[("preprocess", pre), ("model", clf)])


def train_and_eval(df: pd.DataFrame, seed: int = 42):
    """Train model and return (model, metrics_dict)."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"Training data must include target column: {TARGET_COL}")

    feats = make_features(df)

    # Separate X/y
    y = pd.to_numeric(feats[TARGET_COL], errors="coerce")
    X = feats.drop(columns=[TARGET_COL])

    # Basic cleaning: drop rows where y is missing
    keep = ~y.isna()
    X = X.loc[keep].copy()
    y = y.loc[keep].astype(int)

    if len(X) < 200:
        st.warning("Training rows are quite low; metrics may be unstable.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train)

    # Predictions
    pred = pipe.predict(X_test)
    proba = None
    try:
        proba = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        proba = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }
    if proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return pipe, metrics


def pretty_metrics(m: dict) -> str:
    auc = m.get("roc_auc", None)
    auc_s = f"{auc:.4f}" if isinstance(auc, (int, float)) and auc is not None else "NA"
    return f"acc={m['accuracy']:.4f}, f1={m['f1']:.4f}, auc={auc_s}"


def predict_with_model(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    feats = make_features(df)
    has_target = TARGET_COL in feats.columns
    X = feats.drop(columns=[TARGET_COL]) if has_target else feats

    preds = model.predict(X)
    out = df.copy()
    out["PRED_LOAN_DEFAULT"] = preds

    # probas (if available)
    try:
        prob = model.predict_proba(X)[:, 1]
        out["PRED_PROB_DEFAULT_1"] = prob
    except Exception:
        pass

    # If labels exist, compute quick accuracy on uploaded file
    if has_target:
        y_true = pd.to_numeric(df[TARGET_COL], errors="coerce")
        mask = ~y_true.isna()
        if mask.any():
            out.attrs["eval_accuracy"] = float(accuracy_score(y_true[mask].astype(int), preds[mask]))

    return out


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Recurrent Retraining ML (Old vs New Models)", layout="wide")
st.title("Recurrent Retraining ML App (Old vs New Model Selector)")

# Initialize session state
if "master_train_df" not in st.session_state:
    st.session_state.master_train_df = None
if "models" not in st.session_state:
    st.session_state.models = []  # list of dicts: {name, model, metrics, rows}
if "selected_model_idx" not in st.session_state:
    st.session_state.selected_model_idx = 0

# Load base data once
@st.cache_data(show_spinner=False)
def load_base_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

with st.spinner("Loading base training dataset..."):
    base_df = load_base_df(BASE_TRAIN_PATH)

if st.session_state.master_train_df is None:
    st.session_state.master_train_df = base_df.copy()

# Train base model once (if not trained)
if len(st.session_state.models) == 0:
    with st.spinner("Training base (old) model..."):
        base_model, base_metrics = train_and_eval(st.session_state.master_train_df)
    st.session_state.models.append({
        "name": "Old/Base Model",
        "model": base_model,
        "metrics": base_metrics,
        "rows": int(len(st.session_state.master_train_df)),
    })

# Summary block
colA, colB = st.columns([1.2, 1])
with colA:
    st.subheader(" Current Models")
    for i, item in enumerate(st.session_state.models):
        st.write(f"**{i}. {item['name']}**  — {pretty_metrics(item['metrics'])}  — rows={item['rows']}")

with colB:
    st.subheader(" Choose Model for Next Prediction")
    options = [
        f"{i}. {m['name']} ({pretty_metrics(m['metrics'])})"
        for i, m in enumerate(st.session_state.models)
    ]
    st.session_state.selected_model_idx = st.selectbox(
        "Model Selector (old vs new)",
        options=range(len(options)),
        format_func=lambda i: options[i],
        index=min(st.session_state.selected_model_idx, len(options)-1)
    )

st.divider()

# -----------------------------
# Retraining section
# -----------------------------
st.subheader("Upload New Labeled Data (for Retraining)")
st.caption(f"Uploaded CSV must contain the same feature columns and include target column: `{TARGET_COL}`")

new_train_file = st.file_uploader("Upload New Training CSV", type=["csv"], key="new_train_csv")

if new_train_file is not None:
    try:
        new_df = pd.read_csv(new_train_file)
        if TARGET_COL not in new_df.columns:
            st.error(f"Uploaded training file missing required target column `{TARGET_COL}`.")
        else:
            st.write("Preview of new training data:")
            st.dataframe(new_df.head(10), use_container_width=True)

            if st.button("Add & Retrain New Model"):
                with st.spinner("Merging data and training new model..."):
                    # merge
                    st.session_state.master_train_df = pd.concat(
                        [st.session_state.master_train_df, new_df],
                        axis=0, ignore_index=True
                    )

                    # train new model on updated master
                    new_model, new_metrics = train_and_eval(st.session_state.master_train_df)

                    version = len(st.session_state.models)
                    st.session_state.models.append({
                        "name": f"New Model v{version}",
                        "model": new_model,
                        "metrics": new_metrics,
                        "rows": int(len(st.session_state.master_train_df)),
                    })

                    # auto-select latest model
                    st.session_state.selected_model_idx = len(st.session_state.models) - 1

                st.success(f"New Model v{version} trained successfully!")
                st.rerun()

    except Exception as e:
        st.error(f"Error reading/training from uploaded file: {e}")

st.divider()

# -----------------------------
# Prediction section
# -----------------------------
st.subheader(" Upload File for Prediction / Evaluation")
st.caption(
    f"- If uploaded file contains `{TARGET_COL}`, the app will compute accuracy for the selected model on that file.\n"
    "- If not, it will only output predictions."
)

pred_file = st.file_uploader("Upload CSV for Prediction", type=["csv"], key="pred_csv")

if pred_file is not None:
    try:
        pred_df = pd.read_csv(pred_file)
        st.write("Preview of prediction file:")
        st.dataframe(pred_df.head(10), use_container_width=True)

        selected = st.session_state.models[st.session_state.selected_model_idx]
        st.info(f"Using: **{selected['name']}**  — {pretty_metrics(selected['metrics'])}")

        if st.button("Run Prediction"):
            with st.spinner("Predicting..."):
                result_df = predict_with_model(selected["model"], pred_df)

            # Show evaluation accuracy if possible
            if "eval_accuracy" in result_df.attrs:
                st.success(f"Accuracy on uploaded evaluation file: **{result_df.attrs['eval_accuracy']:.4f}**")

            st.write("Prediction Output (top rows):")
            st.dataframe(result_df.head(50), use_container_width=True)

            # Download
            out_csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions CSV",
                data=out_csv,
                file_name="predictions_output.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.divider()

# -----------------------------
# Reset / Maintenance
# -----------------------------
st.subheader(" Reset (Optional)")
if st.button("Reset to Base Model Only"):
    st.session_state.master_train_df = base_df.copy()
    st.session_state.models = []
    st.session_state.selected_model_idx = 0

    with st.spinner("Training base model again..."):
        base_model, base_metrics = train_and_eval(st.session_state.master_train_df)

    st.session_state.models.append({
        "name": "Old/Base Model",
        "model": base_model,
        "metrics": base_metrics,
        "rows": int(len(st.session_state.master_train_df)),
    })
    st.success("Reset completed.")
    st.rerun()
