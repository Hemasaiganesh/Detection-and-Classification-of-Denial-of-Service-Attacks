"""Training script for WSN-DS lightweight DoS detection."""

import argparse
import os
import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

from src.utils import infer_label_column, to_binary_labels, load_csv_any


def split_features_labels(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataset into features (X) and labels (y)."""
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y


def build_model(X_train: pd.DataFrame, model_name: str, gini_threshold: float) -> Pipeline:
    """Build ML pipeline with preprocessing, feature selection, and classifier."""
    numeric_features = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])

    # Feature selection with Decision Tree + Gini importance
    selector_estimator = DecisionTreeClassifier(criterion="gini", random_state=42)
    selector = SelectFromModel(selector_estimator, threshold=gini_threshold, prefit=False)

    # Choose classifier
    if model_name == "dt":
        clf = DecisionTreeClassifier(criterion="gini", random_state=42)
    elif model_name == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    elif model_name == "xgb":
        clf = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, objective="multi:softprob", n_jobs=-1, eval_metric="mlogloss"
        )
    elif model_name == "knn":
        clf = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError("--model must be one of: dt, rf, xgb, knn")

    return Pipeline([("pre", pre), ("select", selector), ("clf", clf)])


def run_cli():
    """Command-line entry point for training and evaluation."""
    ap = argparse.ArgumentParser(description="WSN-DS Lightweight ML Trainer")
    ap.add_argument("--data", required=True, help="Path to dataset CSV/ARFF")
    ap.add_argument("--label-col", default=None, help="Label column")
    ap.add_argument("--binary", action="store_true", help="Binary labels (0=normal,1=attack)")
    ap.add_argument("--cv-folds", type=int, default=10, help="Cross-validation folds")
    ap.add_argument("--threshold", type=float, default=0.01, help="Gini threshold")
    ap.add_argument("--model", choices=["dt","rf","xgb","knn"], default="dt", help="Classifier")
    ap.add_argument("--save-dir", default="./artifacts", help="Save directory")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset
    df = load_csv_any(args.data)
    label_col = args.label_col or infer_label_column(df)
    if not label_col:
        raise SystemExit("Could not infer label column; use --label-col")

    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    X, y = split_features_labels(df, label_col)
    if args.binary:
        y = to_binary_labels(y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build model
    model = build_model(X_train, args.model, args.threshold)

    # Cross-validation
    print("Starting cross-validation...")
    scoring = ["accuracy","precision_macro","recall_macro","f1_macro"]
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    print("[CV] Results:")
    for k in scoring:
        vals = cv_results[f"test_{k}"]
        print(f"{k:>16}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Train on train split
    model.fit(X_train, y_train)

    # Holdout evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print("[Holdout] Metrics:")
    print(f"      Accuracy: {acc:.4f}")
    print(f"Precision_macro: {prec:.4f}")
    print(f"   Recall_macro: {rec:.4f}")
    print(f"       F1_macro: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Save model
    model_path = os.path.join(args.save_dir, f"wsnds_{args.model}_gini{args.threshold}.joblib")
    joblib.dump(model, model_path)
    print(f"[INFO] Saved model: {model_path}")
