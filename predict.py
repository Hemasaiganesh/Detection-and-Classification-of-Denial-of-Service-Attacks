import argparse
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Make predictions using a trained WSN model")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--model", required=True, help="Path to saved .joblib model")
    parser.add_argument("--output", required=True, help="Path to save predictions CSV")
    parser.add_argument("--limit", type=int, default=None, help="Number of rows to predict (optional)")
    parser.add_argument("--encoder", required=False, help="Path to saved LabelEncoder (optional)")
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.data)
    if args.limit:
        df = df.head(args.limit)
    df.columns = df.columns.str.strip()
    
    # Keep original dataframe for output
    df_original = df.copy()

    # Detect target column
    target_col = next((col for col in ["label", "Attack type", "Attack_type"] if col in df.columns), None)
    y = None
    le = None

    if target_col:
        y = df[target_col]
        X = df.drop(columns=[target_col])
        if y.dtype == 'object':
            if args.encoder:
                try:
                    le = joblib.load(args.encoder)
                    y = le.transform(y)
                    print("[INFO] Loaded encoder and transformed true labels")
                except:
                    y = pd.factorize(y)[0]
                    print("[WARN] Could not load encoder. Using numeric factorize for labels")
            else:
                y = pd.factorize(y)[0]
        else:
            y = y.values
    else:
        X = df.copy()

    # Load model
    model = joblib.load(args.model)

    # Prepare features for prediction
    if hasattr(model, "feature_names_in_"):
        missing = set(model.feature_names_in_) - set(X.columns)
        extra = set(X.columns) - set(model.feature_names_in_)
        for col in missing:
            X[col] = 0
        if extra:
            print(f"[INFO] Ignoring extra columns: {extra}")
        X = X[model.feature_names_in_]

    # Convert object columns to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    # Make predictions
    preds = model.predict(X)

    # Add predictions to original dataframe
    df_original["prediction"] = preds

    # Compute correct column if labels exist
    if y is not None:
        df_original["correct"] = (y == preds)

        # Accuracy and metrics
        acc = accuracy_score(y, preds)
        print(f"\n[ACCURACY]: {acc*100:.2f}%")
        print("\nClassification Report:\n", classification_report(y, preds, zero_division=0))
        print("Confusion Matrix:\n", confusion_matrix(y, preds))

        os.makedirs("artifacts", exist_ok=True)

        # Plot confusion matrix
        plt.figure(figsize=(6,5))
        sns.heatmap(confusion_matrix(y, preds), annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig("artifacts/confusion_matrix.png")
        plt.close()
        print("[INFO] Confusion matrix saved as artifacts/confusion_matrix.png")

        # Plot ROC curve (binary)
        if len(np.unique(y)) == 2 and hasattr(model, "predict_proba"):
            probas = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, probas)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color="blue", lw=2, label="ROC curve (AUC = %0.2f)" % roc_auc)
            plt.plot([0,1],[0,1], color="red", lw=2, linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig("artifacts/roc_curve.png")
            plt.close()
            print("[INFO] ROC curve saved as artifacts/roc_curve.png")

    else:
        df_original["correct"] = False

    # Sort by id for readability
    df_sorted = df_original.sort_values(by="id")

    # Top 10 correct predictions
    correct_preds = df_sorted[df_sorted["correct"]]
    print("\n[Top 10 Correct Predictions]")
    print(correct_preds.head(10))

    # Top 10 wrong predictions
    wrong_preds = df_sorted[~df_sorted["correct"]]
    print("\n[Top 10 Wrong Predictions]")
    print(wrong_preds.head(10))

    # Save final predictions
    df_original.to_csv(args.output, index=False)
    print(f"[INFO] Predictions saved to {args.output}")

if __name__ == "__main__":
    main()
