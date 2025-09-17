# Capstone Project — DoS Attack Detection in WSN-DS (Lightweight ML)

This repository implements the reference paper’s lightweight machine-learning approach for detecting Denial-of-Service (DoS) attacks in Wireless Sensor Networks (WSNs).

## Features
- **Dataset**: WSN-DS (from Kasasbeh et al.)
- **Preprocessing**: Missing value handling, categorical encoding
- **Feature Selection**: Gini-based ranking with threshold (default 0.01)
- **Models**: Decision Tree (lightweight), Random Forest, XGBoost, KNN
- **Validation**: 10-fold Stratified Cross-Validation + holdout evaluation
- **Metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC AUC
- **Artifacts**: Model + report JSON saved in `artifacts/`

## Usage
```bash
python main.py --data ./data/WSN-DS.csv --cv-folds 10 --threshold 0.01 --model dt
```