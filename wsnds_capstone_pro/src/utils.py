"""Utility functions for WSN-DS capstone project."""

import os
import pandas as pd
from typing import Optional

# Possible label column names in datasets
POSSIBLE_LABELS = [
    "class", "Class", "label", "Label", "attack", "Attack",
    "attack_type", "Attack_type", "Attack type",  # ✅ added this
    "type", "Type", "target", "Target"
]

# Words that usually indicate attacks
ATTACK_STRINGS = {"attack", "dos", "flooding", "blackhole", "grayhole", "scheduling", "malicious"}


def infer_label_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to guess which column is the label/target.
    Priority:
      1. Exact match in POSSIBLE_LABELS
      2. Low-cardinality columns with values like 'normal', 'attack', '0', '1'
    """
    for c in POSSIBLE_LABELS:
        if c in df.columns:
            return c

    # fallback: low-cardinality columns
    candidates = []
    for c in df.columns:
        nunique = df[c].nunique(dropna=True)
        if nunique <= max(10, int(0.02 * len(df))):  # <=10 unique or <=2% of dataset size
            candidates.append((c, nunique))

    if candidates:
        for c, _ in candidates:
            vals = set(map(lambda x: str(x).lower(), df[c].dropna().unique()))
            if any(any(tok in v for tok in ATTACK_STRINGS) or v in {"0", "1", "normal"} for v in vals):
                return c
        # if nothing looks like an attack label, return smallest cardinality column
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    return None


def to_binary_labels(y: pd.Series) -> pd.Series:
    """
    Convert multi-class labels into binary {0=Normal, 1=Attack}.
    """
    def map_val(v):
        s = str(v).strip().lower()
        if s in {"0", "normal", "benign", "nominal"}:
            return 0
        return 1

    return y.apply(map_val)


def load_csv_any(path: str) -> pd.DataFrame:
    """
    Load CSV or ARFF dataset.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".arff":
        import arff
        data = arff.load(open(path, 'r'))
        attrs = [a[0] for a in data['attributes']]
        return pd.DataFrame(data['data'], columns=attrs)
    raise ValueError(f"Unsupported extension: {ext}")
