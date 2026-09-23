"""二分类总体指标、污染类型/等级漏检指标与混淆矩阵。 / Binary metrics and pollution-wise FNR metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

LABEL_ORDER = [0, 1]
LABEL_NAMES = ["clean", "contaminated"]


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def confusion_matrix_counts(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)


def pollution_metrics(records: pd.DataFrame, group_columns=("pollution_type", "severity")) -> pd.DataFrame:
    """按指定污染字段统计 Recall/FNR。 / Group contaminated samples and compute Recall/FNR."""
    contaminated = records[records["true_label"] == 1].copy()
    group_columns = tuple(group_columns)
    columns = [
        *group_columns, "samples", "true_positive",
        "false_negative", "recall", "fnr",
    ]
    if contaminated.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for group_key, group in contaminated.groupby(list(group_columns), dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        samples = len(group)
        fn = int((group["pred_label"] == 0).sum())
        tp = samples - fn
        row = {name: value for name, value in zip(group_columns, group_key)}
        row.update({
            "samples": samples,
            "true_positive": tp,
            "false_negative": fn,
            "recall": tp / samples if samples else np.nan,
            "fnr": fn / samples if samples else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)
