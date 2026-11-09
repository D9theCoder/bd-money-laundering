"""One metric implementation for every model (Spark MLlib and XGBoost), so values are comparable.
Input: numpy arrays of labels (0/1) and scores. PR-AUC is reported as average precision."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)


def threshold_max_f1(y, s):
    """Threshold rule from config (max F1 on validation). Returns (threshold, f1)."""
    p, r, t = precision_recall_curve(y, s)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)[:-1]
    if len(t) == 0:
        return 1.0, 0.0
    i = int(np.argmax(f1))
    return float(t[i]), float(f1[i])


def precision_at_k(y, s, k):
    """Precision among the k highest scores. Ties at the k-th score are resolved by their expected value
    (as if the tied rows were ordered at random), so a constant-score model gets the positive rate."""
    k = min(k, len(s))
    if k == 0:
        return 0.0, 0.0
    sk = np.sort(s)[::-1][k - 1]
    above, tied = s > sk, s == sk
    need = k - int(above.sum())
    hits = float(y[above].sum()) + need * float(y[tied].mean())
    return hits / k, round(hits, 2)


def evaluate(y, s, threshold, k):
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    single = len(np.unique(y)) < 2
    pk, hits = precision_at_k(y, s, k)
    return {
        "rows": int(len(y)), "positives": int(y.sum()),
        "pr_auc": None if single else float(average_precision_score(y, s)),
        "roc_auc": None if single else float(roc_auc_score(y, s)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        f"precision_at_{k}": pk, f"hits_at_{k}": hits,
        "threshold": float(threshold),
    }
