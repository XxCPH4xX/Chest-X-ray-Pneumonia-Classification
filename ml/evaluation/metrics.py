import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def classification_metrics(labels, scores, threshold: float = 0.5) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(labels) == 0 or len(labels) != len(scores):
        raise ValueError("Nonempty labels and equally sized scores required")
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid labels or scores")
    tn, fp, fn, tp = confusion_matrix(labels, scores >= threshold, labels=[0, 1]).ravel()

    def ratio(a, b):
        return float(a / b) if b else None

    recall = ratio(tp, tp + fn)
    return {
        "accuracy": ratio(tp + tn, len(labels)),
        "sensitivity": recall,
        "recall": recall,
        "specificity": ratio(tn, tn + fp),
        "precision": ratio(tp, tp + fp),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "support": {"NORMAL": int(tn + fp), "PNEUMONIA": int(tp + fn)},
    }
