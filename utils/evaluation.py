import numpy as np
from sklearn.metrics import f1_score


def competition_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_binary = (y_true < 8).astype(int)
    pred_binary = (y_pred < 8).astype(int)
    binary_f1 = f1_score(true_binary, pred_binary, zero_division=0)

    # Collapse all 10 non-target gestures into class 8.
    true_nine = np.where(y_true < 8, y_true, 8)
    pred_nine = np.where(y_pred < 8, y_pred, 8)
    macro_f1 = f1_score(
        true_nine, pred_nine,
        labels=list(range(9)),
        average="macro",
        zero_division=0,
    )

    return {
        "binary_f1": binary_f1,
        "macro_f1": macro_f1,
        "score": (binary_f1 + macro_f1) / 2,
    }