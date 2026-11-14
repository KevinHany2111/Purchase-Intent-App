"""Helper functions shared by the notebook and the web app."""

import numpy as np

COUNT_COLS = ["Administrative", "Informational", "ProductRelated"]
DURATION_COLS = [
    "Administrative_Duration",
    "Informational_Duration",
    "ProductRelated_Duration",
]


def add_features(X):
    """Clean durations, add total columns and log transform the skewed columns."""
    X = X.copy()

    # a duration can not be negative, so we set those values to 0
    for col in DURATION_COLS:
        X[col] = X[col].clip(lower=0)

    # new features
    X["TotalPages"] = X[COUNT_COLS].sum(axis=1)
    X["TotalDuration"] = X[DURATION_COLS].sum(axis=1)

    # log1p makes very large values smaller (outliers)
    for col in COUNT_COLS + DURATION_COLS + ["TotalPages", "TotalDuration"]:
        X[col] = np.log1p(X[col])

    return X


def suggest_action(prob, threshold):
    """Turn a purchase probability into one of three actions."""
    if prob >= threshold:
        return "No action"
    elif prob >= threshold / 2:
        return "Show coupon"
    else:
        return "Show chat"
