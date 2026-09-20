"""SHAP explanations for the federated recommender."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import torch

from models.ncf import NeuMF


def _score(model, X):
    """Model forward pass over a numpy (user_id, item_id) matrix."""
    arr = np.asarray(X)
    users = torch.as_tensor(arr[:, 0], dtype=torch.long)
    items = torch.as_tensor(arr[:, 1], dtype=torch.long)
    with torch.no_grad():
        return model(users, items).numpy()


def explain_predictions(model: NeuMF, X: np.ndarray,
                        nsamples: int = 200, seed: int = 0):
    """Return a Partition-mask SHAP explainer over the feature matrix ``X``."""
    np.random.seed(seed)
    masker = shap.maskers.Partition(X)
    return shap.Explainer(lambda xb: _score(model, xb), masker)


def summarize(explainer, X_sample: np.ndarray,
              feature_names: list[str]) -> pd.DataFrame:
    """Mean absolute SHAP per feature for the sample."""
    sv = explainer(X_sample)
    means = np.abs(np.asarray(sv.values)).mean(axis=0)
    return pd.DataFrame(
        {f: m for f, m in zip(feature_names, means)}, index=["mean|SHAP"]
    )