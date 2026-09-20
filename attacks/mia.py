"""Membership-inference attacks against the federated recommender.

Two probes:
- ``logit_threshold_attack``: share your own item purchases never leave the
  client, so an attacker who only sees the final model tests whether a given
  (user, item) pair is part of the training set by comparing the model's
  confidence against a per-user threshold.
- ``shadow_model_attack`` (skeleton): trains shadow models on public data to
  calibrate attack scores, then applies them to the target model.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from models.ncf import NeuMF


def logit_threshold_attack(
    model: nn.Module, pairs: torch.Tensor, user_threshold: dict[int, float],
) -> np.ndarray:
    """Return predicted membership for each (user, item) pair.

    ``pairs``: LongTensor of shape (N, 2). A pair is "member" if the model
    confidence exceeds the user-specific decision threshold (calibrated on a
    held-out public dataset in practice).
    """
    model.eval()
    users, items = pairs[:, 0], pairs[:, 1]
    with torch.no_grad():
        probs = model(users, items).numpy()
    thresh = np.array([user_threshold.get(int(u), 0.5) for u in users.numpy()])
    return probs >= thresh


def calibrate_thresholds(model: nn.Module, public_data) -> dict[int, float]:
    """Fit per-user confidence thresholds from a public (non-member) set."""
    model.eval()
    users, items, labels = public_data
    user_scores: dict[int, list] = {}
    with torch.no_grad():
        probs = model(users, items).numpy()
        for u, i, l, p in zip(users.numpy(), items.numpy(), labels.numpy(), probs):
            if l == 1:
                user_scores.setdefault(int(u), []).append(p)
    return {u: np.percentile(score, 95) for u, score in user_scores.items()}


def shadow_model_attack(target_model: nn.Module, public_train, public_test,
                        num_shadow: int = 3, epochs: int = 1) -> tuple[float, float]:
    """Two-stage shadow evaluation: returns (members, nonmembers) mean scores.

    Skeleton: trains ``num_shadow`` copies on disjoint public splits and scores
    held-out members/non-members in a fixed feature space.
    """
    raise NotImplementedError(
        "Train NeuMF shadow models on public data; score target predictions.\n"
        "Replace this stub in experiments/run_attacks.py."
    )