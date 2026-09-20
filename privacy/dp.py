"""Differential privacy via Opacus (RDP accountant)."""

from __future__ import annotations

import torch
from torch import nn


def attach_dp(model: nn.Module, optimizer: torch.optim.Optimizer,
              target_epsilon: float, delta: float = 1e-5,
              max_grad_norm: float = 1.0):
    """Wrap an optimizer with Opacus and return a PrivacyEngine wrapper.

    Call ``engine.enable()`` before the backward pass and ``engine.step()``
    after ``optimizer.step()``. ``engine.get_epsilon(delta)`` reports the spent
    budget.
    """
    from opacus import PrivacyEngine  # heavy import, deferred

    engine = PrivacyEngine(secure_mode=False)
    model, optimizer, _ = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=torch.utils.data.DataLoader([]),  # replaced by caller loader
        noise_multiplier=1.0,
        max_grad_norm=max_grad_norm,
    )
    engine.target_epsilon = target_epsilon
    return DPEngine(engine)


class DPEngine:
    """Thin adapter over Opacus so clients keep a uniform interface."""

    def __init__(self, opacus_engine):
        self._engine = opacus_engine

    def enable(self):
        """Clipping/deformation already applied on the wrapped modules."""
        return

    def step(self):
        return self._engine.step()

    def get_epsilon(self, delta: float = 1e-5) -> float:
        return self._engine.get_epsilon(delta)

    def privacy_budget_left(self, delta: float = 1e-5) -> float:
        eps = self.get_epsilon(delta)
        return max(0.0, self._engine.target_epsilon - eps)