"""Gradient-inversion attacks (Deep Leakage from Gradients).

Given the gradients a client shared during federated training, an attacker
tries to reconstruct the client's private (user, item) signals by optimizing
a dummy input to match those gradients. Used to audit whether DP or HE actually
block data leakage.
"""

from __future__ import annotations

import torch
from torch import nn


def gradient_inversion_attack(model: nn.Module, target_grads, num_users: int,
                              num_items: int, steps: int = 300, lr: float = 0.05,
                              seed: int = 0) -> torch.Tensor:
    """Reconstruct a (user, item) pair whose gradients match ``target_grads``.

    ``target_grads``: list of tensors, one per model parameter (in model order),
    e.g. captured from the client audit hook before aggregation.

    Returns a LongTensor ``[2]`` with the recovered (user, item) ids.
    """
    torch.manual_seed(seed)
    model.train()

    cand_users = torch.nn.Parameter(torch.rand(1) * num_users, requires_grad=True)
    cand_items = torch.nn.Parameter(torch.rand(1) * num_items, requires_grad=True)
    opt = torch.optim.Adam([cand_users, cand_items], lr=lr)

    params = list(model.parameters())

    def matching_loss(u, i):
        users = u.round().long()
        items = i.round().long()
        out = model(users, items)
        model.zero_grad()
        grads = torch.autograd.grad(out.sum(), params, retain_graph=True)
        return sum(
            torch.nn.functional.mse_loss(g, t.float())
            for g, t in zip(grads, target_grads)
        )

    for _ in range(steps):
        opt.zero_grad()
        loss = matching_loss(cand_users, cand_items)
        loss.backward()
        opt.step()
        with torch.no_grad():
            cand_users.clamp_(0, num_users - 1)
            cand_items.clamp_(0, num_items - 1)

    return torch.stack([cand_users.round().long(), cand_items])


def capture_gradients(model: nn.Module, users: torch.Tensor, items: torch.Tensor,
                      labels: torch.Tensor) -> list[torch.Tensor]:
    """Capture the per-batch gradients a client would share during fit."""
    model.train()
    model.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy(model(users, items), labels)
    grads = torch.autograd.grad(loss, list(model.parameters()))
    return [g.detach() for g in grads]