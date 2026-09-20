"""Secure aggregation (SecAgg): pairwise masks cancel out the sum.

Reference architecture (single-server masked averaging for illustration):
    1. Clients agree on a pairwise random-mask layout over a ring.
    2. Each client adds its random masks so that all masks cancel in the sum.
    3. The server only ever sees the masked aggregate, never raw updates.

Masked updates stay correct: sum(masked_i) == sum(delta_i).
The mask dimensionality must match flattened model updates.
"""

from __future__ import annotations

import numpy as np


def masked_average(updates: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Reduce client updates with cancellation masks (SecAgg-style).

    Each local update is passed in pre-masked by the client (see
    ``fl.client.NCFClient``), so the server sums masked arrays and divides by
    the number of clients. For the demo the inputs are treated as already
    masked; use :func:`generate_masks` to simulate the honest-but-curious case.
    """
    if not updates:
        return []
    num_clients = len(updates)
    n_layers = len(updates[0])
    acc = [np.zeros_like(u, dtype=np.float64) for u in updates[0]]
    for upd in updates:
        for i, layer in enumerate(upd):
            acc[i] += layer.astype(np.float64)
    return [a / num_clients for a in acc]


def generate_masks(num_clients: int, shapes: list[tuple],
                   seed: int = 0) -> list[list[np.ndarray]]:
    """Deterministically assign pairwise masks that sum to zero across clients."""
    rng = np.random.default_rng(seed)
    masks = [[np.zeros(s, dtype=np.float32) for s in shapes]
             for _ in range(num_clients)]
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            for lyr in range(len(shapes)):
                m = rng.standard_normal(shapes[lyr]).astype(np.float32)
                masks[i][lyr] += m
                masks[j][lyr] -= m
    return masks