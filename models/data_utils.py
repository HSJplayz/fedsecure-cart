"""Interaction-matrix + negative-sampling utilities for NCF training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class InteractionData:
    user_ids: torch.Tensor
    item_ids: torch.Tensor
    labels: torch.Tensor
    num_users: int
    num_items: int


def build_interactions(
    df: pd.DataFrame, user_col="user_id", item_col="item_id"
) -> InteractionData:
    """Build user/item id tensors from a cart-event frame.

    Expected columns: ``user_id``, ``item_id`` (one row = one cart interaction).
    """
    users = df[user_col].astype("category")
    items = df[item_col].astype("category")
    codes_u = users.cat.codes.to_numpy()
    codes_i = items.cat.codes.to_numpy()
    return InteractionData(
        user_ids=torch.as_tensor(codes_u, dtype=torch.long),
        item_ids=torch.as_tensor(codes_i, dtype=torch.long),
        labels=torch.ones(len(df), dtype=torch.float32),
        num_users=len(users.cat.categories),
        num_items=len(items.cat.categories),
    )


def negative_sample(data: InteractionData, negatives_per_positive: int = 4) -> InteractionData:
    """Pair positive rows with uniformly sampled negative (item, 0) pairs."""
    rng = np.random.default_rng(0)
    pos_u, pos_i = data.user_ids.numpy(), data.item_ids.numpy()

    u_ids = np.repeat(pos_u, negatives_per_positive + 1)
    i_ids = np.repeat(pos_i, negatives_per_positive + 1)
    labels = np.zeros(len(u_ids), dtype=np.float32)

    for start in range(0, len(u_ids), negatives_per_positive + 1):
        labels[start] = 1.0
        for k in range(1, negatives_per_positive + 1):
            i_ids[start + k] = rng.integers(0, data.num_items)

    return InteractionData(
        user_ids=torch.as_tensor(u_ids, dtype=torch.long),
        item_ids=torch.as_tensor(i_ids, dtype=torch.long),
        labels=torch.as_tensor(labels),
        num_users=data.num_users,
        num_items=data.num_items,
    )


def random_split(data: InteractionData, frac: float = 0.8):
    """Split the interaction tensor into (train, val) by row index."""
    perm = torch.randperm(len(data.labels))
    cut = int(len(perm) * frac)

    def take(idx):
        return InteractionData(
            user_ids=data.user_ids[idx],
            item_ids=data.item_ids[idx],
            labels=data.labels[idx],
            num_users=data.num_users,
            num_items=data.num_items,
        )

    return take(perm[:cut]), take(perm[cut:])