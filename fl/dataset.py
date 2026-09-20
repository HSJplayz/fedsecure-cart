"""Federated dataset partitioning: split interactions across clients."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from models.data_utils import build_interactions, negative_sample


def split_clients(
    df: pd.DataFrame,
    num_clients: int,
    iid: bool = True,
    negatives_per_positive: int = 4,
    seed: int = 0,
) -> list[TensorDataset]:
    """Partition cart events into ``num_clients`` local TensorDatasets.

    - ``iid=True``: user IDs are sharded round-robin across clients.
    - ``iid=False``: users are grouped into a few cart-size buckets and
      clients get one bucket each (realistic skew).
    """
    rng = np.random.default_rng(seed)
    users = np.sort(df["user_id"].astype("category").cat.codes.to_numpy())

    if iid:
        shards = np.array_split(rng.permutation(np.unique(users)), num_clients)
    else:
        order = np.argsort(pd.Series(users).value_counts().to_numpy())
        buckets = np.array_split(order, num_clients)
        shards = [[int(users[i]) for i in b] for b in buckets]

    shard_ids = set()
    for shard in shards:
        shard_ids.update(shard)

    client_data = []
    local = df[df["user_id"].astype("category").cat.codes.isin(shard_ids)]
    user_groups = local.groupby(local["user_id"].astype("category").cat.codes)

    for shard in shards:
        subset = pd.concat([user_groups.get_group(int(u)) for u in shard]).reset_index(drop=True)
        data = build_interactions(subset)
        data = negative_sample(data, negatives_per_positive)
        client_data.append(
            TensorDataset(data.user_ids, data.item_ids, data.labels)
        )
    return client_data