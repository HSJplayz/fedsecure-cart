"""Run the federated NCF training with the configured privacy layer.

Usage:
    python -m experiments.run_fed
    python -m experiments.run_fed --config-name=config dataset=cart \
        privacy.mode=dp+he rounds=20

Hydra overrides: privacy.mode, training.*, model.type, ...
"""

from __future__ import annotations

import logging

import flwr as fl

import hydra
from omegaconf import DictConfig

from fl.dataset import split_clients
from fl.client import NCFClient
from models.data_utils import build_interactions, negative_sample

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("run_fed")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import tempfile
    import pandas as pd

    # Toy in-memory dataset so the scaffold runs without real data.
    rng = __import__("numpy").random.default_rng(cfg.seed)
    n_users, n_items, n_events = 50, 30, 1500
    df = pd.DataFrame({
        "user_id": rng.integers(0, n_users, n_events),
        "item_id": rng.integers(0, n_items, n_events),
    })
    df = df.drop_duplicates(subset=["user_id", "item_id"]).reset_index(drop=True)

    data = build_interactions(df)
    global_count = (int(df["user_id"].nunique()), int(df["item_id"].nunique()))

    client_data = split_clients(
        df,
        num_clients=cfg.training.num_clients,
        iid=bool(cfg.training.iid),
        negatives_per_positive=cfg.training.negatives_per_positive,
    )

    def client_fn(cid: str):
        return NCFClient(
            cid=int(cid),
            train_sets=client_data,
            model_type=cfg.model.type,
            num_users=global_count[0],
            num_items=global_count[1],
            epochs=cfg.training.local_epochs,
            lr=cfg.training.lr,
            batch_size=cfg.training.batch_size,
            dp_epsilon=cfg.privacy.dp_epsilon
            if "dp" in cfg.privacy.mode else None,
        )

    use_secagg = "secagg" in cfg.privacy.mode
    use_he = "he" in cfg.privacy.mode

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.training.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.training.rounds),
        strategy=__import__("fl.server", fromlist=["SecureFedAvg"]).SecureFedAvg(
            use_secagg=use_secagg, use_he=use_he,
            fraction_fit=float(cfg.training.fraction_fit),
            fraction_evaluate=0.0,
        ),
    )
    log.info("final round losses: %s", list(history.losses_distributed)[-3:])


if __name__ == "__main__":
    main()