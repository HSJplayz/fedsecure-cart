"""SHAP attributions on a trained federated recommender.

Usage:
    python -m experiments.run_xai
"""

from __future__ import annotations

import logging

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from models.ncf import build_model
from xai.shap_explain import explain_predictions, summarize

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("run_xai")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    num_users, num_items = 50, 30
    model = build_model(cfg.model.type, num_users, num_items)

    X = np.column_stack([
        np.random.randint(0, num_users, cfg.xai.n_explain),
        np.random.randint(0, num_items, cfg.xai.n_explain),
    ])
    explainer = explain_predictions(model, X, nsamples=cfg.xai.nsamples)
    table = summarize(explainer, X[:20], feature_names=["user_id", "item_id"])
    log.info("mean |SHAP| attributions:\n%s", table.to_string())


if __name__ == "__main__":
    main()