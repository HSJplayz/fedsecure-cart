"""Audit the trained model with MIA and gradient-inversion probes.

Usage:
    python -m experiments.run_attacks
"""

from __future__ import annotations

import logging

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from attacks.mia import calibrate_thresholds, logit_threshold_attack
from attacks.gradient_inversion import capture_gradients, gradient_inversion_attack
from fl.client import NCFClient
from models.ncf import build_model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("run_attacks")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    num_users, num_items = 50, 30
    model = build_model(cfg.model.type, num_users, num_items)

    # --- MIA (logit-threshold probe) on a synthetic public set ---
    public = torch.randint(0, num_users, (400,)), torch.randint(0, num_items, (400,)), \
        torch.ones(400)
    thresh = calibrate_thresholds(model, public)
    pairs = torch.stack([torch.randint(0, num_users, (100,)),
                         torch.randint(0, num_items, (100,))], dim=1)
    pred_membership = logit_threshold_attack(model, pairs, thresh)
    log.info("MIA predicted-membership rate: %.3f", pred_membership.mean())

    # --- Gradient inversion on a single shared gradient vector ---
    u = torch.tensor([7])
    i = torch.tensor([12])
    grad_vec = capture_gradients(model, u, i, torch.tensor([1.0]))
    rec = gradient_inversion_attack(
        model, grad_vec, num_users, num_items,
        steps=cfg.attack.grad_inversion_steps,
        lr=cfg.attack.grad_inversion_lr,
    )
    log.info("true (user,item)=(%d,%d) recovered=(%d,%d)",
             int(u[0]), int(i[0]), int(rec[0]), int(rec[1]))


if __name__ == "__main__":
    main()