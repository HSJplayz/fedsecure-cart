"""Flower client: trains the local NCF on its own cart interactions."""

from __future__ import annotations

import logging
from collections import OrderedDict

import flwr as fl
import torch
from torch import nn

from models.ncf import build_model
from privacy.dp import attach_dp

logger = logging.getLogger(__name__)


class NCFClient(fl.client.NumPyClient):
    """Federated client holding a private, local cart interaction set."""

    def __init__(self, cid: int, train_sets, model_type: str, num_users: int,
                 num_items: int, epochs: int = 1, lr: float = 1e-3,
                 dp_epsilon: float | None = None, batch_size: int = 64):
        self.cid = cid
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.num_users = num_users
        self.num_items = num_items
        self.model = build_model(model_type, num_users, num_items)
        self.train_loader = torch.utils.data.DataLoader(
            train_sets[cid], batch_size=batch_size, shuffle=True
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        self.engine = None
        if dp_epsilon is not None:
            self.engine = attach_dp(self.model, self.optimizer, dp_epsilon)

    def get_parameters(self, config):  # returns top-level params as lists
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        if self.engine:
            self.engine.enable()

        for _ in range(self.epochs):
            for users, items, labels in self.train_loader:
                self.optimizer.zero_grad()
                if self.engine:
                    # per-sample gradient clipping inside Opacus
                    self.engine.enable()
                loss = self.criterion(self.model(users, items), labels)
                loss.backward()
                self.optimizer.step()
            if self.engine:
                self.engine.step()

        # raw update = local fit minus incoming global params
        local = self.get_parameters(config)
        updated = [(torch.tensor(l) - torch.tensor(g)).numpy()
                   for l, g in zip(local, parameters)]
        if self.engine:
            eps = self.engine.get_epsilon(delta=1e-5)
            logger.info("client %s: approx epsilon=%.3f", self.cid, eps)
        return local, len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for users, items, labels in self.train_loader:
                out = self.model(users, items)
                loss += torch.nn.functional.binary_cross_entropy(
                    out, labels, reduction="sum"
                ).item()
                preds = (out > 0.5).long()
                correct += (preds == labels.long()).sum().item()
                total += len(labels)
        return float(loss) / total, total, {"acc": float(correct) / total}