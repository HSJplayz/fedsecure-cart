"""Flower server: aggregates client updates, optionally via secure aggregation."""

from __future__ import annotations

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from privacy.secagg import masked_average
from privacy.he import encrypt_and_aggregate


class SecureFedAvg(FedAvg):
    """FedAvg whose aggregate step routes raw updates through the privacy layer."""

    def __init__(self, use_secagg: bool = False, use_he: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.use_secagg = use_secagg
        self.use_he = use_he

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # flower may already aggregate via FedAvg; we re-aggregate raw updates.
        aggregated = super().aggregate_fit(server_round, results, failures)
        parameters, metrics = aggregated
        if parameters is None:
            return None, {}

        updates = [parameters_to_ndarrays(fit_res.parameters)
                   for _, fit_res in results]

        if self.use_secagg:
            aggregated_updates = masked_average(updates)
        elif self.use_he:
            aggregated_updates = encrypt_and_aggregate(updates)
        else:
            aggregated_updates = updates[0]
            for u in updates[1:]:
                aggregated_updates = [
                    a + b for a, b in zip(aggregated_updates, u)
                ]
            aggregated_updates = [u / len(updates) for u in aggregated_updates]

        # FedAvg would already have averaged; apply mean of means consistently.
        return ndarrays_to_parameters(aggregated_updates), metrics


def run_server(num_rounds: int = 10, fraction_fit: float = 1.0,
               use_secagg: bool = False, use_he: bool = False) -> None:
    """Launch the Flower server (start_server) for local simulation."""
    strategy = SecureFedAvg(
        use_secagg=use_secagg,
        use_he=use_he,
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,
    )
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=num_rounds),
                           strategy=strategy)