# fedsecure-cart

Secure **federated recommender** research build on cart/purchase signals.
Trains a neural collaborative-filtering (NCF) model across decentralized
clients (Flower) with differential privacy (Opacus), secure aggregation and
optional homomorphic encryption, and audits the system with privacy attacks
(MIA, gradient inversion) and model explanation (SHAP).

## Why federated + secure

Cart data is personal. We never move raw cart events off the client; clients
only share encrypted, differentially-private model updates.

## Layout

```
fedsecure-cart/
├── models/          # NCF (GMF / MLP / NeuMF), interaction-matrix utils
├── fl/              # Flower clients, federated dataset partitioning, server
├── privacy/         # Opacus DP, secure aggregation (SecAgg), HE aggregation
├── attacks/         # membership inference (MIA), gradient-inversion attacks
├── xai/             # SHAP explanation helpers
├── configs/         # Hydra / YAML experiment configs
├── experiments/     # driver scripts (training, attacks, XAI)
├── notebooks/       # Colab-facing only
└── data/            # dataset notes / placeholders (git-ignored payloads)
```

## Quickstart (local simulation)

```bash
pip install -r requirements.txt
python -m experiments.run_fed               # 1-process federated simulation w/ DP
python -m experiments.run_attacks           # MIA + gradient inversion probes
python -m experiments.run_xai               # SHAP feature attributions
```

## Security stack

| Concern       | Tool/technique               |
|---------------|------------------------------|
| Federated orchestration | Flower |
| Differential privacy    | Opacus (RDP accountant) |
| Confidential aggregation| SecAgg masks + optional Homomorphic Encryption (TenSEAL) |
| Attack audit           | Membership inference, Deep Leakage (gradient inversion) |
| Explainability         | SHAP (model-agnostic / per-feature) |

## Colab

The `notebooks/` folder is Colab-facing: open `federated_cart.ipynb` in
Google Colab, install the requirements cell, and run the simulation end-to-end
in browser (no GPU needed for the toy dataset).