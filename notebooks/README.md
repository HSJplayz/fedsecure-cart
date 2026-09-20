# Colab-facing Notes

These notebooks are meant to run in Google Colab, not locally.

1. Open `federated_cart.ipynb` in Colab: `File > Upload notebook`, or sync via
   the GitHub page.
2. Run the first cell to install dependencies (`torch`, `flwr`, `opacus`,
   `shap`, `hydra-core`).
3. Each notebook calls into the repo modules via `import sys;
   sys.path.append('/content/fedsecure-cart')` (upload or `!git clone` first).

Notebooks:
- `federated_cart.ipynb` — end-to-end federated NCF simulation with DP+SecAgg.
- `attack_audit.ipynb` — MIA and gradient-inversion probes on shared updates.

Keep heavy compute here minimal: toy dataset, 5 clients, ~10 rounds.