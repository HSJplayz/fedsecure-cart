"""Neural Collaborative Filtering (He et al., 2017): GMF, MLP and NeuMF."""

from collections import OrderedDict

import torch
from torch import nn


class GMF(torch.nn.Module):
    """Generalized Matrix Factorization: element-wise product of embeddings."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.predictor = nn.Linear(embedding_dim, 1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        z = self.user_emb(user_ids) * self.item_emb(item_ids)
        return torch.sigmoid(self.predictor(z).squeeze(-1))


class MLP(torch.nn.Module):
    """Multi-layer perceptron tower over concatenated user/item embeddings."""

    def __init__(self, num_users: int, num_items: int, layers=(64, 32, 16, 8)):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, layers[0] // 2)
        self.item_emb = nn.Embedding(num_items, layers[0] // 2)
        tower = OrderedDict()
        prev = layers[0]
        for i, hidden in enumerate(layers[1:]):
            tower[f"fc{i}"] = nn.Linear(prev, hidden)
            tower[f"act{i}"] = nn.ReLU()
            tower[f"do{i}"] = nn.Dropout(0.2)
            prev = hidden
        self.tower = nn.Sequential(tower)
        self.predictor = nn.Linear(prev, 1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.user_emb(user_ids), self.item_emb(item_ids)], dim=-1)
        return torch.sigmoid(self.predictor(self.tower(x)).squeeze(-1))


class NeuMF(torch.nn.Module):
    """Neural Matrix Factorization merging GMF and MLP branches (concatenation)."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32,
                 mlp_layers=(64, 32, 16, 8), alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

        # GMF branch
        self.gmf_user = nn.Embedding(num_users, embedding_dim)
        self.gmf_item = nn.Embedding(num_items, embedding_dim)

        # MLP branch
        self.mlp_user = nn.Embedding(num_users, mlp_layers[0] // 2)
        self.mlp_item = nn.Embedding(num_items, mlp_layers[0] // 2)
        tower = OrderedDict()
        prev = mlp_layers[0]
        for i, hidden in enumerate(mlp_layers[1:]):
            tower[f"fc{i}"] = nn.Linear(prev, hidden)
            tower[f"act{i}"] = nn.ReLU()
            prev = hidden
        self.tower = nn.Sequential(tower)
        self.predictor = nn.Linear(embedding_dim + prev, 1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        gmf = self.gmf_user(user_ids) * self.gmf_item(item_ids)
        mlp = self.tower(torch.cat([self.mlp_user(user_ids), self.mlp_item(item_ids)], dim=-1))
        out = self.predictor(torch.cat([gmf, mlp], dim=-1))
        return torch.sigmoid(out.squeeze(-1))


def build_model(model_type: str, num_users: int, num_items: int, **kwargs) -> nn.Module:
    if model_type == "gmf":
        return GMF(num_users, num_items, **kwargs)
    if model_type == "mlp":
        return MLP(num_users, num_items, **kwargs)
    if model_type == "neumf":
        return NeuMF(num_users, num_items, **kwargs)
    raise ValueError(f"unknown model_type: {model_type}")