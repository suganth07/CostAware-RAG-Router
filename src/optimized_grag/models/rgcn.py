from __future__ import annotations

import random
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCNLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_relations: int,
        dropout: float = 0.1,
        message_batch_size: int = 16_384,
    ) -> None:
        super().__init__()
        if message_batch_size < 1:
            raise ValueError("message_batch_size must be at least 1")
        self.self_loop = nn.Linear(dim, dim)
        self.rel_weights = nn.Parameter(torch.randn(num_relations, dim, dim) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.message_batch_size = int(message_batch_size)

    def forward(self, x, edge_index, edge_type):
        src, dst = edge_index
        out = torch.zeros_like(x)
        # Indexing rel_weights with every edge creates [E, D, D]. With D=64,
        # 400k edges require ~6.1 GiB before gradients. Grouping by relation
        # computes [batch, D] messages instead and bounds the largest temporary
        # independently of graph size.
        for relation in torch.unique(edge_type, sorted=True):
            positions = torch.nonzero(edge_type == relation, as_tuple=False).flatten()
            weight = self.rel_weights[relation]
            for start in range(0, positions.numel(), self.message_batch_size):
                selected = positions[start : start + self.message_batch_size]
                messages = x[src[selected]] @ weight
                out.index_add_(0, dst[selected], messages)
        degree = torch.bincount(dst, minlength=x.size(0)).to(dtype=x.dtype)
        out = out / degree.clamp_min(1).unsqueeze(1)
        return self.dropout(F.relu(out + self.self_loop(x)))


class CapsuleRGCN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        *,
        dim: int = 64,
        layers: int = 2,
        message_batch_size: int = 16_384,
    ) -> None:
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, dim)
        self.layers = nn.ModuleList(
            [
                RGCNLayer(
                    dim,
                    num_relations * 2,
                    message_batch_size=message_batch_size,
                )
                for _ in range(layers)
            ]
        )
        self.relation_diag = nn.Embedding(num_relations, dim)

    def encode(self, edge_index, edge_type):
        x = self.node_emb.weight
        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
        return F.normalize(x, dim=-1)

    def score_triples(self, node_emb, left, relation, right):
        return (
            node_emb[left] * self.relation_diag(relation) * node_emb[right]
        ).sum(dim=-1)


def split_edges(
    edge_rows: Sequence[tuple[int, int, int]],
    *,
    seed: int = 42,
    val_ratio: float = 0.1,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    shuffled = list(edge_rows)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    if val_size >= len(shuffled):
        val_size = 1
    return shuffled[val_size:], shuffled[:val_size]


def build_message_tensors(
    edge_rows: Sequence[tuple[int, int, int]],
    *,
    num_relations: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward = [(source, target, relation) for source, target, relation in edge_rows]
    inverse = [
        (target, source, relation + num_relations)
        for source, target, relation in edge_rows
    ]
    rows = forward + inverse
    edge_index = torch.tensor(
        [[source, target] for source, target, _ in rows],
        dtype=torch.long,
        device=device,
    ).T
    edge_type = torch.tensor(
        [relation for _, _, relation in rows],
        dtype=torch.long,
        device=device,
    )
    return edge_index, edge_type
