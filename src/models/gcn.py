from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GraphBatch:
    """
    A padded batch of graphs.

    x: (B, N, F) node features
    adj: (B, N, N) adjacency matrix (0/1), including self-loops is OK
    mask: (B, N) 1 for valid nodes, 0 for padding
    """

    x: torch.Tensor
    adj: torch.Tensor
    mask: torch.Tensor


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B,N,F)
        adj: (B,N,N) unnormalized 0/1
        mask: (B,N) 1/0
        """
        # Add self-loops (safe even if already present).
        b, n, _ = x.shape
        eye = torch.eye(n, device=x.device, dtype=adj.dtype).unsqueeze(0).expand(b, -1, -1)
        a = adj + eye

        # Mask out padding rows/cols.
        node_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B,N,N)
        a = a * node_mask

        # Normalize: D^{-1/2} A D^{-1/2}
        deg = a.sum(dim=-1).clamp_min(1.0)  # (B,N)
        deg_inv_sqrt = deg.pow(-0.5)
        a_norm = a * deg_inv_sqrt.unsqueeze(2) * deg_inv_sqrt.unsqueeze(1)

        h = self.lin(x)
        h = torch.bmm(a_norm, h)
        h = F.relu(h)
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        # Keep padding nodes at 0 for stability
        return h * mask.unsqueeze(-1)


class GraphEncoderGCN(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if pooling not in {"mean", "sum", "max"}:
            raise ValueError("pooling must be one of: mean, sum, max")

        dims = [node_feat_dim] + [hidden_dim] * (num_layers - 1) + [emb_dim]
        self.layers = nn.ModuleList(
            [GCNLayer(dims[i], dims[i + 1], dropout=dropout if i < num_layers - 1 else 0.0) for i in range(num_layers)]
        )
        self.pooling = pooling

    def forward(self, batch: GraphBatch | dict) -> torch.Tensor:
        if isinstance(batch, dict):
            x, adj, mask = batch["x"], batch["adj"], batch["mask"]
        else:
            x, adj, mask = batch.x, batch.adj, batch.mask

        h = x
        for layer in self.layers:
            h = layer(h, adj, mask)

        # Pool nodes -> graph embedding
        if self.pooling == "sum":
            g = (h * mask.unsqueeze(-1)).sum(dim=1)
        elif self.pooling == "max":
            # put very negative values for padded nodes so they don't win the max
            neg_inf = torch.finfo(h.dtype).min
            h_masked = h.masked_fill(mask.unsqueeze(-1).eq(0), neg_inf)
            g = h_masked.max(dim=1).values
        else:  # mean
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            g = (h * mask.unsqueeze(-1)).sum(dim=1) / denom

        return F.normalize(g, dim=1)

