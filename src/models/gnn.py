import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool


class GINEEmbeddingNet(nn.Module):

    def __init__(
        self,
        num_node_types,
        edge_vocab_size,
        emb_dim=128,
        hidden_dim=256,
        num_layers=4,
        dropout=0.2,
        use_bbox=True,
        node_emb_dim=64,
        # backward compat (ignored if num_node_types is set)
        node_in_dim=None,
    ):
        super().__init__()

        if node_in_dim is not None and num_node_types is None:
            raise ValueError("Use num_node_types=len(node_vocab) instead of node_in_dim")

        self.use_bbox = use_bbox
        self.dropout = dropout

        self.node_emb = nn.Embedding(num_node_types, node_emb_dim, padding_idx=0)
        node_in = node_emb_dim + (4 if use_bbox else 0)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.edge_encoder = nn.Embedding(edge_vocab_size, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        pool_dim = hidden_dim * 2
        self.projector = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )

    def encode_nodes(self, x):
        types = x[:, 0].long().clamp_min(0)
        h = self.node_emb(types)

        if self.use_bbox:
            h = torch.cat([h, x[:, 1:5]], dim=1)

        return self.node_encoder(h)

    def forward(self, x, edge_index, edge_attr, batch, num_graphs=None):
        x = self.encode_nodes(x)
        edge_attr = self.edge_encoder(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        pool_size = num_graphs if num_graphs is not None else int(batch.max().item()) + 1
        mean_pool = global_mean_pool(x, batch, size=pool_size)
        max_pool = global_max_pool(x, batch, size=pool_size)
        x = torch.cat([mean_pool, max_pool], dim=1)

        x = self.projector(x)
        return F.normalize(x, dim=1)
