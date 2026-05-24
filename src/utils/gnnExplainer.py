
import sys
import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import Data

from src.datasets.gqa_graph_dataset import GQAGraphDataset, build_node_vocab, build_rel_vocab
from src.models.gnn import GCNEmbeddingNet, GINEEmbeddingNet
from src.evaluation.retrieval import evaluate_retrieval, print_metrics


# ============================================================
# CELLA 2 - Wrapper modelli per GNNExplainer
# (GNNExplainer vuole forward(x, edge_index) senza batch)
# ============================================================
class GCNWrapper(torch.nn.Module):
    """Wrapper GCN per GNNExplainer: assume un singolo grafo (no batch)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index):
        # batch = tutti zero (un solo grafo)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.model(x, edge_index, batch)


class GINEWrapper(torch.nn.Module):
    """Wrapper GINE per GNNExplainer: assume un singolo grafo (no batch)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_attr):
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.model(x, edge_index, edge_attr, batch, num_graphs=1)




# ============================================================
#  Funzione di visualizzazione importanza nodi
# ============================================================
def get_node_names(data, node_vocab):
    """Recupera i nomi dei nodi dal vocabolario."""
    idx2node = {v: k for k, v in node_vocab.items()}
    node_types = data.x[:, 0].long().cpu().numpy()
    return [idx2node.get(int(t), f"node_{t}") for t in node_types]


def plot_node_importance(explanation, node_names, title="Node Importance", topk=10):
    """
    Visualizza l'importanza dei nodi come bar chart.
    node_mask ha shape (N, F) — media sulle feature per ottenere importanza per nodo.
    """
    node_mask = explanation.node_mask              # (N, F)
    importance = node_mask.mean(dim=1).cpu().numpy()  # (N,)

    # ordina per importanza decrescente
    sorted_idx = np.argsort(importance)[::-1][:topk]
    sorted_names = [node_names[i] for i in sorted_idx]
    sorted_vals  = importance[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = cm.RdYlGn(sorted_vals / (sorted_vals.max() + 1e-8))
    ax.barh(sorted_names[::-1], sorted_vals[::-1], color=colors[::-1])
    ax.set_xlabel("Importanza")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_edge_importance(explanation, node_names, data, title="Edge Importance", topk=10):
    """Visualizza le relazioni (edge) più importanti."""
    edge_mask  = explanation.edge_mask.cpu().numpy()   # (E,)
    edge_index = data.edge_index.cpu().numpy()         # (2, E)

    # top-k edge per importanza
    sorted_idx = np.argsort(edge_mask)[::-1][:topk]

    labels = []
    vals   = []
    for i in sorted_idx:
        src = node_names[edge_index[0, i]]
        dst = node_names[edge_index[1, i]]
        labels.append(f"{src} → {dst}")
        vals.append(edge_mask[i])

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = cm.RdYlGn(np.array(vals) / (max(vals) + 1e-8))
    ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
    ax.set_xlabel("Importanza")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def show_image(image_id, label, ax=None):
    IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
    
    """Mostra l'immagine corrispondente al grafo."""
    img_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
    img = Image.open(img_path).convert("RGB")
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    ax.set_title(label, fontsize=9)
    ax.axis("off")
    if ax is None:
        plt.show()

