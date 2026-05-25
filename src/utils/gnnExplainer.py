import os
import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# Wrapper modelli per GNNExplainer
# ---------------------------------------------------------------------------
class GCNWrapper(torch.nn.Module):
    """Wrapper GCN per GNNExplainer: assume un singolo grafo (no batch)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index):
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


# ---------------------------------------------------------------------------
# Builder explainer
# ---------------------------------------------------------------------------
def build_explainer(wrapped_model, epochs=200):
    """Crea un Explainer GNNExplainer per un modello wrappato."""
    return Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="regression",
            task_level="graph",
            return_type="raw",
        ),
    )


# ---------------------------------------------------------------------------
# Utility vocabolari
# ---------------------------------------------------------------------------
def get_node_names(data, node_vocab):
    """Recupera i nomi dei nodi dal vocabolario."""
    idx2node   = {v: k for k, v in node_vocab.items()}
    node_types = data.x[:, 0].long().cpu().numpy()
    return [idx2node.get(int(t), f"node_{t}") for t in node_types]


def get_edge_labels_gcn(edge_index, node_names):
    """Edge label GCN: solo topologia src → dst."""
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    return [f"{node_names[s]} → {node_names[d]}" for s, d in zip(src, dst)]


def get_edge_labels_gine(edge_index, edge_attr, node_names, rel_vocab):
    """Edge label GINE: src →[tipo]→ dst."""
    idx2rel = {v: k for k, v in rel_vocab.items()}
    src     = edge_index[0].cpu().numpy()
    dst     = edge_index[1].cpu().numpy()
    rels    = edge_attr.cpu().numpy() if edge_attr.dim() == 1 else edge_attr[:, 0].cpu().numpy()
    return [
        f"{node_names[s]} →[{idx2rel.get(int(r), '?')}]→ {node_names[d]}"
        for s, d, r in zip(src, dst, rels)
    ]


# ---------------------------------------------------------------------------
# Visualizzazione
# ---------------------------------------------------------------------------
def show_image(image_id, image_dir, label, ax):
    """Mostra l'immagine corrispondente al grafo su un ax esistente."""
    img_path = os.path.join(image_dir, f"{image_id}.jpg")
    img = Image.open(img_path).convert("RGB")
    ax.imshow(img)
    ax.set_title(label, fontsize=9)
    ax.axis("off")


def plot_comparison(
    idx,
    df_val,
    image_dir,
    idx2label,
    val_dataset_gcn,
    val_dataset_gine,
    node_vocab_gcn,
    node_vocab_gine,
    rel_vocab_gine,
    explainer_gcn,
    explainer_gine,
    device,
    topk=8,
):
    """
    Plot 2x3 per un singolo campione:
      riga 0: GCN  — immagine | node importance | edge importance (topologia)
      riga 1: GINE — immagine | node importance | edge importance (tipo relazione)

    Args:
        idx            : indice nel val dataset
        df_val         : dataframe di validazione
        image_dir      : path alla cartella immagini
        idx2label      : dizionario idx -> label string
        val_dataset_gcn/gine : dataset di validazione
        node_vocab_gcn/gine  : vocabolari nodi
        rel_vocab_gine       : vocabolario relazioni GINE
        explainer_gcn/gine   : Explainer già configurati
        device         : torch device
        topk           : numero di nodi/edge da mostrare
    """
    label    = idx2label.get(int(val_dataset_gcn[idx].y.item()), "?")
    image_id = df_val.iloc[idx]["image_id"]

    data_gcn  = val_dataset_gcn[idx].to(device)
    data_gine = val_dataset_gine[idx].to(device)

    names_gcn  = get_node_names(data_gcn,  node_vocab_gcn)
    names_gine = get_node_names(data_gine, node_vocab_gine)

    exp_gcn  = explainer_gcn(x=data_gcn.x,  edge_index=data_gcn.edge_index)
    exp_gine = explainer_gine(x=data_gine.x, edge_index=data_gine.edge_index,
                               edge_attr=data_gine.edge_attr)

    # importanza nodi
    imp_nodes_gcn  = exp_gcn.node_mask.mean(dim=1).cpu().numpy()
    imp_nodes_gine = exp_gine.node_mask.mean(dim=1).cpu().numpy()
    top_nodes_gcn  = np.argsort(imp_nodes_gcn)[::-1][:topk]
    top_nodes_gine = np.argsort(imp_nodes_gine)[::-1][:topk]

    # importanza edge
    imp_edges_gcn  = exp_gcn.edge_mask.cpu().numpy()
    imp_edges_gine = exp_gine.edge_mask.cpu().numpy()
    top_edges_gcn  = np.argsort(imp_edges_gcn)[::-1][:topk]
    top_edges_gine = np.argsort(imp_edges_gine)[::-1][:topk]

    edge_labels_gcn  = get_edge_labels_gcn(data_gcn.edge_index, names_gcn)
    edge_labels_gine = get_edge_labels_gine(
        data_gine.edge_index, data_gine.edge_attr, names_gine, rel_vocab_gine
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # riga 0: GCN
    show_image(image_id, image_dir, f"{label}\n(GCN)", ax=axes[0, 0])

    axes[0, 1].barh(
        [names_gcn[i] for i in top_nodes_gcn][::-1],
        imp_nodes_gcn[top_nodes_gcn][::-1],
        color=cm.Blues(imp_nodes_gcn[top_nodes_gcn][::-1] / (imp_nodes_gcn.max() + 1e-8))
    )
    axes[0, 1].set_title("GCN — Node Importance")
    axes[0, 1].set_xlabel("Importanza")

    axes[0, 2].barh(
        [edge_labels_gcn[i] for i in top_edges_gcn][::-1],
        imp_edges_gcn[top_edges_gcn][::-1],
        color=cm.Blues(imp_edges_gcn[top_edges_gcn][::-1] / (imp_edges_gcn.max() + 1e-8))
    )
    axes[0, 2].set_title("GCN — Edge Importance (topologia)")
    axes[0, 2].set_xlabel("Importanza")

    # riga 1: GINE
    show_image(image_id, image_dir, f"{label}\n(GINE)", ax=axes[1, 0])

    axes[1, 1].barh(
        [names_gine[i] for i in top_nodes_gine][::-1],
        imp_nodes_gine[top_nodes_gine][::-1],
        color=cm.Oranges(imp_nodes_gine[top_nodes_gine][::-1] / (imp_nodes_gine.max() + 1e-8))
    )
    axes[1, 1].set_title("GINE — Node Importance")
    axes[1, 1].set_xlabel("Importanza")

    axes[1, 2].barh(
        [edge_labels_gine[i] for i in top_edges_gine][::-1],
        imp_edges_gine[top_edges_gine][::-1],
        color=cm.Oranges(imp_edges_gine[top_edges_gine][::-1] / (imp_edges_gine.max() + 1e-8))
    )
    axes[1, 2].set_title("GINE — Edge Importance (tipo relazione)")
    axes[1, 2].set_xlabel("Importanza")

    plt.suptitle(f"GCN vs GINE — scena {idx}: {label}", fontsize=13)
    plt.tight_layout()
    plt.show()