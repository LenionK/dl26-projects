import os
import warnings
import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
import pandas as pd
import seaborn as sns


#import torch
import torch.nn.functional as F

class GraphGradCAM:
    def __init__(self, model, target_layer, prototypes):
        """
        prototypes: dict {class_id: embedding tensor [D]}
        """
        self.model = model
        self.target_layer = target_layer
        self.prototypes = prototypes

        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    # ---------------- hooks ----------------
    def _save_activations(self, module, input, output):
        self.activations = output.detach().clone()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach().clone()

    # ---------------- main ----------------

    def compute_node_importance(self, x, edge_index, batch, target_class, edge_attr=None):

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        self.model.eval()

        self.activations = None
        self.gradients = None

        # forward pass
        try:
            if edge_attr is not None:
                z = self.model(x, edge_index, edge_attr, batch)
            else:
                z = self.model(x, edge_index, batch)
        except TypeError:
            z = self.model(x, edge_index, batch)

        z = F.normalize(z, dim=1)

        prototype = self.prototypes[target_class].to(z.device)
        prototype = F.normalize(prototype, dim=0)

        score = -torch.norm(z - prototype, p=2)

        self.model.zero_grad()
        score.backward()

        alpha = self.gradients.mean(dim=0, keepdim=True)
        node_weights = torch.sum(alpha * self.activations, dim=-1)
        node_importance = torch.relu(node_weights)

        max_val = node_importance.max()
        if max_val > 0:
            node_importance = node_importance / max_val
            

        return node_importance.cpu().numpy()
    
    def remove_hooks(self):
        if hasattr(self, "forward_hook"):
            self.forward_hook.remove()
        if hasattr(self, "backward_hook"):
            self.backward_hook.remove()

from collections import defaultdict
import torch
import torch.nn.functional as F

def compute_prototypes(model, loader, device):
    model.eval()

    class_embeddings = defaultdict(list)

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            labels = data.y
            batch = data.batch

            
            try:
                z = model(data.x, data.edge_index, data.edge_attr, batch)
            except TypeError:
                z = model(data.x, data.edge_index, batch)

            z = F.normalize(z, dim=1)

            for emb, y in zip(z, labels):
                class_embeddings[int(y)].append(emb.cpu())

    return {
        c: torch.stack(v).mean(dim=0)
        for c, v in class_embeddings.items()
    }
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
    """
    Crea un Explainer GNNExplainer per un modello wrappato.

    Nota: mode="regression" è impostato perché i modelli producono un embedding
    vettoriale (metric learning) e non una distribuzione di classe. GNNExplainer
    ottimizza la maschera rispetto alla singola coordinata di embedding indicata
    da target_class in compute_node_importance.
    """
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


# ---------------------------------------------------------------------------
# Pipeline di Visualizzazione e Confronto (Plot a 3 colonne)
# ---------------------------------------------------------------------------
def plot_gradcam_comparison(idx, df_val, image_dir, idx2label, val_dataset_gcn, val_dataset_gine,
                            node_vocab_gcn, node_vocab_gine, cam_gcn, cam_gine, device, topk=8):

    data_gcn  = val_dataset_gcn[idx].to(device)
    data_gine = val_dataset_gine[idx].to(device)

    row       = df_val.iloc[idx]
    image_id  = row['image_id']
    target_id = int(row['labels'])

    # Nome della classe reale (es. "Cane", "Automobile", ecc.) invece di "ID d'Embedding"
    target_name = idx2label.get(target_id, f"Classe {target_id}")

    imp_nodes_gcn = cam_gcn.compute_node_importance(
        data_gcn.x, data_gcn.edge_index, batch=data_gcn.batch, target_class=target_id
    )

    imp_nodes_gine = cam_gine.compute_node_importance(
        data_gine.x, data_gine.edge_index, edge_attr=data_gine.edge_attr, batch=data_gine.batch, target_class=target_id
    )
    names_gcn  = get_node_names(data_gcn, node_vocab_gcn)
    names_gine = get_node_names(data_gine, node_vocab_gine)

    top_nodes_gcn  = np.argsort(imp_nodes_gcn)[-topk:]
    top_nodes_gine = np.argsort(imp_nodes_gine)[-topk:]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    # Colonna 1: Immagine query
    show_image(
        image_id, image_dir,
        f"Query di Partenza: {target_name}\n(Sample Index: {idx})",
        ax=axes[0]
    )

    # Colonna 2: GCN (TITOLO CORRETTO)
    axes[1].barh(
        [names_gcn[i] for i in top_nodes_gcn],
        imp_nodes_gcn[top_nodes_gcn],
        color=cm.Oranges(imp_nodes_gcn[top_nodes_gcn] / (imp_nodes_gcn.max() + 1e-8))
    )
    axes[1].set_title(
        f"GCN — Nodi chiave per l'allineamento\nal prototipo di: {target_name}",
        fontsize=10, fontweight='bold', color='#D35400'
    )
    axes[1].set_xlabel("Importanza del Nodo (Grad-CAM su Distanza Metrica)")
    axes[1].grid(axis='x', linestyle='--', alpha=0.4)

    # Colonna 3: GINE (TITOLO CORRETTO)
    axes[2].barh(
        [names_gine[i] for i in top_nodes_gine],
        imp_nodes_gine[top_nodes_gine],
        color=cm.Greens(imp_nodes_gine[top_nodes_gine] / (imp_nodes_gine.max() + 1e-8))
    )
    axes[2].set_title(
        f"GINE — Nodi chiave per l'allineamento\nal prototipo di: {target_name}",
        fontsize=10, fontweight='bold', color='#1E8449'
    )
    axes[2].set_xlabel("Importanza del Nodo (Grad-CAM su Distanza Metrica)")
    axes[2].grid(axis='x', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Analisi globale degli archi — GINE (GNNExplainer)
# ---------------------------------------------------------------------------
def analyze_global_edge_importance_with_explainer(
    dataset_gine, model_gine, rel_vocab_gine, device, num_samples=50
):
    """
    Calcola l'importanza media degli archi aggregata per tipo di relazione
    utilizzando GNNExplainer sul modello GINE.
    """
    idx2rel    = {v: k for k, v in rel_vocab_gine.items()}
    rel_scores = defaultdict(list)

    wrapped_model = GINEWrapper(model_gine)
    explainer     = build_explainer(wrapped_model, epochs=100)

    print(f"Analisi archi (GINE — GNNExplainer) su {num_samples} campioni...")
    samples_to_test = min(num_samples, len(dataset_gine))

    for idx in range(samples_to_test):
        data        = dataset_gine[idx].to(device)
        explanation = explainer(data.x, data.edge_index, edge_attr=data.edge_attr)

        if not (hasattr(explanation, 'edge_mask') and explanation.edge_mask is not None):
            continue

        imp_edges = explanation.edge_mask.cpu().numpy()
        rels = (
            data.edge_attr.cpu().numpy()
            if data.edge_attr.dim() == 1
            else data.edge_attr[:, 0].cpu().numpy()
        )

        for r_id, importance in zip(rels, imp_edges):
            rel_name = idx2rel.get(int(r_id), f"Rel_{r_id}")
            rel_scores[rel_name].append(importance)

    global_stats = [
        {'Relazione': rel_name, 'Importanza Media': np.mean(scores), 'Conteggio': len(scores)}
        for rel_name, scores in rel_scores.items()
    ]

    df_edges = pd.DataFrame(global_stats)
    df_edges = df_edges.sort_values(by='Importanza Media', ascending=False).reset_index(drop=True)
    return df_edges


# ---------------------------------------------------------------------------
# Analisi globale degli archi — GCN (GNNExplainer)
# ---------------------------------------------------------------------------
def analyze_global_edge_importance_gcn(
    dataset_gcn, model_gcn, node_vocab_gcn, device, num_samples=50
):
    """
    Calcola l'importanza media degli archi per il modello GCN aggregandoli
    per coppia di tipi di nodo (Sorgente → Destinazione).
    """
    idx2node   = {v: k for k, v in node_vocab_gcn.items()}
    rel_scores = defaultdict(list)

    wrapped_model = GCNWrapper(model_gcn)
    explainer     = build_explainer(wrapped_model, epochs=100)

    print(f"Analisi archi (GCN — GNNExplainer) su {num_samples} campioni...")
    samples_to_test = min(num_samples, len(dataset_gcn))

    for idx in range(samples_to_test):
        data        = dataset_gcn[idx].to(device)
        explanation = explainer(data.x, data.edge_index)

        if not (hasattr(explanation, 'edge_mask') and explanation.edge_mask is not None):
            continue

        imp_edges  = explanation.edge_mask.cpu().numpy()
        src_nodes  = data.edge_index[0].cpu().numpy()
        dst_nodes  = data.edge_index[1].cpu().numpy()
        node_types = data.x[:, 0].long().cpu().numpy()

        for s, d, importance in zip(src_nodes, dst_nodes, imp_edges):
            type_src      = idx2node.get(int(node_types[s]), f"Node_{node_types[s]}")
            type_dst      = idx2node.get(int(node_types[d]), f"Node_{node_types[d]}")
            relation_pair = f"{type_src} → {type_dst}"
            rel_scores[relation_pair].append(importance)

    global_stats = [
        {'Relazione': pair_name, 'Importanza Media': np.mean(scores), 'Conteggio': len(scores)}
        for pair_name, scores in rel_scores.items()
    ]

    df_edges = pd.DataFrame(global_stats)
    df_edges = df_edges.sort_values(by='Importanza Media', ascending=False).reset_index(drop=True)
    return df_edges


# ---------------------------------------------------------------------------
# Plot studio globale archi
# ---------------------------------------------------------------------------
def plot_global_edge_study(df_edges, topk=15, model_name="GINE"):
    """
    Mostra un grafico a barre globale delle relazioni più importanti nello spazio latente.
    'model_name' viene usato nel titolo per distinguere GCN da GINE.
    """
    plt.figure(figsize=(12, 6))
    df_plot = df_edges.head(topk)

    sns.barplot(
        x='Importanza Media',
        y='Relazione',
        data=df_plot,
        palette='viridis',
        hue='Relazione',
        legend=False
    )

    plt.title(
        f"Studio Globale sugli Archi ({model_name}) — "
        f"Top {topk} Relazioni Semantiche nello Spazio Latente",
        fontsize=12, fontweight='bold'
    )
    plt.xlabel("Importanza Relativa Media (GNNExplainer — Edge Mask)")  # FIX: era "Grad-CAM"
    plt.ylabel("Tipo di Relazione (Arco)")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()