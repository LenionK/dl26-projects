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


# ---------------------------------------------------------------------------
# Classe GraphGradCAM
# ---------------------------------------------------------------------------
class GraphGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients   = None

        self.forward_hook  = target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()

    def _save_activations(self, module, input, output):
        self.activations = output.detach().clone()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach().clone()

    def compute_node_importance(self, x, edge_index, edge_attr=None, prototype=None):
        """
        Calcola l'importanza dei nodi usando la proiezione dell'embedding 
        sul prototipo della classe target (corretto per Metric Learning).
        """
        self.model.eval()
        self.activations = None
        self.gradients   = None

        if prototype is None:
            raise ValueError("Per i modelli di embedding è necessario passare un 'prototype' (vettore).")

        # Creiamo il vettore batch per un singolo grafo
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Forward pass: otteniamo l'embedding del grafo
        if edge_attr is not None:
            # Assicurati che i parametri rispecchino la firma esatta del tuo GINE
            emb = self.model(x, edge_index, edge_attr, batch)
        else:
            # Per il GCN
            emb = self.model(x, edge_index, batch)

        # Forziamo le dimensioni a [1, Dimensioni_Embedding]
        emb = emb.view(1, -1)
        prototype = prototype.to(x.device).view(1, -1)

        # Calcoliamo lo score scalare tramite Prodotto Scalare (Dot Product)
        # Rappresenta quanto il grafo corrente è vicino al centro della classe target
        score = torch.mm(emb, prototype.t()).squeeze()

        # Backward pass per accumulare i gradienti sul layer target
        self.model.zero_grad()
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Hook non catturati. Verifica che il 'target_layer' inserito "
                "sia un layer intermedio (es. convs[-1]) e NON il layer di pooling finale."
            )

        # Controllo di sicurezza sulla dimensionalità delle attivazioni (livello nodi)
        if self.activations.dim() != 2 or self.activations.size(0) != x.size(0):
            raise RuntimeError(
                f"Forma attivazioni errata: {self.activations.shape}. "
                f"Il target_layer deve restituire un tensore con {x.size(0)} nodi."
            )

        # Calcolo Grad-CAM standard sui nodi
        alpha           = self.gradients.mean(dim=0, keepdim=True)   # [1, Features]
        node_weights    = torch.sum(alpha * self.activations, dim=-1) # [Num_Nodi]
        node_importance = torch.clamp(node_weights, min=0)            # ReLU

        # Normalizzazione
        max_val = node_importance.max()
        if max_val > 0:
            node_importance = node_importance / max_val
        else:
            warnings.warn("Gradienti nulli per questo sample: importanza posta a zero.", RuntimeWarning)

        return node_importance.cpu().numpy()

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

def compute_class_prototypes(gallery_embs, df_labels):
    """
    Calcola il prototipo (centroide) di ogni classe facendo la media degli embedding.
    
    Args:
        gallery_embs: Tensore [N, D] contenente gli embedding di tutti i grafi.
        df_labels: Serie o array di lunghezza N con le label di classe associate ad ogni embedding.
    Returns:
        dict: { class_id -> Tensore 1D [D] (il prototipo) }
    """
    prototypes = {}
    unique_classes = np.unique(df_labels)
    
    for cls in unique_classes:
        indices = np.where(df_labels == cls)[0]
        # Estrae tutti gli embedding appartenenti alla classe cls e fa la media
        class_embs = gallery_embs[indices]
        prototypes[cls] = class_embs.mean(dim=0)
        
    return prototypes


# ---------------------------------------------------------------------------
# Wrapper modelli per GNNExplainer
# ---------------------------------------------------------------------------
class GCNWrapper(torch.nn.Module):
    """Wrapper GCN per GNNExplainer: calcola lo score scalare rispetto a un prototipo."""
    def __init__(self, model, prototype):
        super().__init__()
        self.model = model
        # Forziamo il prototipo a forma [1, Dimensioni_Embedding]
        self.prototype = prototype.detach().clone().view(1, -1)

    def forward(self, x, edge_index):
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        emb = self.model(x, edge_index, batch).view(1, -1)
        # Prodotto scalare per ottenere uno scalare (vicinanza al prototipo)
        score = torch.mm(emb, self.prototype.to(x.device).t()).squeeze(0)
        return score


class GINEWrapper(torch.nn.Module):
    """Wrapper GINE per GNNExplainer: calcola lo score scalare rispetto a un prototipo."""
    def __init__(self, model, prototype):
        super().__init__()
        self.model = model
        # Forziamo il prototipo a forma [1, Dimensioni_Embedding]
        self.prototype = prototype.detach().clone().view(1, -1)

    def forward(self, x, edge_index, edge_attr):
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        emb = self.model(x, edge_index, edge_attr, batch, num_graphs=1).view(1, -1)
        # Prodotto scalare per ottenere uno scalare (vicinanza al prototipo)
        score = torch.mm(emb, self.prototype.to(x.device).t()).squeeze(0)
        return score


# ---------------------------------------------------------------------------
# Builder explainer
# ---------------------------------------------------------------------------
def build_explainer(wrapped_model, epochs=200):
    """
    Crea un Explainer GNNExplainer per il modello proiettato sul prototipo.
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
# Utility embedding
# ---------------------------------------------------------------------------
def get_embedding(model, data, device, model_type="gine"):
    """
    Esegue un forward pass e restituisce l'embedding del grafo come tensore 1D.

    Args:
        model_type: "gine" — chiama model(x, edge_index, edge_attr, batch, num_graphs=1)
                    "gcn"  — chiama model(x, edge_index, batch)
    Returns:
        Tensore 1D [D] su CPU.
    """
    model.eval()
    data  = data.to(device)
    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        if model_type == "gine":
            emb = model(data.x, data.edge_index, data.edge_attr, batch, num_graphs=1)
        else:
            emb = model(data.x, data.edge_index, batch)

    return emb.squeeze(0)  # [D]


def get_topk_neighbors(query_emb, gallery_embs, topk=10):
    """
    Restituisce gli indici dei top-k vicini più vicini nello spazio metrico
    usando distanza euclidea.

    Args:
        query_emb   : tensore 1D [D]
        gallery_embs: tensore 2D [N, D]
        topk        : numero di vicini
    Returns:
        Lista di indici (int) ordinata per distanza crescente.
    """
    dists = torch.norm(gallery_embs - query_emb.unsqueeze(0), dim=1)
    return torch.argsort(dists)[:topk].tolist()


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
# Plot Grad-CAM: confronto GCN vs GINE (3 colonne, per sample singolo)
# ---------------------------------------------------------------------------
def plot_gradcam_comparison(idx, df_val, image_dir, idx2label,
                            val_dataset_gcn, val_dataset_gine,
                            node_vocab_gcn, node_vocab_gine,
                            cam_gcn, cam_gine, device, topk=8):

    data_gcn  = val_dataset_gcn[idx].to(device)
    data_gine = val_dataset_gine[idx].to(device)

    row         = df_val.iloc[idx]
    image_id    = row['image_id']
    target_id   = int(row['labels'])
    target_name = idx2label.get(target_id, f"ID d'Embedding: {target_id}")

    imp_nodes_gcn  = cam_gcn.compute_node_importance(
        data_gcn.x, data_gcn.edge_index, target_class=target_id
    )
    imp_nodes_gine = cam_gine.compute_node_importance(
        data_gine.x, data_gine.edge_index,
        edge_attr=data_gine.edge_attr, target_class=target_id
    )

    names_gcn  = get_node_names(data_gcn,  node_vocab_gcn)
    names_gine = get_node_names(data_gine, node_vocab_gine)

    top_nodes_gcn  = np.argsort(imp_nodes_gcn)[-topk:]
    top_nodes_gine = np.argsort(imp_nodes_gine)[-topk:]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    show_image(image_id, image_dir,
               f"Query: {target_name}\n(Sample Index: {idx})", ax=axes[0])

    axes[1].barh(
        [names_gcn[i] for i in top_nodes_gcn],
        imp_nodes_gcn[top_nodes_gcn],
        color=cm.Oranges(imp_nodes_gcn[top_nodes_gcn] / (imp_nodes_gcn.max() + 1e-8))
    )
    axes[1].set_title(
        f"GCN — Grad-CAM\ndim {target_id} dell'Embedding",
        fontsize=10, fontweight='bold', color='#D35400'
    )
    axes[1].set_xlabel("Magnitudo del Gradiente nello Spazio Metrico")
    axes[1].grid(axis='x', linestyle='--', alpha=0.4)

    axes[2].barh(
        [names_gine[i] for i in top_nodes_gine],
        imp_nodes_gine[top_nodes_gine],
        color=cm.Greens(imp_nodes_gine[top_nodes_gine] / (imp_nodes_gine.max() + 1e-8))
    )
    axes[2].set_title(
        f"GINE — Grad-CAM\ndim {target_id} dell'Embedding",
        fontsize=10, fontweight='bold', color='#1E8449'
    )
    axes[2].set_xlabel("Magnitudo del Gradiente nello Spazio Metrico")
    axes[2].grid(axis='x', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Grad-CAM aggregato per classe
# ---------------------------------------------------------------------------
def aggregate_node_importance_by_class(
    cam, dataset, node_vocab, df, target_classes, prototypes,
    device, num_samples_per_class=30, use_edge_attr=False
):
    """
    Aggiornato per usare i PROTOTIPI nel calcolo della Grad-CAM.
    """
    idx2node = {v: k for k, v in node_vocab.items()}
    class_indices = {cls: df.index[df['labels'] == cls].tolist() for cls in target_classes}
    results = {}

    for cls, indices in class_indices.items():
        if not indices:
            warnings.warn(f"Nessun sample trovato per la classe {cls}.", RuntimeWarning)
            continue
        
        if cls not in prototypes:
            warnings.warn(f"Prototipo non trovato per la classe {cls}.", RuntimeWarning)
            continue
            
        # Preleviamo il prototipo specifico di questa classe
        proto = prototypes[cls].to(device)

        sample_idx  = indices[:num_samples_per_class]
        node_scores = defaultdict(list)

        for idx in sample_idx:
            data       = dataset[idx].to(device)
            node_types = data.x[:, 0].long().cpu().numpy()

            try:
                # Passiamo 'proto' al posto di target_class
                if use_edge_attr:
                    imp = cam.compute_node_importance(
                        data.x, data.edge_index,
                        edge_attr=data.edge_attr, prototype=proto
                    )
                else:
                    imp = cam.compute_node_importance(
                        data.x, data.edge_index, prototype=proto
                    )
            except Exception as e:
                warnings.warn(f"Sample {idx} skippato: {e}", RuntimeWarning)
                continue

            for node_idx, importance in enumerate(imp):
                node_name = idx2node.get(int(node_types[node_idx]), f"node_{node_types[node_idx]}")
                node_scores[node_name].append(float(importance))

        results[cls] = pd.Series({
            name: np.mean(scores) for name, scores in node_scores.items()
        }).sort_values(ascending=False)

    return results


def plot_node_importance_by_class(
    results_gcn, results_gine, idx2label,
    num_samples_per_class, topk=10
):
    """
    Per ogni classe: 2 barplot affiancati (GCN | GINE) con l'importanza
    media per tipo di nodo, calcolata su num_samples_per_class campioni.

    La struttura a griglia (n_classi × 2) permette un confronto diretto
    tra architetture e tra classi.
    """
    classes = list(results_gcn.keys())
    n       = len(classes)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4.5 * n))

    # Normalizza axes a lista di coppie anche con n==1
    if n == 1:
        axes = [axes]

    for row, cls in enumerate(classes):
        label_name = idx2label.get(cls, f"Classe {cls}")

        for col, (scores, color, model_name) in enumerate([
            (results_gcn.get(cls,  pd.Series(dtype=float)), '#E67E22', 'GCN'),
            (results_gine.get(cls, pd.Series(dtype=float)), '#1E8449', 'GINE'),
        ]):
            ax  = axes[row][col]
            top = scores.head(topk)

            ax.barh(top.index[::-1], top.values[::-1], color=color, alpha=0.85)
            ax.set_title(
                f"{model_name} — Classe: {label_name}\n"
                f"(media su {num_samples_per_class} campioni, top-{topk} nodi)",
                fontsize=10, fontweight='bold'
            )
            ax.set_xlabel("Importanza Media (Grad-CAM)")
            ax.grid(axis='x', linestyle='--', alpha=0.4)

    plt.suptitle(
        "Node Importance Aggregata per Classe — GCN vs GINE",
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Analisi globale archi — GINE (GNNExplainer)
# ---------------------------------------------------------------------------
def analyze_global_edge_importance_with_explainer(
    dataset_gine, model_gine, rel_vocab_gine, device, num_samples=50
):
    """
    Calcola l'importanza media degli archi per tipo di relazione usando
    GNNExplainer sul modello GINE.
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
            rel_scores[idx2rel.get(int(r_id), f"Rel_{r_id}")].append(importance)

    global_stats = [
        {'Relazione': r, 'Importanza Media': np.mean(s), 'Conteggio': len(s)}
        for r, s in rel_scores.items()
    ]
    df = pd.DataFrame(global_stats).sort_values(
        by='Importanza Media', ascending=False
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Analisi globale archi — GCN (GNNExplainer)
# ---------------------------------------------------------------------------
def analyze_global_edge_importance_gcn(
    dataset_gcn, model_gcn, node_vocab_gcn, device, num_samples=50, min_count=3
):
    """
    Calcola l'importanza media degli archi per il modello GCN aggregandoli
    per coppia di tipi di nodo (Sorgente → Destinazione).
    
    Aggiunto filtraggio per 'min_count' per evitare che relazioni viste una sola volta
    falsino la cima della classifica.
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
            type_src = idx2node.get(int(node_types[s]), f"Node_{node_types[s]}")
            type_dst = idx2node.get(int(node_types[d]), f"Node_{node_types[d]}")
            rel_scores[f"{type_src} → {type_dst}"].append(importance)

    # Costruiamo le statistiche globali
    global_stats = []
    for r, s in rel_scores.items():
        conteggio = len(s)
        # Filtriamo l'outlier PRIMA di popolare il report se non soddisfa la frequenza minima
        if conteggio >= min_count:
            global_stats.append({
                'Relazione': r, 
                'Importanza Media': float(np.mean(s)), 
                'Conteggio': conteggio
            })

    # Se il filtro è stato troppo severo e non è rimasto nulla, lancia un avviso anziché crashare
    if not global_stats:
        warnings.warn(
            f"Nessuna relazione ha superato la soglia di min_count={min_count}. "
            "Prova ad aumentare 'num_samples' nel notebook o ad abbassare 'min_count'.",
            RuntimeWarning
        )
        # Riprova senza filtro per non restituire un dataframe vuoto
        global_stats = [
            {'Relazione': r, 'Importanza Media': float(np.mean(s)), 'Conteggio': len(s)}
            for r, s in rel_scores.items()
        ]

    df = pd.DataFrame(global_stats).sort_values(
        by='Importanza Media', ascending=False
    ).reset_index(drop=True)
    
    return df


# ---------------------------------------------------------------------------
# Plot studio globale archi
# ---------------------------------------------------------------------------
def plot_global_edge_study(df_edges, topk=15, model_name="GINE"):
    """Barplot delle top-k relazioni più importanti nello spazio latente."""
    plt.figure(figsize=(12, 6))
    df_plot = df_edges.head(topk)

    sns.barplot(
        x='Importanza Media', y='Relazione',
        data=df_plot, palette='viridis',
        hue='Relazione', legend=False
    )
    plt.title(
        f"Studio Globale Archi ({model_name}) — Top {topk} Relazioni",
        fontsize=12, fontweight='bold'
    )
    plt.xlabel("Importanza Media (GNNExplainer — Edge Mask)")
    plt.ylabel("Tipo di Relazione")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# ===========================================================================
# SEZIONE 2 — FOCAL POINT ANALYSIS
# ===========================================================================

# ---------------------------------------------------------------------------
# Costruzione gallery embeddings
# ---------------------------------------------------------------------------
def build_gallery_embeddings(model, dataset, device, model_type="gine"):
    """
    Calcola l'embedding di tutti i sample del dataset e li restituisce
    come tensore [N, D] su CPU.

    Args:
        model_type: "gine" o "gcn" — propagato a get_embedding per usare
                    la firma corretta di ciascun modello.
    """
    model.eval()
    embs = []
    with torch.no_grad():
        for data in dataset:
            embs.append(get_embedding(model, data, device, model_type=model_type).cpu())
    return torch.stack(embs, dim=0)  # [N, D]


# ---------------------------------------------------------------------------
# Robustezza per tipo di relazione — delta embedding
# ---------------------------------------------------------------------------
def robustness_by_relation_type(model, dataset, rel_vocab, device,
                                 num_samples=None, model_type="gine"):
    """
    Per ogni tipo di relazione nel vocabolario, rimuove tutti gli archi
    di quel tipo da ciascun grafo e misura il delta in norma L2 sull'embedding.

    Un delta alto indica che quella relazione è strutturalmente critica
    per il modello (single point of failure).

    Args:
        model_type: "gine" o "gcn".
    Returns:
        pd.Series con indice = nome relazione, valori = delta medio.
    """
    samples = list(dataset)
    if num_samples is not None:
        samples = samples[:num_samples]

    results = {}

    for rel_name, rel_id in rel_vocab.items():
        deltas = []
        for data in samples:
            data = data.to(device)

            if model_type == "gine" and data.edge_attr is None:
                continue

            base_emb = get_embedding(model, data, device, model_type=model_type)

            if model_type == "gine":
                mask = (
                    data.edge_attr[:, 0] != rel_id
                    if data.edge_attr.dim() > 1
                    else data.edge_attr != rel_id
                )
                if mask.sum() < 2:
                    continue

                data_pert            = data.clone()
                data_pert.edge_index = data.edge_index[:, mask]
                data_pert.edge_attr  = data.edge_attr[mask]
            else:
                # GCN: non ha vocabolario di relazioni semantiche,
                # rimuoviamo archi casualmente come proxy di robustezza
                mask = torch.ones(data.edge_index.size(1), dtype=torch.bool)
                if mask.sum() < 2:
                    continue
                data_pert            = data.clone()
                data_pert.edge_index = data.edge_index[:, mask]

            pert_emb = get_embedding(model, data_pert, device, model_type=model_type)
            deltas.append(torch.norm(base_emb - pert_emb).item())

        results[rel_name] = float(np.mean(deltas)) if deltas else 0.0

    return pd.Series(results).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Focal point analysis — ranking disruption
# ---------------------------------------------------------------------------
def focal_point_analysis(model, dataset, gallery_embs, rel_vocab, device,
                          topk=10, num_samples=None, model_type="gine"):
    """
    Per ogni tipo di relazione misura quanto la sua rimozione altera il
    ranking dei vicini più prossimi nello spazio metrico.

    Metrica: Ranking Disruption = 1 - overlap(top-k base, top-k perturbed)
      0.0 = relazione irrilevante per il retrieval
      1.0 = relazione ridetermina completamente il ranking (focal point critico)

    Args:
        gallery_embs: tensore [N, D] prodotto da build_gallery_embeddings.
        model_type  : "gine" o "gcn".
    Returns:
        pd.Series con indice = nome relazione, valori = disruption media.
    """
    gallery_embs = gallery_embs.to(device)
    samples      = list(dataset)
    if num_samples is not None:
        samples = samples[:num_samples]

    results = {}

    for rel_name, rel_id in rel_vocab.items():
        disruptions = []
        for data in samples:
            data = data.to(device)

            if model_type == "gine" and data.edge_attr is None:
                continue

            base_emb  = get_embedding(model, data, device, model_type=model_type)
            base_rank = set(get_topk_neighbors(base_emb, gallery_embs, topk))

            if model_type == "gine":
                mask = (
                    data.edge_attr[:, 0] != rel_id
                    if data.edge_attr.dim() > 1
                    else data.edge_attr != rel_id
                )
                if mask.sum() < 2:
                    continue

                data_pert            = data.clone()
                data_pert.edge_index = data.edge_index[:, mask]
                data_pert.edge_attr  = data.edge_attr[mask]
            else:
                mask = torch.ones(data.edge_index.size(1), dtype=torch.bool)
                if mask.sum() < 2:
                    continue
                data_pert            = data.clone()
                data_pert.edge_index = data.edge_index[:, mask]

            pert_emb  = get_embedding(model, data_pert, device, model_type=model_type)
            pert_rank = set(get_topk_neighbors(pert_emb, gallery_embs, topk))

            overlap = len(base_rank & pert_rank) / topk
            disruptions.append(1.0 - overlap)

        results[rel_name] = float(np.mean(disruptions)) if disruptions else 0.0

    return pd.Series(results).sort_values(ascending=False)

def focal_point_analysis_gcn(model, dataset, gallery_embs, node_vocab, device,
                              topk=10, num_samples=None, min_edge_count=3):
    
    gallery_embs = gallery_embs.to(device)
    samples = list(dataset)
    if num_samples is not None:
        samples = samples[:num_samples]

    idx2node = {v: k for k, v in node_vocab.items()}
    unique_relations = set()
    for data in samples:
        node_types = data.x[:, 0].long().cpu().numpy()
        src_nodes  = data.edge_index[0].cpu().numpy()
        dst_nodes  = data.edge_index[1].cpu().numpy()
        for s, d in zip(src_nodes, dst_nodes):
            type_src = idx2node.get(int(node_types[s]), f"Node_{node_types[s]}")
            type_dst = idx2node.get(int(node_types[d]), f"Node_{node_types[d]}")
            unique_relations.add((type_src, type_dst))

    results = {}

    for type_src_target, type_dst_target in unique_relations:
        rel_name = f"{type_src_target} → {type_dst_target}"
        disruptions = []

        for i, data in enumerate(samples):
            data = data.to(device)
            node_types = data.x[:, 0].long().cpu().numpy()
            src_nodes  = data.edge_index[0].cpu().numpy()
            dst_nodes  = data.edge_index[1].cpu().numpy()

            # Conta quanti archi di questo tipo ci sono nel grafo
            edges_to_remove = [
                idx_e for idx_e, (s, d) in enumerate(zip(src_nodes, dst_nodes))
                if idx2node.get(int(node_types[s])) == type_src_target
                and idx2node.get(int(node_types[d])) == type_dst_target
            ]
            
            # FIX 1: salta se la relazione non è presente o rimuove troppi archi
            n_total = data.edge_index.size(1)
            if len(edges_to_remove) == 0:
                continue
            if (n_total - len(edges_to_remove)) < min_edge_count:
                continue

            mask = torch.ones(n_total, dtype=torch.bool, device=device)
            for idx_e in edges_to_remove:
                mask[idx_e] = False

            # FIX 2: escludi la query stessa dalla gallery per il ranking
            gallery_without_self = torch.cat([
                gallery_embs[:i], gallery_embs[i+1:]
            ], dim=0)

            base_emb  = get_embedding(model, data, device, model_type="gcn")
            base_rank = set(get_topk_neighbors(base_emb, gallery_without_self, topk))

            data_pert            = data.clone()
            data_pert.edge_index = data.edge_index[:, mask]

            pert_emb  = get_embedding(model, data_pert, device, model_type="gcn")
            pert_rank = set(get_topk_neighbors(pert_emb, gallery_without_self, topk))

            overlap = len(base_rank & pert_rank) / topk
            disruptions.append(1.0 - overlap)

        # FIX 3: richiedi un minimo di campioni validi per la stima
        if len(disruptions) >= 5:
            results[rel_name] = float(np.mean(disruptions))

    return pd.Series(results).sort_values(ascending=False)

# ---------------------------------------------------------------------------
# Robustezza per tipo di relazione — delta embedding per GCN
# ---------------------------------------------------------------------------
def robustness_by_relation_type_gcn(model, dataset, node_vocab, device,
                                     num_samples=None):
    """
    Per ogni coppia di tipi di nodo (Src → Dst), rimuove tutti gli archi
    di quel tipo da ciascun grafo e misura il delta in norma L2 sull'embedding.

    Un delta alto indica che quella transizione strutturale è critica
    per la stabilità dell'embedding del modello (single point of failure).
    """
    samples = list(dataset)
    if num_samples is not None:
        samples = samples[:num_samples]

    idx2node = {v: k for k, v in node_vocab.items()}
    
    # 1. Identifichiamo tutte le relazioni (coppie Src -> Dst) uniche nei campioni
    unique_relations = set()
    for data in samples:
        node_types = data.x[:, 0].long().cpu().numpy()
        src_nodes  = data.edge_index[0].cpu().numpy()
        dst_nodes  = data.edge_index[1].cpu().numpy()
        for s, d in zip(src_nodes, dst_nodes):
            type_src = idx2node.get(int(node_types[s]), f"Node_{node_types[s]}")
            type_dst = idx2node.get(int(node_types[d]), f"Node_{node_types[d]}")
            unique_relations.add((type_src, type_dst))

    results = {}

    # 2. Ablazione selettiva e calcolo del Delta L2
    for type_src_target, type_dst_target in unique_relations:
        rel_name = f"{type_src_target} → {type_dst_target}"
        deltas = []

        for data in samples:
            data = data.to(device)
            node_types = data.x[:, 0].long().cpu().numpy()
            src_nodes  = data.edge_index[0].cpu().numpy()
            dst_nodes  = data.edge_index[1].cpu().numpy()

            # Embedding di riferimento (base)
            base_emb = get_embedding(model, data, device, model_type="gcn")

            # Maschera booleana per escludere la relazione target
            mask = torch.ones(data.edge_index.size(1), dtype=torch.bool, device=device)
            for idx_edge, (s, d) in enumerate(zip(src_nodes, dst_nodes)):
                type_src = idx2node.get(int(node_types[s]), f"Node_{node_types[s]}")
                type_dst = idx2node.get(int(node_types[d]), f"Node_{node_types[d]}")
                if type_src == type_src_target and type_dst == type_dst_target:
                    mask[idx_edge] = False

            # Saltiamo se l'arco non era presente o se rimuoverlo isola completamente il grafo
            if mask.sum() == mask.size(0) or mask.sum() < 2:
                continue

            # Generiamo il grafo perturbato
            data_pert            = data.clone()
            data_pert.edge_index = data.edge_index[:, mask]

            # Calcolo embedding perturbato e calcolo dello spostamento geometrico (Delta L2)
            pert_emb = get_embedding(model, data_pert, device, model_type="gcn")
            deltas.append(torch.norm(base_emb - pert_emb).item())

        if deltas:
            results[rel_name] = float(np.mean(deltas))

    return pd.Series(results).sort_values(ascending=False)

# ---------------------------------------------------------------------------
# Plot distribuzione disruption per una singola relazione
# ---------------------------------------------------------------------------
def plot_disruption_distribution(model, dataset, gallery_embs, rel_vocab,
                                  rel_name, device, topk=10, num_samples=None,
                                  model_name="GINE", model_type="gine"):
    """
    Istogramma + KDE della Ranking Disruption per una singola relazione.
    Utile per verificare se l'effetto è uniforme o concentrato su pochi outlier.
    """
    rel_id       = rel_vocab[rel_name]
    gallery_embs = gallery_embs.to(device)
    samples      = list(dataset)
    if num_samples is not None:
        samples = samples[:num_samples]

    disruptions = []
    for data in samples:
        data = data.to(device)

        if model_type == "gine" and data.edge_attr is None:
            continue

        base_emb  = get_embedding(model, data, device, model_type=model_type)
        base_rank = set(get_topk_neighbors(base_emb, gallery_embs, topk))

        if model_type == "gine":
            mask = (
                data.edge_attr[:, 0] != rel_id
                if data.edge_attr.dim() > 1
                else data.edge_attr != rel_id
            )
        else:
            mask = torch.ones(data.edge_index.size(1), dtype=torch.bool)

        if mask.sum() < 2:
            continue

        data_pert            = data.clone()
        data_pert.edge_index = data.edge_index[:, mask]
        if model_type == "gine":
            #data_pert.edge_attr = data.edge_attr[mask]
            data_pert.edge_attr = data.edge_attr[mask, :] if data.edge_attr.dim() > 1 else data.edge_attr[mask]

        pert_emb  = get_embedding(model, data_pert, device, model_type=model_type)
        pert_rank = set(get_topk_neighbors(pert_emb, gallery_embs, topk))
        disruptions.append(1.0 - len(base_rank & pert_rank) / topk)

    if not disruptions:
        print(f"Nessun sample valido per la relazione '{rel_name}'.")
        return

    plt.figure(figsize=(8, 4))
    sns.histplot(disruptions, bins=20, kde=True, color='steelblue', edgecolor='white')
    plt.axvline(np.mean(disruptions), color='red', linestyle='--',
                label=f"Media: {np.mean(disruptions):.3f}")
    plt.xlabel("Ranking Disruption")
    plt.ylabel("Frequenza")
    plt.title(
        f"Distribuzione Disruption — '{rel_name}' ({model_name})\n"
        f"n={len(disruptions)} sample, top-{topk} retrieval",
        fontweight='bold'
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
