# retrieval.py

import faiss
import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import average_precision_score


# =========================================================
# EMBEDDING EXTRACTION
# =========================================================

@torch.no_grad()
def compute_embeddings(
    model,
    dataset,
    device,
    batch_size=64,
    num_workers=4,
    normalize=True,
):
    """
    Compute embeddings for a dataset.

    Returns:
        emb : torch.Tensor [N, D]
        lbl : torch.Tensor [N]
    """

    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_emb = []
    all_lbl = []

    for images, labels in loader:

        images = images.to(device, non_blocking=True)

        use_cuda  =  torch.cuda.is_available()

        with torch.cuda.amp.autocast(enabled= use_cuda ):
            emb = model(images)

        emb = emb.detach()

        if normalize:
            emb = F.normalize(emb, dim=1)

        all_emb.append(emb.cpu())
        all_lbl.append(labels.cpu())

    emb = torch.cat(all_emb, dim=0)
    lbl = torch.cat(all_lbl, dim=0)

    return emb, lbl


@torch.no_grad()
def compute_graph_embeddings(
    model,
    dataset,
    device,
    batch_size=64,
    num_workers=0,
    normalize=True,
):
    """Compute graph-level embeddings for a PyG dataset."""
    from torch_geometric.loader import DataLoader

    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    all_emb = []
    all_lbl = []

    for batch in loader:
        batch = batch.to(device)
        emb = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.num_graphs,
        )

        if normalize:
            emb = F.normalize(emb, dim=1)

        all_emb.append(emb.cpu())
        all_lbl.append(batch.y.cpu())

    return torch.cat(all_emb, dim=0), torch.cat(all_lbl, dim=0)


# =========================================================
# FAISS INDEX
# =========================================================

def build_faiss_index(embeddings):
    """
    Build FAISS cosine similarity index.
    Embeddings must already be normalized.
    """

    x = embeddings.cpu().numpy().astype("float32")

    dim = x.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(x)

    return index


# =========================================================
# SEARCH
# =========================================================

def faiss_search(index, embeddings, k=5):
    """
    Search top-k nearest neighbors.

    Returns:
        scores : np.ndarray [N, k+1]
        indices : np.ndarray [N, k+1]
    """

    x = embeddings.cpu().numpy().astype("float32")

    scores, indices = index.search(x, k + 1)

    return scores, indices


# =========================================================
# RETRIEVAL @1
# =========================================================

def retrieval_at_1(indices, labels):
    """
    Compute Retrieval@1 accuracy + macro precision/recall.
    """

    y_true = labels.cpu().numpy()

    # remove self-match
    nn_idx = indices[:, 1]

    y_pred = y_true[nn_idx]

    acc = (y_true == y_pred).mean()

    prec = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    rec = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "acc@1": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
    }


# =========================================================
# RECALL@K
# =========================================================

def recall_at_k(indices, labels, k=5):
    """
    Recall@K

    A query is correct if at least one
    retrieved sample in top-k has same label.
    """

    y = labels.cpu().numpy()

    retrieved = indices[:, 1:k+1]

    retrieved_labels = y[retrieved]

    hits = (retrieved_labels == y[:, None]).any(axis=1)

    return float(hits.mean())


# =========================================================
# PRECISION@K
# =========================================================

def precision_at_k(indices, labels, k=5):
    """
    Precision@K

    Fraction of retrieved samples in top-k
    that belong to the correct class.
    """

    y = labels.cpu().numpy()

    retrieved = indices[:, 1:k+1]

    retrieved_labels = y[retrieved]

    correct = (retrieved_labels == y[:, None])

    precision = correct.mean(axis=1)

    return float(precision.mean())


# =========================================================
# MRR
# =========================================================

def mean_reciprocal_rank(indices, labels):
    """
    Mean Reciprocal Rank (MRR)
    """

    y = labels.cpu().numpy()

    retrieved = indices[:, 1:]

    retrieved_labels = y[retrieved]

    reciprocal_ranks = []

    for i in range(len(y)):

        matches = np.where(retrieved_labels[i] == y[i])[0]

        if len(matches) == 0:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / (matches[0] + 1))

    return float(np.mean(reciprocal_ranks))


# =========================================================
# MAP
# =========================================================

def mean_average_precision(indices, labels):
    """
    Mean Average Precision (mAP)

    Computes AP for each query and averages.
    """

    y = labels.cpu().numpy()

    retrieved = indices[:, 1:]

    retrieved_labels = y[retrieved]

    APs = []

    for i in range(len(y)):

        relevant = (retrieved_labels[i] == y[i]).astype(np.int32)

        if relevant.sum() == 0:
            APs.append(0.0)
            continue

        scores = np.arange(len(relevant), 0, -1)

        ap = average_precision_score(relevant, scores)

        APs.append(ap)

    return float(np.mean(APs))


# =========================================================
# FULL EVALUATION
# =========================================================

def evaluate_retrieval(
    embeddings,
    labels,
    ks=(1, 5, 10),
):
    """
    Full retrieval evaluation.
    """

    index = build_faiss_index(embeddings)

    max_k = max(ks)

    scores, indices = faiss_search(
        index,
        embeddings,
        k=max_k,
    )

    results = {}

    # Retrieval@1
    r1 = retrieval_at_1(indices, labels)
    results.update(r1)

    # Recall@K
    for k in ks:
        results[f"recall@{k}"] = recall_at_k(
            indices,
            labels,
            k=k,
        )

    # Precision@K
    for k in ks:
        results[f"precision@{k}"] = precision_at_k(
            indices,
            labels,
            k=k,
        )

    # Ranking metrics
    results["mrr"] = mean_reciprocal_rank(
        indices,
        labels,
    )

    results["map"] = mean_average_precision(
        indices,
        labels,
    )

    return results, indices, scores


# =========================================================
# PRETTY PRINT
# =========================================================

def print_metrics(results):

    print("\n========== Retrieval Metrics ==========\n")

    for k, v in results.items():
        print(f"{k:20s}: {v:.4f}")

    print()



# =========================================================
# QUALITATIVE RETRIEVAL VISUALIZATION
# =========================================================

import os

import matplotlib.pyplot as plt
import torch
from PIL import Image


def image_path_from_id(image_dir, image_id):
    """Build image path from GQA image_id (numeric or string)."""
    image_id = str(image_id)
    if not image_id.lower().endswith(".jpg"):
        image_id = f"{image_id}.jpg"
    return os.path.join(image_dir, image_id)


def load_retrieval_image(df, image_dir, sample_idx):
    path = image_path_from_id(image_dir, df.iloc[sample_idx]["image_id"])
    return np.array(Image.open(path).convert("RGB"))


def unnormalize(img):
    """
    Undo ImageNet normalization.
    """

    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    return (img * std + mean).clamp(0, 1)


def plot_retrieval(
    dataset,
    labels,
    indices,
    scores,
    query_idx,
    idx2label=None,
    topk=5,
    title=None,
):
    """
    Visualize retrieval results.

    Parameters
    ----------
    dataset : Dataset
        PyTorch dataset

    labels : torch.Tensor
        Labels tensor [N]

    indices : np.ndarray
        FAISS retrieved indices [N, k+1]

    scores : np.ndarray
        FAISS similarity scores [N, k+1]

    query_idx : int
        Query sample index

    idx2label : dict
        Optional class-name mapping

    topk : int
        Number of retrieved samples to display
    """

    # -----------------------------------------------------
    # QUERY
    # -----------------------------------------------------

    img_q, _ = dataset[query_idx]

    img_q = (
        unnormalize(img_q)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    q_label_idx = int(labels[query_idx])

    q_label = (
        idx2label[q_label_idx]
        if idx2label
        else q_label_idx
    )

    # -----------------------------------------------------
    # RETRIEVED
    # remove self-match
    # -----------------------------------------------------

    retrieved_idx = indices[query_idx][1 : topk + 1]
    retrieved_scores = scores[query_idx][1 : topk + 1]

    # -----------------------------------------------------
    # PLOT
    # -----------------------------------------------------

    ncols = topk + 1

    plt.figure(figsize=(3 * ncols, 3))

    # -----------------------------------------------------
    # QUERY IMAGE
    # -----------------------------------------------------

    plt.subplot(1, ncols, 1)

    plt.imshow(img_q)

    plt.axis("off")

    plt.title(f"QUERY\n{q_label}")

    # -----------------------------------------------------
    # RETRIEVED IMAGES
    # -----------------------------------------------------

    for j, (ridx, score) in enumerate(
        zip(retrieved_idx, retrieved_scores),
        start=2,
    ):

        img_r, _ = dataset[ridx]

        img_r = (
            unnormalize(img_r)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        label_idx = int(labels[ridx])

        label = (
            idx2label[label_idx]
            if idx2label
            else label_idx
        )

        correct = label_idx == q_label_idx

        color = "green" if correct else "red"

        plt.subplot(1, ncols, j)

        plt.imshow(img_r)

        plt.axis("off")

        plt.title(
            f"#{j-1} {label}\n{score:.3f}",
            color=color,
        )

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    if title:
        plt.suptitle(title)

    plt.tight_layout()

    plt.show()


def plot_graph_retrieval(
    df,
    image_dir,
    labels,
    indices,
    scores,
    query_idx,
    idx2label=None,
    topk=5,
    title=None,
):
    """
    Visualize graph retrieval results using scene images from image_id.

    df : pandas.DataFrame or GQAGraphDataset (uses .df)
        Must contain an ``image_id`` column aligned with embedding indices.
    """
    if hasattr(df, "df"):
        df = df.df

    q_label_idx = int(labels[query_idx])
    q_label = idx2label.get(q_label_idx, q_label_idx) if idx2label else q_label_idx

    retrieved_idx = indices[query_idx][1 : topk + 1]
    retrieved_scores = scores[query_idx][1 : topk + 1]

    ncols = topk + 1
    plt.figure(figsize=(3 * ncols, 3))

    plt.subplot(1, ncols, 1)
    plt.imshow(load_retrieval_image(df, image_dir, query_idx))
    plt.axis("off")
    plt.title(f"QUERY\n{q_label}", fontsize=9)

    for j, (ridx, score) in enumerate(zip(retrieved_idx, retrieved_scores), start=2):
        label_idx = int(labels[ridx])
        label = idx2label.get(label_idx, label_idx) if idx2label else label_idx
        correct = label_idx == q_label_idx
        color = "green" if correct else "red"

        plt.subplot(1, ncols, j)
        plt.imshow(load_retrieval_image(df, image_dir, ridx))
        plt.axis("off")
        plt.title(f"#{j - 1} {label}\n{score:.3f}", color=color, fontsize=9)

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def plot_tsne(embeddings, labels, title="t-SNE Embeddings"):
    emb = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings
    lbl = labels.cpu().numpy() if torch.is_tensor(labels) else labels

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(emb)

    plt.figure(figsize=(6,6))
    scatter = plt.scatter(emb_2d[:,0], emb_2d[:,1], c=lbl, cmap="tab10", s=10)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(title)
    plt.show()