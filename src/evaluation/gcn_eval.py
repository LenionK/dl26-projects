import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import yaml
import pandas as pd
import torch

from src.datasets.gqa_graph_dataset import GQAGraphDataset
from src.models.gnn import GCNEmbeddingNet
from src.evaluation.retrieval import gcn_compute_graph_embeddings, evaluate_retrieval, print_metrics


def main():
    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------
    config_path = os.path.join(BASE_DIR, "experiments", "configs", "gcn_config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Carica checkpoint e vocabolari
    # -------------------------------------------------------------------------
    ckpt_path  = os.path.join(BASE_DIR, cfg["training"]["checkpoint"])
    checkpoint = torch.load(ckpt_path, map_location=device)

    node_vocab = checkpoint["node_vocab"]
    rel_vocab  = checkpoint["rel_vocab"]

    print(f"Checkpoint caricato da: {ckpt_path}")
    print(f"node_vocab size: {len(node_vocab)} | rel_vocab size: {len(rel_vocab)}")

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    df_train = pd.read_csv(os.path.join(BASE_DIR, cfg["data"]["train_csv"]))
    df_val   = pd.read_csv(os.path.join(BASE_DIR, cfg["data"]["val_csv"]))

    all_labels = df_train["labels"].unique()
    label2idx  = {l: i for i, l in enumerate(all_labels)}
    idx2label  = {i: l for l, i in label2idx.items()}

    df_train["labels"] = df_train["labels"].map(label2idx)
    df_val["labels"]   = df_val["labels"].map(label2idx)

    val_dataset = GQAGraphDataset(
        df=df_val,
        label_col="labels",
        node_vocab=node_vocab,
        rel_vocab=rel_vocab,
        label2idx=label2idx,
        idx2label=idx2label,
        use_bbox=cfg["model"]["use_bbox"],
    )

    # -------------------------------------------------------------------------
    # Modello
    # -------------------------------------------------------------------------
    gcn_model = GCNEmbeddingNet(
        num_node_types=len(node_vocab) + 1,
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        use_bbox=cfg["model"]["use_bbox"],
        node_emb_dim=cfg["model"]["node_emb_dim"],
    ).to(device)

    gcn_model.load_state_dict(checkpoint["model_state_dict"])
    gcn_model.eval()

    # -------------------------------------------------------------------------
    # Retrieval Evaluation
    # -------------------------------------------------------------------------
    embeddings, labels = gcn_compute_graph_embeddings(
        model=gcn_model,
        dataset=val_dataset,
        device=device,
        batch_size=cfg["dataloader"]["batch_size"],
        num_workers=cfg["dataloader"]["num_workers"],
        normalize=True,
    )

    print("Embeddings shape:", embeddings.shape)
    print("Labels shape:    ", labels.shape)

    results, indices, scores = evaluate_retrieval(
        embeddings,
        labels,
        ks=(1, 5, 10),
    )

    print_metrics(results)


if __name__ == "__main__":
    main()