import sys
import os
import argparse
import random
import yaml
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.datasets.gqa_graph_dataset import (
    GQAGraphDataset,
    get_graph_dataloader,
    build_node_vocab,
    build_rel_vocab,
)

from src.models.gnn import GCNEmbeddingNet

from src.training.contrastiveloss import (
    supervised_contrastive_loss_GNN,
    ProxyTripletLoss
)

from src.training.train_model import train_graph_model_gcn


# -------------------------
# ARGPARSE
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GCN model with YAML config"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path config YAML"
    )

    parser.add_argument(
        "--loss",
        type=str,
        default="supcon",
        choices=["supcon", "proxy"],
        help="Loss function: supcon or proxy"
    )

    return parser.parse_args()


# -------------------------
# SEED
# -------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# MAIN
# -------------------------
def main():

    args = parse_args()

    # -------------------------
    # CONFIG
    # -------------------------
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(BASE_DIR, config_path)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"Config: {config_path}")
    print(f"Loss: {args.loss}")

    set_seed(cfg["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # -------------------------
    # DATA
    # -------------------------
    train_csv = os.path.join(BASE_DIR, cfg["data"]["train_csv"])
    val_csv = os.path.join(BASE_DIR, cfg["data"]["val_csv"])

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    all_labels = sorted(df_train["labels"].unique())

    label2idx = {
        l: i for i, l in enumerate(all_labels)
    }

    idx2label = {
        i: l for l, i in label2idx.items()
    }

    df_train["labels"] = df_train["labels"].map(label2idx)
    df_val["labels"] = df_val["labels"].map(label2idx)

    node_vocab = build_node_vocab(df_train)
    rel_vocab = build_rel_vocab(df_train)

    train_dataset = GQAGraphDataset(
        df=df_train,
        label_col="labels",
        node_vocab=node_vocab,
        rel_vocab=rel_vocab,
        label2idx=label2idx,
        idx2label=idx2label,
        use_bbox=cfg["model"]["use_bbox"],
    )

    num_classes = len(label2idx)

    train_loader = get_graph_dataloader(
        train_dataset,
        n_classes=num_classes,
        n_samples=cfg["dataloader"]["n_samples"],
    )

    # -------------------------
    # MODEL
    # -------------------------
    model = GCNEmbeddingNet(
        num_node_types=len(node_vocab) + 1,
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        use_bbox=cfg["model"]["use_bbox"],
        node_emb_dim=cfg["model"]["node_emb_dim"],
    )

    model = model.to(device)

    # -------------------------
    # LOSS SELECTION
    # -------------------------
    emb_dim = cfg["model"]["emb_dim"]

    if args.loss == "supcon":

        loss_fn = lambda emb, labels: supervised_contrastive_loss_GNN(
            emb,
            labels,
            temperature=cfg["training"]["temperature"],
        )

        proxy_params = None

    elif args.loss == "proxy":

        proxy_loss = ProxyTripletLoss(
            num_classes=num_classes,
            embedding_dim=emb_dim,
            margin=0.2
        ).to(device)

        loss_fn = lambda emb, labels: proxy_loss(emb, labels)

        proxy_params = list(proxy_loss.parameters())

    # -------------------------
    # OPTIMIZER (IMPORTANT)
    # -------------------------
    optimizer_params = list(model.parameters())

    if proxy_params is not None:
        optimizer_params += proxy_params

    optimizer = torch.optim.Adam(
        optimizer_params,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    # -------------------------
    # CHECKPOINT
    # -------------------------
    resume_from = cfg["training"].get("resume_from")
    if resume_from:
        resume_from = os.path.join(BASE_DIR, resume_from)

    save_path = os.path.join(
        BASE_DIR,
        cfg["training"]["checkpoint"]
    )

    # -------------------------
    # TRAINING
    # -------------------------
    model, history = train_graph_model_gcn(
        model=model,
        loader=train_loader,
        loss_fn=loss_fn,
        device=device,
        node_vocab=node_vocab,
        rel_vocab=rel_vocab,
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        checkpoint_path=resume_from,
        resume_epoch=cfg["training"]["resume_epoch"],
        save_path=save_path,
    )

    print("\nTraining completato.")
    print(f"Loss finale: {history[-1]:.4f}")
    print(f"Checkpoint salvato in: {save_path}")


if __name__ == "__main__":
    main()