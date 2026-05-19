import sys
import os
import argparse
import random

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import yaml
import numpy as np
import pandas as pd
import torch

from src.datasets.gqa_graph_dataset import (
    GQAGraphDataset,
    get_graph_dataloader,
    build_node_vocab,
    build_rel_vocab,
)

from src.models.gnn import GINEEmbeddingNet

from src.training.contrastiveloss import (
    supervised_contrastive_loss_GNN,
)

from src.training.train_model import (
    train_graph_model,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(config_name):

    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------
    config_path = os.path.join(
        BASE_DIR,
        "experiments",
        "conf",
        config_name,
    )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # -------------------------------------------------------------------------
    # Seed
    # -------------------------------------------------------------------------
    set_seed(cfg["seed"])

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    df_train = pd.read_csv(
        os.path.join(BASE_DIR, cfg["data"]["train_csv"])
    )

    df_val = pd.read_csv(
        os.path.join(BASE_DIR, cfg["data"]["val_csv"])
    )

    all_labels = df_train["labels"].unique()

    label2idx = {
        l: i for i, l in enumerate(all_labels)
    }

    idx2label = {
        i: l for l, i in label2idx.items()
    }

    df_train["labels"] = df_train["labels"].map(label2idx)
    df_val["labels"] = df_val["labels"].map(label2idx)

    # -------------------------------------------------------------------------
    # Vocabulary
    # -------------------------------------------------------------------------
    node_vocab = build_node_vocab(df_train)

    rel_vocab = build_rel_vocab(df_train)

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    train_dataset = GQAGraphDataset(
        df=df_train,
        label_col="labels",
        node_vocab=node_vocab,
        rel_vocab=rel_vocab,
        label2idx=label2idx,
        idx2label=idx2label,
        use_bbox=cfg["model"]["use_bbox"],
    )

    # -------------------------------------------------------------------------
    # Dataloader
    # -------------------------------------------------------------------------
    NUM_CLASSES = len(label2idx)

    train_loader = get_graph_dataloader(
        train_dataset,
        n_classes=NUM_CLASSES,
        n_samples=cfg["dataloader"]["n_samples"],
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = GINEEmbeddingNet(
        num_node_types=len(node_vocab) + 1,
        edge_vocab_size=len(rel_vocab),
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        use_bbox=cfg["model"]["use_bbox"],
        node_emb_dim=cfg["model"]["node_emb_dim"],
    )

    # -------------------------------------------------------------------------
    # Resume checkpoint
    # -------------------------------------------------------------------------
    resume_from = cfg["training"]["resume_from"]

    resume_from = (
        os.path.join(BASE_DIR, resume_from)
        if resume_from
        else None
    )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    model, history = train_graph_model(
        model=model,
        loader=train_loader,
        loss_fn=lambda emb, labels: supervised_contrastive_loss_GNN(
            emb,
            labels,
            temperature=cfg["training"]["temperature"],
        ),
        device=device,
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        checkpoint_path=resume_from,
        resume_epoch=cfg["training"]["resume_epoch"],
        save_path=os.path.join(
            BASE_DIR,
            cfg["training"]["checkpoint"],
        ),
    )

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------
    print("Training completato.")
    print(f"Loss finale: {history[-1]:.4f}")


# -------------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="GINE training script"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="gine_config.yaml",
        help="Nome del file YAML in experiments/conf",
    )

    args = parser.parse_args()

    main(args.config)