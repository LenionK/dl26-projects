import sys
import os
import argparse

import yaml
import pandas as pd
import torch
from torchvision import transforms

from src.datasets.datasets import SceneDataset, get_dataloader
from src.models.baseline import EmbeddingNet
from src.training.train_model import train_model
from src.training.contrastiveloss import supervised_contrastive_loss


# ---------------------------------------------------------------------------
# BASE DIR
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def main(config_name):

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------
    config_path = os.path.join(
        BASE_DIR,
        "experiments",
        "configs",
        config_name
    )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # -----------------------------------------------------------------------
    # Transform
    # -----------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    df_train = pd.read_csv(
        os.path.join(BASE_DIR, cfg["data"]["train_csv"])
    )

    all_labels = df_train["labels"].unique()
    label2idx = {l: i for i, l in enumerate(all_labels)}

    dataset_tr = SceneDataset(
        df_train,
        os.path.join(BASE_DIR, cfg["data"]["images_dir"]),
        label2idx=label2idx,
        transform=transform,
    )

    loader_train = get_dataloader(dataset_tr)

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    model, loss_history = train_model(
        model=EmbeddingNet(
            dim=cfg["model"]["dim"],
            pretrained=cfg["model"]["pretrained"],
        ),
        loader=loader_train,
        loss_fn=lambda emb, labels: supervised_contrastive_loss(
            emb,
            labels,
            temperature=cfg["training"]["temperature"],
        ),
        device=device,
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["lr"],
        save_path=os.path.join(BASE_DIR, cfg["training"]["checkpoint"]),
    )

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    print("Training completato.")
    print(f"Loss finale: {loss_history[-1]:.4f}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline training script")

    parser.add_argument(
        "--config",
        type=str,
        default="baseline_config.yaml",
        help="Nome del file YAML in experiments/configs",
    )

    args = parser.parse_args()

    main(args.config)