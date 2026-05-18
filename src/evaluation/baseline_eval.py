import sys
import os
import argparse

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import yaml
import pandas as pd
import torch
from torchvision import transforms

from src.datasets.datasets import SceneDataset
from src.models.baseline import EmbeddingNet
from src.evaluation.retrieval import (
    compute_embeddings,
    evaluate_retrieval,
    print_metrics,
)


def main(config_name):

    # ---------------------------------------------------------------------------
    # Config
    # ---------------------------------------------------------------------------
    config_path = os.path.join(
        BASE_DIR,
        "experiments",
        "configs",
        config_name
    )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ---------------------------------------------------------------------------
    # Transform
    # ---------------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    # ---------------------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------------------
    df_train = pd.read_csv(
        os.path.join(BASE_DIR, cfg["data"]["train_csv"])
    )
    df_val = pd.read_csv(
        os.path.join(BASE_DIR, cfg["data"]["val_csv"])
    )

    all_labels = df_train["labels"].unique()
    label2idx = {l: i for i, l in enumerate(all_labels)}

    dataset_val = SceneDataset(
        df_val,
        os.path.join(BASE_DIR, cfg["data"]["images_dir"]),
        label2idx=label2idx,
        transform=transform,
    )

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = os.path.join(BASE_DIR, cfg["training"]["checkpoint"])

    model = EmbeddingNet(
        dim=cfg["model"]["dim"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Checkpoint caricato da: {ckpt_path}")

    # ---------------------------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------------------------
    embeddings, labels = compute_embeddings(
        model=model,
        dataset=dataset_val,
        device=device,
        batch_size=cfg["dataloader"]["batch_size"],
        num_workers=cfg["dataloader"]["num_workers"],
        normalize=True,
    )

    print("Embeddings shape:", embeddings.shape)
    print("Labels shape:    ", labels.shape)

    # ---------------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------------
    results, indices, scores = evaluate_retrieval(
        embeddings,
        labels,
        ks=(1, 5, 10),
    )

    print_metrics(results)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval evaluation script")

    parser.add_argument(
        "--config",
        type=str,
        default="baseline_config.yaml",
        help="Nome del file YAML in experiments/configs",
    )

    args = parser.parse_args()

    main(args.config)