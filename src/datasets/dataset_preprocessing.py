"""
Pipeline di trasformazione: dataset raw (scene graph GQA) -> CSV etichettato.

Passaggi:
1. Caricamento degli scene graph train/val (JSON)
2. Campionamento riproducibile di una porzione del dataset
3. Costruzione dei DataFrame
4. Estrazione della lista di oggetti per ogni immagine
5. Assegnazione di un'etichetta di scena/stanza tramite CLIP (OpenAI)
6. Salvataggio dei DataFrame risultanti in CSV
"""

import json
import os
import random
from pathlib import Path

import pandas as pd
import torch
import clip
from PIL import Image

# Cartella del progetto, calcolata in modo relativo alla posizione di questo
# file (src/datasets/raw_to_csv.py -> root del progetto due livelli sopra).
# In questo modo i path funzionano indipendentemente da dove viene lanciato
# lo script (cwd), evitando i problemi dei path relativi del notebook originale.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def load_scene_graphs(train_path: str, val_path: str):
    """Carica gli scene graph train e val dai file JSON."""
    with open(train_path) as f:
        data_train = json.load(f)
    with open(val_path) as f:
        data_val = json.load(f)
    return data_train, data_val


def sample_data(data: dict, part: float = 0.3, seed: int = 42) -> dict:
    """Estrae un sottoinsieme riproducibile del dizionario dati."""
    random.seed(seed)
    keys = list(data.keys())
    k = max(1, int(len(keys) * part))
    sampled_keys = random.sample(keys, k)
    return {key: data[key] for key in sampled_keys}


def build_dataframe(data: dict) -> pd.DataFrame:
    """Converte il dizionario scene graph in DataFrame con colonna image_id."""
    df = pd.DataFrame.from_dict(data, orient="index")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "image_id"}, inplace=True)
    return df


def extract_objects(df: pd.DataFrame, col_input: str, col_output: str = "objects") -> pd.DataFrame:
    """Estrae i nomi degli oggetti da uno scene graph GQA, senza duplicati."""

    def get_objects(scene_graph):
        if not isinstance(scene_graph, dict):
            return []

        objects = []
        seen = set()

        for obj_id, obj_data in scene_graph.items():
            if isinstance(obj_data, dict) and "name" in obj_data:
                name = obj_data["name"]
                if name not in seen:
                    objects.append(name)
                    seen.add(name)

        return objects

    df[col_output] = df[col_input].apply(get_objects)
    return df


def assign_room_clip(df: pd.DataFrame, image_id_col: str, images_dir: str) -> pd.DataFrame:
    """Assegna a ogni immagine l'etichetta di scena più probabile tramite CLIP."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    labels = [
        "a kitchen interior",
        "a bedroom interior",
        "a bathroom interior",
        "a living room interior",
        "a street scene in a city",
        "a railway scene",
        "an airplane flying in the sky",
        "a person playing a sport",
        "a wild animal in nature",
        "a beach scene",
        "a countryside landscape",
        "a food scene",
        "a building exterior",
        "a restaurant dining scene",
    ]
    text = clip.tokenize(labels).to(device)

    results = []
    for _, row in df.iterrows():
        img_path = os.path.join(images_dir, row[image_id_col] + ".jpg")
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            logits = image_features @ text_features.T
            probs = logits.softmax(dim=-1)
            pred = labels[probs.argmax().item()]

        results.append(pred)

    df["labels"] = results
    return df


def save_df_csv(df: pd.DataFrame, name: str, output_dir: str = "../data") -> pd.DataFrame:
    """Salva le colonne rilevanti del DataFrame in un file CSV."""
    df_clean = df[["image_id", "objects", "labels", "obj_list"]]
    out_path = os.path.join(output_dir, f"{name}.csv")
    df_clean.to_csv(out_path, index=False)
    return df_clean


def process_split(
    data_raw: dict,
    split_name: str,
    images_dir: str,
    output_dir: str,
    part: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Esegue l'intera pipeline raw -> CSV per uno split (train o val)."""
    sampled = sample_data(data_raw, part=part, seed=seed)
    df = build_dataframe(sampled)
    df = extract_objects(df, "objects", "obj_list")
    df = assign_room_clip(df, "image_id", images_dir)
    df_clean = save_df_csv(df, f"df_{split_name}", output_dir=output_dir)
    return df_clean


def main():
    train_json = DATA_DIR / "sceneGraphs" / "train_sceneGraphs.json"
    val_json = DATA_DIR / "sceneGraphs" / "val_sceneGraphs.json"
    images_dir = DATA_DIR / "images"
    output_dir = DATA_DIR

    data_train, data_val = load_scene_graphs(str(train_json), str(val_json))

    process_split(data_train, "train", str(images_dir), str(output_dir))
    process_split(data_val, "val", str(images_dir), str(output_dir))


if __name__ == "__main__":
    main()