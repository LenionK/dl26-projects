from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.gcn import GraphBatch


@dataclass(frozen=True)
class GraphSample:
    x: torch.Tensor  # (N,F)
    adj: torch.Tensor  # (N,N)
    label: int
    image_id: str


def _build_vocab(items: Iterable[str]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for s in items:
        if s not in vocab:
            vocab[s] = len(vocab) + 1  # reserve 0 for unknown/pad
    return vocab


class GQAGraphSceneDataset(Dataset):
    """
    Minimal GQA scene-graph dataset wrapper.

    Expected dataframe-like input: any object with __len__ and iloc, with columns:
    - image_id: str/int
    - labels: scene label (string or int)

    Scene graph files: one JSON per image, located at:
        graphs_dir/{image_id}.json

    Each graph JSON should contain:
    - objects: {obj_id: {name: str, bbox: [x,y,w,h] or [x1,y1,x2,y2], ...}, ...}
    - relations: [{subject: obj_id, object: obj_id, name: str}, ...]  (name optional)
    This matches the common GQA sceneGraphs structure.
    """

    def __init__(
        self,
        df,
        graphs_dir: str | os.PathLike,
        *,
        obj_name_vocab: dict[str, int] | None = None,
        rel_name_vocab: dict[str, int] | None = None,
        max_nodes: int | None = None,
    ):
        self.df = df.reset_index(drop=True) if hasattr(df, "reset_index") else df
        self.graphs_dir = str(graphs_dir)

        labels_col = df["labels"] if hasattr(df, "__getitem__") else [row["labels"] for row in df]
        uniq = list(dict.fromkeys(list(labels_col)))
        self.label2idx = {l: i for i, l in enumerate(uniq)}
        self.idx2label = {i: l for l, i in self.label2idx.items()}
        self.labels = np.array([self.label2idx[l] for l in labels_col], dtype=np.int64)

        # Build (or accept) vocabularies for object / relation names.
        if obj_name_vocab is None or rel_name_vocab is None:
            obj_names: list[str] = []
            rel_names: list[str] = []
            for i in range(len(self)):
                image_id = str(self._get_row(i)["image_id"])
                g = self._load_graph(image_id)
                objects = g.get("objects", {})
                for _, o in objects.items():
                    if "name" in o:
                        obj_names.append(str(o["name"]))
                for r in g.get("relations", []):
                    if "name" in r:
                        rel_names.append(str(r["name"]))
            obj_name_vocab = obj_name_vocab or _build_vocab(obj_names)
            rel_name_vocab = rel_name_vocab or _build_vocab(rel_names)

        self.obj_name_vocab = obj_name_vocab
        self.rel_name_vocab = rel_name_vocab
        self.max_nodes = int(max_nodes) if max_nodes is not None else None

    def __len__(self) -> int:
        return len(self.df)

    def _get_row(self, idx: int) -> Any:
        if hasattr(self.df, "iloc"):
            return self.df.iloc[idx]
        return self.df[idx]

    def _load_graph(self, image_id: str) -> dict[str, Any]:
        import json

        if not image_id.lower().endswith(".json"):
            fname = f"{image_id}.json"
        else:
            fname = image_id
        path = os.path.join(self.graphs_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __getitem__(self, idx: int) -> GraphSample:
        row = self._get_row(idx)
        image_id = str(row["image_id"])
        label = int(self.labels[idx])

        g = self._load_graph(image_id)
        objects = g.get("objects", {})
        relations = g.get("relations", [])

        # Sort object ids deterministically to create a stable node ordering.
        obj_ids = sorted(objects.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        if self.max_nodes is not None:
            obj_ids = obj_ids[: self.max_nodes]

        id2node = {oid: i for i, oid in enumerate(obj_ids)}
        n = len(obj_ids)

        # Node features: [obj_name_id, bbox(4)] (normalized later in preprocessing if desired)
        x = torch.zeros((n, 5), dtype=torch.float32)
        for oid, i in id2node.items():
            o = objects[oid]
            name = str(o.get("name", ""))
            name_id = float(self.obj_name_vocab.get(name, 0))
            bbox = o.get("bbox", [0, 0, 0, 0])
            bbox = list(bbox)[:4] + [0, 0, 0, 0]
            x[i, 0] = name_id
            x[i, 1:5] = torch.tensor(bbox[:4], dtype=torch.float32)

        # Adjacency from relations (undirected for simplicity).
        adj = torch.zeros((n, n), dtype=torch.float32)
        for r in relations:
            s = r.get("subject")
            o = r.get("object")
            if s in id2node and o in id2node and s != o:
                si, oi = id2node[s], id2node[o]
                adj[si, oi] = 1.0
                adj[oi, si] = 1.0

        return GraphSample(x=x, adj=adj, label=label, image_id=image_id)


def collate_graph_samples(samples: list[GraphSample]) -> tuple[GraphBatch, torch.Tensor]:
    """
    Pads nodes to max N in batch.
    Returns (GraphBatch, labels)
    """
    if len(samples) == 0:
        raise ValueError("Empty batch")

    b = len(samples)
    n_max = max(s.x.shape[0] for s in samples)
    f = samples[0].x.shape[1]

    x = torch.zeros((b, n_max, f), dtype=torch.float32)
    adj = torch.zeros((b, n_max, n_max), dtype=torch.float32)
    mask = torch.zeros((b, n_max), dtype=torch.float32)
    y = torch.tensor([s.label for s in samples], dtype=torch.long)

    for i, s in enumerate(samples):
        n = s.x.shape[0]
        x[i, :n] = s.x
        adj[i, :n, :n] = s.adj
        mask[i, :n] = 1.0

    return GraphBatch(x=x, adj=adj, mask=mask), y

