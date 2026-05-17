import torch
import numpy as np
import ast

from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import torch
import ast
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data


class GQAGraphDataset(Dataset):

    def __init__(self, df, label_col="labels",
                 node_vocab=None,
                 rel_vocab=None,
                 label2idx=None,
                 idx2label=None,
                 use_bbox=True,
                 normalize_bbox=True,
                 bidirectional_edges=True):

        self.df = df.reset_index(drop=True)

        self.label_col = label_col

    
        self.node_vocab = node_vocab
        self.rel_vocab = rel_vocab

        self.use_bbox = use_bbox
        self.normalize_bbox = normalize_bbox
        self.bidirectional_edges = bidirectional_edges

        # labels
        self.raw_labels = df[label_col].values

        self.label2idx = label2idx
        self.idx2label = idx2label

        self.labels = self.df[self.label_col].values

        # fallback (safe)
        if self.label2idx is None:
            self.label2idx = self.build_label_mapping()[0]

    # -------------------------
    # LABEL MAPPING (SAFE)
    # -------------------------
    def build_label_mapping(self):
        unique_labels = sorted(np.unique(self.raw_labels))

        label2idx = {l: i for i, l in enumerate(unique_labels)}
        idx2label = {i: l for l, i in label2idx.items()}

        return label2idx, idx2label

    # -------------------------
    def __len__(self):
        return len(self.df)

    # -------------------------
    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        objects = ast.literal_eval(row["objects"])

        label_raw = row[self.label_col]

        if isinstance(label_raw, (int, np.integer)):
            label = int(label_raw)
        else:
            label = self.label2idx.get(label_raw, 0)

        graph = self.build_graph(objects, label)

        return graph

    def _bbox_scale(self, objects):
        scale_x, scale_y = 1.0, 1.0
        for obj in objects.values():
            scale_x = max(scale_x, float(obj["x"] + obj["w"]))
            scale_y = max(scale_y, float(obj["y"] + obj["h"]))
        return scale_x, scale_y

    def _bbox_features(self, obj, scale):
        sx, sy = scale
        return [
            float(obj["x"]) / sx,
            float(obj["y"]) / sy,
            float(obj["w"]) / sx,
            float(obj["h"]) / sy,
        ]

    # -------------------------
    def build_graph(self, objects, label):

        node_ids = list(objects.keys())
        node_map = {nid: i for i, nid in enumerate(node_ids)}

        x, edge_index, edge_attr = [], [], []

        if len(node_ids) == 0:
            feat = [0]
            if self.use_bbox:
                feat.extend([0.0, 0.0, 0.0, 0.0])
            x = torch.tensor([feat], dtype=torch.float)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0,), dtype=torch.long)
            return Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(label, dtype=torch.long),
            )

        scale = self._bbox_scale(objects) if self.normalize_bbox else (1.0, 1.0)

        # -------------------
        # NODES
        # -------------------
        for nid in node_ids:

            obj = objects[nid]
            name = obj["name"]

            node_id = self.node_vocab.get(name, 0)

            feat = [node_id]

            if self.use_bbox:
                if self.normalize_bbox:
                    feat.extend(self._bbox_features(obj, scale))
                else:
                    feat.extend([
                        obj["x"],
                        obj["y"],
                        obj["w"],
                        obj["h"],
                    ])

            x.append(feat)

        x = torch.tensor(x, dtype=torch.float)

        # -------------------
        # EDGES
        # -------------------
        for src_id, obj in objects.items():

            src = node_map[src_id]

            for rel in obj.get("relations", []):

                dst_id = rel["object"]

                if dst_id not in node_map:
                    continue

                dst = node_map[dst_id]

                rel_id = self.rel_vocab.get(rel["name"], 0)

                edge_index.append([src, dst])
                edge_attr.append(rel_id)

                if self.bidirectional_edges:
                    edge_index.append([dst, src])
                    edge_attr.append(rel_id)

        # -------------------
        # SAFE EMPTY GRAPH
        # -------------------
        if len(edge_index) == 0:

            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0,), dtype=torch.long)

        else:

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)

        # -------------------
        # FINAL DATA
        # -------------------
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(label, dtype=torch.long)
        )

        return data






class BalancedGraphBatchSampler(Sampler):

    def __init__(self, labels, n_classes, n_samples):
        self.labels = np.array(labels)
        self.labels_set = np.unique(self.labels)

        self.label_to_indices = {
            label: np.where(self.labels == label)[0]
            for label in self.labels_set
        }

        self.n_classes = int(n_classes)
        self.n_samples = int(n_samples)

        if len(self.labels_set) < self.n_classes:
            raise ValueError("Not enough classes")

    def __iter__(self):

        for _ in range(len(self)):

            classes = np.random.choice(
                self.labels_set,
                self.n_classes,
                replace=False
            )

            indices = []

            for c in classes:
                idxs = np.random.choice(
                    self.label_to_indices[c],
                    self.n_samples,
                    replace=True
                )
                indices.extend(idxs.tolist())

            yield indices

    def __len__(self):
        return len(self.labels) // (self.n_classes * self.n_samples)

def get_graph_dataloader(dataset, n_classes=None, n_samples=4):

    if n_classes is None:
        n_classes = len(np.unique(dataset.labels))

    sampler = BalancedGraphBatchSampler(
        dataset.labels,
        n_classes=n_classes,
        n_samples=n_samples
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler
    )

    return loader


import ast

def build_node_vocab(df):
    object_names = set()

    for obj_str in df["objects"]:
        objects = ast.literal_eval(obj_str)

        for _, obj in objects.items():
            object_names.add(obj["name"])

    object_names = sorted(object_names)   

    node_vocab = {name: i for i, name in enumerate(object_names)}

    return node_vocab

def build_rel_vocab(df):
    relations = set()

    for obj_str in df["objects"]:
        objects = ast.literal_eval(obj_str)

        for _, obj in objects.items():
            for rel in obj.get("relations", []):
                relations.add(rel["name"])

    relations = sorted(relations)   

    rel_vocab = {rel: i for i, rel in enumerate(relations)}

    return rel_vocab