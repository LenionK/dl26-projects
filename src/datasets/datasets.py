import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import Sampler
from torch.utils.data import DataLoader

class SceneDataset(Dataset):
    def __init__(self, df, image_dir, label2idx, transform=None):

        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

        self.label2idx = label2idx
        self.idx2label = {i: l for l, i in label2idx.items()}

        self.labels = df['labels'].map(self.label2idx).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_id = str(row["image_id"])
        if not image_id.lower().endswith(".jpg"):
            image_id = image_id + ".jpg"

        img_path = os.path.join(self.image_dir, image_id)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]

        return img, label



def get_dataloader(dataset, n_classes=6, n_samples=4):

    sampler = BalancedBatchSampler(
        dataset.labels,
        n_classes=6,
        n_samples=4
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler
    )

    return loader



class BalancedBatchSampler(Sampler):
    """Genera batch finiti per epoch: n_classes x n_samples."""

    def __init__(self, labels, n_classes, n_samples):
        self.labels = np.array(labels)
        self.labels_set = np.unique(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}

        self.n_classes = int(n_classes)
        self.n_samples = int(n_samples)

        if len(self.labels_set) < self.n_classes:
            raise ValueError(f"Not enough classes: {len(self.labels_set)} < n_classes={self.n_classes}")

    def __iter__(self):
        # IMPORTANTE: yield finito, altrimenti l'epoch non termina
        for _ in range(len(self)):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for c in classes:
                idxs = np.random.choice(self.label_to_indices[c], self.n_samples, replace=True)
                indices.extend(idxs.tolist())
            yield indices

    def __len__(self):
        return len(self.labels) // (self.n_classes * self.n_samples)



