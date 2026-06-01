import torch

def supervised_contrastive_loss(features, labels, temperature=0.1):
    device = features.device
    labels = labels.to(device)

    # (B,B) mask dei positivi (stessa classe)
    mask = labels[:, None].eq(labels[None, :]).float()

    logits = (features @ features.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # escludi diagonale
    self_mask = torch.eye(features.size(0), device=device)
    exp_logits = torch.exp(logits) * (1.0 - self_mask)

    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)

    pos_mask = mask * (1.0 - self_mask)
    denom = pos_mask.sum(dim=1).clamp_min(1.0)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / denom

    return -mean_log_prob_pos.mean()


import torch
import torch.nn.functional as F

def supervised_contrastive_loss_GNN(features, labels, temperature=0.1):

    device = features.device
    labels = labels.to(device)

    # safety check (IMPORTANTISSIMO DEBUG)
    assert features.shape[0] == labels.shape[0], \
        f"Mismatch: {features.shape} vs {labels.shape}"

    # normalize embeddings
    features = F.normalize(features, dim=1)

    # similarity matrix
    logits = torch.matmul(features, features.T) / temperature

    # numerical stability
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # labels mask
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float()

    # remove self-comparisons
    self_mask = torch.eye(features.size(0), device=device)
    mask = mask * (1 - self_mask)

    # log-softmax (CORRECT VERSION)
    exp_logits = torch.exp(logits) * (1 - self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)

    # positives
    denom = mask.sum(dim=1)

    # avoid empty classes in batch
    denom = denom.masked_fill(denom == 0, 1.0)

    loss = -(mask * log_prob).sum(dim=1) / denom

    return loss.mean()


import torch
import torch.nn as nn
import torch.nn.functional as F


class ProxyTripletLoss(nn.Module):
    """
    Proxy Triplet Loss

    Ogni classe ha un proxy apprendibile.
    Gli embedding vengono avvicinati al proxy corretto
    e allontanati dal proxy negativo più vicino.
    """

    def __init__(
        self,
        num_classes,
        embedding_dim,
        margin=0.2,
        normalize=True,
    ):
        super().__init__()

        self.margin = margin
        self.normalize = normalize

        self.proxies = nn.Parameter(
            torch.randn(num_classes, embedding_dim)
        )

        nn.init.kaiming_normal_(self.proxies)

    def forward(self, embeddings, labels):

        if self.normalize:
            embeddings = F.normalize(embeddings, dim=1)
            proxies = F.normalize(self.proxies, dim=1)
        else:
            proxies = self.proxies

        # distanza embedding <-> proxy
        dist_matrix = torch.cdist(
            embeddings,
            proxies,
            p=2
        )

        batch_idx = torch.arange(
            embeddings.size(0),
            device=embeddings.device
        )

        # proxy positivo
        pos_dist = dist_matrix[
            batch_idx,
            labels
        ]

        # maschera per escludere la classe corretta
        neg_mask = torch.ones_like(dist_matrix)
        neg_mask[batch_idx, labels] = 0

        neg_dist = dist_matrix.masked_fill(
            neg_mask == 0,
            float("inf")
        )

        # hardest negative proxy
        hardest_neg_dist, _ = neg_dist.min(dim=1)

        loss = F.relu(
            pos_dist - hardest_neg_dist + self.margin
        )

        return loss.mean()
