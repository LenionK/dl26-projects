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