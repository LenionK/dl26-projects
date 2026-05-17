import os

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


def load_model_weights(checkpoint_path, model, device=None):
    """Load model weights from a state_dict file or a training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)


def _load_checkpoint(checkpoint_path, model, optimizer=None, device=None):
    """Load weights or a full training checkpoint. Returns (start_epoch, loss_history)."""
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        loss_history = list(checkpoint.get("loss_history", []))
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return start_epoch, loss_history

    model.load_state_dict(checkpoint)
    return 1, []


def _save_checkpoint(save_path, model, optimizer, epoch, loss_history):
    ckpt_dir = os.path.dirname(save_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": loss_history,
        },
        save_path,
    )


def train_model(
    model,
    loader,
    loss_fn,
    device,
    epochs=10,
    lr=1e-4,
    save_path=None,
    checkpoint_path=None,
    resume_epoch=None,
):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    start_epoch = 1
    loss_history = []

    if checkpoint_path is not None:
        start_epoch, loss_history = _load_checkpoint(
            checkpoint_path, model, optimizer, device
        )
        if resume_epoch is not None:
            start_epoch = resume_epoch
        print(f"Resumed from {checkpoint_path} (epoch {start_epoch}/{epochs})")

    for epoch in range(start_epoch, epochs + 1):
        model.train()

        total_loss = 0.0
        n_batches = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            emb = model(images)
            loss = loss_fn(emb, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}")

    if save_path is not None:
        _save_checkpoint(save_path, model, optimizer, epoch, loss_history)
        print(f"Checkpoint saved to: {save_path}")

    return model, loss_history


def train_graph_model(
    model,
    loader,
    loss_fn,
    device,
    epochs=10,
    lr=1e-4,
    weight_decay=1e-4,
    max_grad_norm=1.0,
    use_scheduler=True,
    save_path=None,
    checkpoint_path=None,
    resume_epoch=None,
):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    start_epoch = 1
    loss_history = []

    if checkpoint_path is not None:
        start_epoch, loss_history = _load_checkpoint(
            checkpoint_path, model, optimizer, device
        )
        if resume_epoch is not None:
            start_epoch = resume_epoch
        print(f"Resumed from {checkpoint_path} (epoch {start_epoch}/{epochs})")

    scheduler = None
    if use_scheduler:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - start_epoch + 1),
            eta_min=lr * 0.05,
        )

    for epoch in range(start_epoch, epochs + 1):
        model.train()

        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)

            emb = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                batch.num_graphs,
            )

            loss = loss_fn(emb, batch.y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / max(1, n_batches)
        loss_history.append(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - lr: {current_lr:.2e}")

    if save_path is not None:
        _save_checkpoint(save_path, model, optimizer, epoch, loss_history)
        print(f"Checkpoint saved to: {save_path}")

    return model, loss_history
