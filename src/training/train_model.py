import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
import os

def train_model(
    model,
    loader,
    loss_fn,
    device,
    epochs=10,
    lr=1e-4,
    save_path=None
):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    loss_history = []

    for epoch in range(1, epochs + 1):

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

    # ===== SAVE MODEL =====
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

    return model, loss_history