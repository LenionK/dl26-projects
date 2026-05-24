import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# Transform (definiti qui così sono sempre disponibili all'import)
# ---------------------------------------------------------------------------
transform_model = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

transform_vis = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ---------------------------------------------------------------------------
# GradCAM++
# ---------------------------------------------------------------------------
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None
        self.gradients   = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x):
        self.model.zero_grad()

        embedding = self.model(x)
        score     = embedding.norm(dim=1).sum()
        score.backward()

        A  = self.activations
        dY = self.gradients

        dY2   = dY ** 2
        dY3   = dY ** 3
        sum_A = A.sum(dim=(2, 3), keepdim=True)

        denom = 2 * dY2 + sum_A * dY3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = dY2 / denom

        weights = (alpha * F.relu(dY)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.flatten(1).min(dim=1).values.view(-1, 1, 1, 1)
        cam = cam / (cam.flatten(1).max(dim=1).values.view(-1, 1, 1, 1) + 1e-8)

        return cam.squeeze(1).cpu().numpy()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def overlay_heatmap(image_np, cam, alpha=0.5):
    heatmap = cm.jet(cam)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = Image.fromarray(heatmap).resize(
        (image_np.shape[1], image_np.shape[0]), Image.BILINEAR
    )
    heatmap = np.array(heatmap) / 255.0
    return np.clip(alpha * heatmap + (1 - alpha) * image_np, 0, 1)


def show_gradcam(image_paths, model, target_layer, titles=None, alpha=0.5, cols=4):
    """
    Visualizza GradCAM++ per una lista di immagini.

    Args:
        image_paths  : lista di path alle immagini
        model        : EmbeddingNet già caricato e in eval()
        target_layer : layer su cui applicare GradCAM++ (es. model.encoder[-2])
        titles       : lista di titoli opzionali
        alpha        : intensità della heatmap (0-1)
        cols         : numero di colonne nella griglia
    """
    device = next(model.parameters()).device
    gradcam = GradCAMPlusPlus(model, target_layer)

    n    = len(image_paths)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2,
                             figsize=(cols * 5, rows * 3))
    axes = axes.flatten()

    for i, img_path in enumerate(image_paths):
        img_pil = Image.open(img_path).convert("RGB")

        x      = transform_model(img_pil).unsqueeze(0).to(device)
        x.requires_grad_(True)
        vis_np = transform_vis(img_pil).permute(1, 2, 0).numpy()

        cam     = gradcam(x)[0]
        overlay = overlay_heatmap(vis_np, cam, alpha)

        title = titles[i] if titles else img_path

        axes[i * 2].imshow(vis_np)
        axes[i * 2].set_title(f"{title}\noriginale", fontsize=8)
        axes[i * 2].axis("off")

        axes[i * 2 + 1].imshow(overlay)
        axes[i * 2 + 1].set_title(f"{title}\nGradCAM++", fontsize=8)
        axes[i * 2 + 1].axis("off")

    for j in range(i * 2 + 2, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()