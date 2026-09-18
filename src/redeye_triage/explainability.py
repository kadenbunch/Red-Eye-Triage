"""Explainability: Grad-CAM++, Integrated Gradients and nearest-neighbour case retrieval.

Applied to the refit (full training pool) checkpoint on held-out test images.
The retrieval index is built from TRAINING-partition images only, so no test
image can be returned as a reference case.
"""

import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402
from PIL import Image  # noqa: E402

from . import config as C  # noqa: E402
from .models import MODEL_DISPLAY_NAMES  # noqa: E402


# ------------------------------------------------------------------ #
# Layer resolution
# ------------------------------------------------------------------ #
def get_cam_target_layer(model, model_name):
    """Last convolutional block: the standard Grad-CAM target for each backbone."""
    if model_name == "resnet18":
        return model.layer4[-1]
    if model_name in ("mobilenet", "efficientnet"):
        return model.features[-1]
    raise ValueError(f"No CAM target layer defined for {model_name}")


def build_feature_extractor(model, model_name):
    """Image tensor -> global-average-pooled penultimate embedding."""
    if model_name == "resnet18":
        backbone = torch.nn.Sequential(*list(model.children())[:-1])
    elif model_name in ("mobilenet", "efficientnet"):
        backbone = torch.nn.Sequential(model.features, torch.nn.AdaptiveAvgPool2d(1))
    else:
        raise ValueError(f"No feature extractor defined for {model_name}")
    return backbone.to(C.get_device()).eval()


@torch.no_grad()
def embed_image(feature_extractor, img_tensor):
    return feature_extractor(img_tensor.to(C.get_device())).flatten().cpu().numpy().astype("float32")


# ------------------------------------------------------------------ #
# Reference-case retrieval (FAISS, exact L2)
# ------------------------------------------------------------------ #
def build_reference_index(feature_extractor, train_data_bgr, train_labels, transform):
    import faiss

    embeddings, kept_imgs, kept_labels = [], [], []
    for img_bgr, lab in zip(train_data_bgr, train_labels):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = transform(Image.fromarray(rgb)).unsqueeze(0)
        embeddings.append(embed_image(feature_extractor, t))
        kept_imgs.append(rgb)
        kept_labels.append(int(lab))
    emb = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    return index, emb, kept_imgs, kept_labels


def retrieve_neighbors(feature_extractor, index, query_tensor, kept_imgs, kept_labels, k=3):
    q = embed_image(feature_extractor, query_tensor).reshape(1, -1)
    dist, idx = index.search(q, k)
    # FAISS IndexFlatL2 returns SQUARED L2 distances.
    return [{"rank": r + 1, "squared_l2": float(dist[0][r]),
             "image": kept_imgs[idx[0][r]], "label": kept_labels[idx[0][r]]} for r in range(k)]


# ------------------------------------------------------------------ #
# Attributions
# ------------------------------------------------------------------ #
def compute_gradcam(model, model_name, input_tensor, target_class):
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    cam = GradCAMPlusPlus(model=model, target_layers=[get_cam_target_layer(model, model_name)])
    return cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(int(target_class))])[0]


def compute_integrated_gradients(model, input_tensor, target_class, n_steps=50):
    """Returns (normalized |attribution| map HxW, mean absolute convergence delta)."""
    from captum.attr import IntegratedGradients

    model.eval()
    inp = input_tensor.to(C.get_device())
    baseline = torch.zeros_like(inp)
    attr, delta = IntegratedGradients(model).attribute(
        inp, baselines=baseline, target=int(target_class), n_steps=n_steps,
        return_convergence_delta=True)
    a = np.mean(np.abs(attr.squeeze(0).cpu().detach().numpy()), axis=0)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return a, float(delta.abs().mean().item())


def denormalize_for_display(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * np.array(C.IMAGENET_STD) + np.array(C.IMAGENET_MEAN)
    return np.clip(img, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ #
# Figures
# ------------------------------------------------------------------ #
def explain_case(model, model_name, feature_extractor, index, kept_imgs, kept_labels, img_tensor,
                 classes=None, true_label=None, k=3, use_true_class=False, save_path=None,
                 ig_steps=50):
    """One panel: [input | Grad-CAM++ | Integrated Gradients | top-k training neighbours]."""
    from pytorch_grad_cam.utils.image import show_cam_on_image

    classes = classes or C.CLASSES
    model.eval()
    inp = img_tensor.unsqueeze(0).to(C.get_device())
    with torch.no_grad():
        probs = F.softmax(model(inp), dim=1).cpu().numpy().ravel()
    pred = int(probs.argmax())
    target = int(true_label) if (use_true_class and true_label is not None) else pred

    gradcam = compute_gradcam(model, model_name, inp, target)
    ig_map, ig_delta = compute_integrated_gradients(model, inp, target, n_steps=ig_steps)
    neighbors = retrieve_neighbors(feature_extractor, index, inp, kept_imgs, kept_labels, k=k)

    rgb = denormalize_for_display(img_tensor)
    cam_overlay = show_cam_on_image(rgb, cv2.resize(gradcam, (C.IMG_SIZE, C.IMG_SIZE)), use_rgb=True)

    ncols = 3 + k
    fig = plt.figure(figsize=(3.0 * ncols, 3.4))
    gs = gridspec.GridSpec(1, ncols, wspace=0.08)

    ax = fig.add_subplot(gs[0])
    ax.imshow(rgb)
    ax.axis("off")
    title = f"Input\nPred: {classes[pred]} ({probs[pred]:.2f})"
    if true_label is not None:
        title += f"\nTrue: {classes[int(true_label)]}"
    ax.set_title(title, fontsize=9)

    ax = fig.add_subplot(gs[1])
    ax.imshow(cam_overlay)
    ax.axis("off")
    ax.set_title(f"Grad-CAM++\n(target: {classes[target]})", fontsize=9)

    ax = fig.add_subplot(gs[2])
    im = ax.imshow(ig_map, cmap="inferno", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title(f"Integrated Gradients\n(Δ={ig_delta:.3f})", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))  # was rendering as 0,0,0,1

    for j, nb in enumerate(neighbors):
        ax = fig.add_subplot(gs[3 + j])
        ax.imshow(nb["image"])
        ax.axis("off")
        match = "match" if nb["label"] == pred else "mismatch"
        ax.set_title(f"Reference #{nb['rank']} ({match})\n{classes[nb['label']]}\n"
                     f"(squared L2={nb['squared_l2']:.1f})", fontsize=8)

    fig.suptitle(f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} case explanation", fontsize=11, y=1.02)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"pred": pred, "target_used": target, "probs": probs.tolist(),
            "ig_convergence_delta": ig_delta,
            "neighbor_labels": [nb["label"] for nb in neighbors],
            "neighbor_squared_l2": [nb["squared_l2"] for nb in neighbors]}


def gradcam_class_grid(model, model_name, dataset, classes=None, save_path=None, per_class=1):
    """Grad-CAM++ montage of the highest-confidence correctly classified example(s) per class."""
    from pytorch_grad_cam.utils.image import show_cam_on_image

    classes = classes or C.CLASSES
    model.eval()
    chosen = {c: [] for c in range(len(classes))}
    with torch.no_grad():
        for i in range(len(dataset)):
            img_t, lab = dataset[i]
            probs = F.softmax(model(img_t.unsqueeze(0).to(C.get_device())), dim=1).cpu().numpy().ravel()
            if probs.argmax() == lab:
                chosen[lab].append((probs[lab], i))
    for c in chosen:
        chosen[c] = [idx for _, idx in sorted(chosen[c], reverse=True)[:per_class]]

    rows = [(c, idx) for c in range(len(classes)) for idx in chosen[c]]
    fig, axes = plt.subplots(len(rows), 2, figsize=(6, 3 * len(rows)), squeeze=False)
    for r, (c, idx) in enumerate(rows):
        img_t, _ = dataset[idx]
        gc = compute_gradcam(model, model_name, img_t.unsqueeze(0).to(C.get_device()), c)
        rgb = denormalize_for_display(img_t)
        overlay = show_cam_on_image(rgb, cv2.resize(gc, (C.IMG_SIZE, C.IMG_SIZE)), use_rgb=True)
        axes[r, 0].imshow(rgb)
        axes[r, 0].axis("off")
        axes[r, 0].set_title("Original" if r == 0 else "", fontsize=9)
        axes[r, 0].text(-0.15, 0.5, classes[c], rotation=90, va="center", ha="center",
                        transform=axes[r, 0].transAxes, fontsize=10, fontweight="bold")
        axes[r, 1].imshow(overlay)
        axes[r, 1].axis("off")
        axes[r, 1].set_title("Grad-CAM++" if r == 0 else "", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return rows
