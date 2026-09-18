"""Model factory (CLAIM checklist item 25: model description and initial weights)."""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from . import config as C

MODEL_DISPLAY_NAMES = {
    "efficientnet": "EfficientNet-B0",
    "mobilenet": "MobileNet-V2",
    "resnet18": "ResNet-18",
}


def get_model(name, num_classes=len(C.CLASSES), pretrained=True):
    """Build an ImageNet-pretrained backbone with a new `num_classes` output layer.

    Returns (model, config_dict). `pretrained=False` is only used by the unit
    tests so they can run without downloading weights.
    """
    if name == "mobilenet":
        w = models.MobileNet_V2_Weights.IMAGENET1K_V1
        model = models.mobilenet_v2(weights=w if pretrained else None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif name == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=w if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=w if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")

    config = {
        "model": name,
        "architecture": MODEL_DISPLAY_NAMES[name],
        "pretrained_weights": f"{w.__class__.__name__}.{w.name}" if pretrained else "none",
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "input_resolution": f"{C.IMG_SIZE}x{C.IMG_SIZE}",
        "output_classes": num_classes,
    }
    return model, config


def compute_class_weights(labels, num_classes, device=None):
    """Inverse-frequency class weights for the cross-entropy loss (CLAIM item 20)."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float, device=device or C.get_device())


def train_model(model, train_loader, class_weights, epochs=None, lr=None, device=None):
    """Fine-tune all layers with Adam and class-weighted cross-entropy."""
    epochs = epochs or C.EPOCHS
    lr = lr or C.LR
    device = device or C.get_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
        history.append({"epoch": epoch + 1, "loss": run_loss / len(train_loader),
                        "acc": correct / total})
    return model, history


@torch.no_grad()
def predict_probs(model, loader, device=None):
    """Return (y_true, y_prob) where y_prob is the N x C softmax matrix."""
    device = device or C.get_device()
    model.eval()
    y_true, y_prob = [], []
    for imgs, labels in loader:
        probs = torch.softmax(model(imgs.to(device)), dim=1).cpu().numpy()
        y_true.extend(labels.numpy())
        y_prob.extend(probs)
    return np.array(y_true), np.array(y_prob)


def load_checkpoint(model_name, ckpt_path, device=None):
    device = device or C.get_device()
    model, _ = get_model(model_name, len(C.CLASSES), pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()
