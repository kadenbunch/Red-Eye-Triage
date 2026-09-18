"""Data loading, preprocessing/augmentation and the fixed train/test split.

CLAIM checklist items 8-10 (data), 12 and 20 (preprocessing, augmentation).
"""

import os
import warnings

import cv2
import numpy as np
import pandas as pd

from . import config as C


# ------------------------------------------------------------------ #
# Transforms
# ------------------------------------------------------------------ #
def get_transforms():
    """Return (train_transform, eval_transform).

    Training augmentation replaces SMOTE oversampling. Transforms are kept
    clinically plausible: horizontal flips are acceptable (the two eyes are
    approximately mirror images); vertical flips and large rotations are
    avoided because they create anatomically implausible presentations.
    """
    from torchvision import transforms

    train_transform = transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])
    return train_transform, eval_transform


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #
try:
    from torch.utils.data import Dataset as _DatasetBase
except ImportError:  # allows the duplicate audit to run without PyTorch installed
    _DatasetBase = object


class EyeDiseaseDataset(_DatasetBase):
    """Holds decoded BGR uint8 arrays; converts to RGB PIL images for torchvision."""

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from PIL import Image
        from torchvision import transforms

        img = cv2.cvtColor(self.data[idx], cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.transform(img) if self.transform else transforms.ToTensor()(img)
        return img.float(), int(self.labels[idx])


def list_class_files(folder):
    """Sorted file names in a class folder, ignoring hidden files (e.g. .DS_Store)."""
    return [f for f in sorted(os.listdir(folder)) if not f.startswith(".")]


def load_dataset(dataset_path=None, classes=None):
    """Load every image once into memory.

    Images are read in a deterministic order (classes in CLASSES order, files
    sorted by name), so the saved split indices remain valid across runs.

    Returns
    -------
    data : np.ndarray (N, 224, 224, 3) uint8, BGR
    labels : np.ndarray (N,) int
    manifest : pd.DataFrame with columns [index, path, file, class, label]
    """
    dataset_path = dataset_path or C.DATASET_PATH
    classes = classes or C.CLASSES
    data, labels, manifest = [], [], []
    for cls_id, cls_name in enumerate(classes):
        folder = os.path.join(dataset_path, cls_name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Expected class folder not found: {folder}. See data/README.md.")
        files = list_class_files(folder)
        if C.MAX_IMAGES_PER_CLASS is not None:
            files = files[:C.MAX_IMAGES_PER_CLASS]
        for img_name in files:
            path = os.path.join(folder, img_name)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                warnings.warn(f"Unreadable image skipped: {path}")
                continue
            img = cv2.resize(img, (C.IMG_SIZE, C.IMG_SIZE))
            manifest.append({"index": len(data), "path": os.path.relpath(path, dataset_path),
                             "file": img_name, "class": cls_name, "label": cls_id})
            data.append(img)
            labels.append(cls_id)
    return np.array(data), np.array(labels), pd.DataFrame(manifest)


# ------------------------------------------------------------------ #
# Held-out split (created once, then reused)
# ------------------------------------------------------------------ #
def make_or_load_split(labels, manifest=None, ckpt_dir=None):
    """Create the stratified 85/15 train/test split once and persist it.

    The split is stored as integer indices (split_indices.npz) and, when a
    manifest is supplied, as a human-readable CSV (split_manifest.csv) listing
    every image file and its partition.
    """
    from sklearn.model_selection import train_test_split

    ckpt_dir = ckpt_dir or C.CKPT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    split_path = os.path.join(ckpt_dir, "split_indices.npz")
    if os.path.exists(split_path):
        s = np.load(split_path)
        idx_train, idx_test = s["idx_train"], s["idx_test"]
        print(f"Reusing existing split: {split_path}")
    else:
        idx_train, idx_test = train_test_split(
            np.arange(len(labels)), test_size=C.TEST_SIZE, stratify=labels,
            random_state=C.SEED)
        np.savez(split_path, idx_train=idx_train, idx_test=idx_test)
        print(f"Created and saved new split: {split_path}")

    if len(idx_train) + len(idx_test) != len(labels):
        raise RuntimeError(
            "Saved split does not match the number of loaded images. The image "
            "folders have changed since the split was created.")

    if manifest is not None:
        m = manifest.copy()
        m["partition"] = "train"
        m.loc[m["index"].isin(idx_test), "partition"] = "test"
        m.to_csv(os.path.join(ckpt_dir, "split_manifest.csv"), index=False)
    return idx_train, idx_test
