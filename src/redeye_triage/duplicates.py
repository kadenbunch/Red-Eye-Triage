"""Exact and near-duplicate image detection, and train/test leakage check.

Uses perceptual hashing (pHash) so detection is independent of file names and
re-encoding. Near-duplicate pairs that straddle the train/test split would
inflate held-out performance and are reported explicitly.

Note: hashing cannot detect *different* photographs of the same patient's eye;
patient identifiers are not available in the source dataset (see manuscript
Limitations).
"""

import os
from itertools import combinations

import pandas as pd
from PIL import Image

from . import config as C
from .data import list_class_files

DUP_HAMMING_THRESHOLD = 5  # pHash Hamming distance <= 5 treated as near-duplicate


def hash_images(dataset_path=None, classes=None):
    import imagehash

    dataset_path = dataset_path or C.DATASET_PATH
    classes = classes or C.CLASSES
    records = []
    for cls in classes:
        folder = os.path.join(dataset_path, cls)
        for fn in list_class_files(folder):
            path = os.path.join(folder, fn)
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:  # noqa: BLE001
                print(f"  unreadable: {path} ({e})")
                continue
            records.append({"path": os.path.relpath(path, dataset_path), "file": fn, "class": cls,
                            "ahash": str(imagehash.average_hash(img)),
                            "phash": imagehash.phash(img)})
    return pd.DataFrame(records)


def find_near_duplicates(df, threshold=DUP_HAMMING_THRESHOLD):
    pairs = []
    ph = df["phash"].tolist()
    for i, j in combinations(range(len(df)), 2):
        d = ph[i] - ph[j]
        if d <= threshold:
            pairs.append({"path_a": df.iloc[i]["path"], "class_a": df.iloc[i]["class"],
                          "path_b": df.iloc[j]["path"], "class_b": df.iloc[j]["class"],
                          "hamming_distance": int(d),
                          "cross_class": df.iloc[i]["class"] != df.iloc[j]["class"]})
    return pd.DataFrame(pairs, columns=["path_a", "class_a", "path_b", "class_b",
                                        "hamming_distance", "cross_class"])


def check_split_leakage(pairs, split_manifest_csv):
    """Flag near-duplicate pairs whose members fall in different partitions.

    Uses split_manifest.csv (file path -> partition) written by
    data.make_or_load_split, so the mapping cannot drift out of alignment with
    the loaded image order.
    """
    split = pd.read_csv(split_manifest_csv).set_index("path")["partition"].to_dict()
    if pairs.empty:
        return pairs.assign(partition_a=[], partition_b=[])
    pairs = pairs.copy()
    pairs["partition_a"] = pairs["path_a"].map(split)
    pairs["partition_b"] = pairs["path_b"].map(split)
    return pairs[pairs["partition_a"].notna() & pairs["partition_b"].notna()
                 & (pairs["partition_a"] != pairs["partition_b"])]


def run_duplicate_audit(output_dir=None):
    output_dir = output_dir or C.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    df = hash_images()
    print(f"Hashed {len(df)} images.")
    exact = df[df.duplicated("ahash", keep=False)].sort_values("ahash")
    near = find_near_duplicates(df)
    exact.drop(columns=["phash"]).to_csv(os.path.join(output_dir, "duplicates_exact.csv"), index=False)
    near.to_csv(os.path.join(output_dir, "duplicates_near.csv"), index=False)
    print(f"Exact-duplicate images: {len(exact)} | near-duplicate pairs (pHash<={DUP_HAMMING_THRESHOLD}): "
          f"{len(near)} ({int(near['cross_class'].sum()) if len(near) else 0} cross-class)")

    split_csv = os.path.join(C.CKPT_DIR, "split_manifest.csv")
    if os.path.exists(split_csv):
        leaks = check_split_leakage(near, split_csv)
        leaks.to_csv(os.path.join(output_dir, "duplicates_train_test_leakage.csv"), index=False)
        if len(leaks):
            print(f"WARNING: {len(leaks)} near-duplicate pairs straddle the train/test split.")
        else:
            print("No near-duplicate pairs straddle the train/test split.")
    else:
        print("split_manifest.csv not found - run training (or prepare_data) first to check leakage.")
    return df, exact, near
