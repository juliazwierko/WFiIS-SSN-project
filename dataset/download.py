#!/usr/bin/env python3
"""Bootstrap: download the Colorful Fashion dataset from Kaggle into ./data/raw.

Setup (once):
    pip install -e .

Run:
    python -m dataset.download

Auth:
    kagglehub looks for credentials in (in order):
      1. env vars KAGGLE_USERNAME + KAGGLE_KEY
      2. ~/.kaggle/kaggle.json
      3. interactive prompt
    Generate kaggle.json from kaggle.com -> Account -> Create New API Token.
"""

from __future__ import annotations

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import kagglehub

KAGGLE_SLUG = "nguyngiabol/colorful-fashion-dataset-for-object-detection"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR = REPO_ROOT / "data" / "raw"


def download_to_dest() -> Path:
    """Pull the dataset via kagglehub and mirror it under DEST_DIR.

    kagglehub stores the archive in its own cache (~/.cache/kagglehub/...).
    We copy into the repo so every teammate has the same predictable path.
    """
    print(f"Downloading {KAGGLE_SLUG} via kagglehub ...")
    cached_path = Path(kagglehub.dataset_download(KAGGLE_SLUG))
    print(f"  cached at: {cached_path}")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for item in cached_path.iterdir():
        target = DEST_DIR / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    return DEST_DIR


def print_tree(root: Path, max_depth: int = 2) -> None:
    """Print up to `max_depth` levels of `root` so we can see the layout."""
    for path in sorted(root.rglob("*")):
        depth = len(path.relative_to(root).parts)
        if depth > max_depth:
            continue
        indent = "  " * (depth - 1)
        marker = "/" if path.is_dir() else ""
        print(f"{indent}{path.name}{marker}")


def _collect_classes(root: Path, labels: list[Path]) -> list[str]:
    """Best-effort class extraction. Tries names files first, then VOC, then YOLO."""
    # Strategy (a): explicit names file lives next to the data
    for nf in sorted(root.rglob("classes.txt")) + sorted(root.rglob("obj.names")):
        names = [ln.strip() for ln in nf.read_text().splitlines() if ln.strip()]
        if names:
            return names

    # Strategy (c): Pascal VOC — read <name> tags
    xml_labels = [p for p in labels if p.suffix.lower() == ".xml"]
    if xml_labels:
        seen: set[str] = set()
        for p in xml_labels:
            try:
                tree = ET.parse(p)
            except ET.ParseError:
                continue
            for tag in tree.findall(".//object/name"):
                if tag.text:
                    seen.add(tag.text.strip())
        if seen:
            return sorted(seen)

    # Strategy (b): YOLO txt — first token of each line is the class id
    txt_labels = [p for p in labels if p.suffix.lower() == ".txt"]
    if txt_labels:
        seen = set()
        for p in txt_labels:
            for ln in p.read_text().splitlines():
                ln = ln.strip()
                if ln:
                    seen.add(ln.split()[0])
        if seen:
            return sorted(seen)

    return []


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
LABEL_EXTS = {".txt", ".xml"}
# Files that LIVE in label dirs but aren't per-image annotations:
LABEL_FILE_DENYLIST = {"classes.txt", "obj.names", "readme.txt", "train.txt", "val.txt", "test.txt"}


def summarize(root: Path) -> dict:
    """Walk `root` and return summary stats (n_images, n_labels, classes).

    Useful as a sanity check before kicking off any training run.
    """
    images = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    labels = [
        p for p in root.rglob("*")
        if p.suffix.lower() in LABEL_EXTS and p.name.lower() not in LABEL_FILE_DENYLIST
    ]

    if not images:
        raise FileNotFoundError(f"No images (jpg/png) found under {root}")
    if not labels:
        raise FileNotFoundError(f"No label files (.txt/.xml) found under {root}")

    classes = _collect_classes(root, labels)

    return {
        "n_images": len(images),
        "n_labels": len(labels),
        "classes": classes,
    }


def main() -> int:
    dest = download_to_dest()
    print(f"\nDataset materialized at: {dest}\n")
    print("Top-level tree:")
    print_tree(dest, max_depth=2)
    print()

    stats = summarize(dest)
    print("Dataset stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
