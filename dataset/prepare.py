#!/usr/bin/env python3
"""Convert raw Colorful Fashion data into a shared, frozen COCO-format dataset.

This is the "common data" step. Every teammate (RF-DETR, YOLO, whatever else)
trains on the bit-identical files this script produces, so their metrics are
actually comparable.

Outputs (under data/processed/):
    classes.json                <- frozen class order { name -> id } (0-indexed)
    split.json                  <- frozen train/val/test image stem lists
    annotations/train.json      <- COCO-format annotations per split
    annotations/val.json
    annotations/test.json

Run:
    python -m dataset.prepare
    python -m dataset.prepare --seed 42 --val-frac 0.15 --test-frac 0.15

Or import individual pieces from sibling training code:
    from dataset import discover_classes, build_coco_split, make_split
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_DIR = REPO_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw", type=Path, default=RAW_DIR, help="Raw dataset root (default: data/raw)"
    )
    p.add_argument(
        "--out", type=Path, default=OUT_DIR, help="Output dir (default: data/processed)"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed — DO NOT change once the split is frozen.",
    )
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    return p.parse_args()


# ---------------------------------------------------------------------------
# 1) Class registry (frozen)
# ---------------------------------------------------------------------------


def discover_classes(labels_file: Path) -> list[str]:
    lines = labels_file.read_text().splitlines()
    classes = []
    for line in lines:
        label = line.strip().rstrip(",")
        if label:
            classes.append(label)

    return classes


def write_classes_registry(classes: list[str], path: Path) -> dict[str, int]:
    """Write classes.json and return a name->id map. (Plumbing — leave it.)"""
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"classes": classes, "class_to_id": class_to_id},
            indent=2,
        )
    )
    print(f"  wrote {path.relative_to(REPO_ROOT)}  ({len(classes)} classes)")
    return class_to_id


# ---------------------------------------------------------------------------
# 2) Train/val/test split (frozen)
# ---------------------------------------------------------------------------


def make_split(
    image_stems: list[str],
    seed: int,
    val_frac: float,
    test_frac: float,
) -> dict[str, list[str]]:
    assert val_frac + test_frac < 1.0, "train slice would be empty"

    rng = random.Random(seed)
    sorted_images = sorted(image_stems)

    rng.shuffle(sorted_images)

    n = len(sorted_images)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test

    train_stems = sorted_images[:n_train]
    val_stems = sorted_images[n_train : n_train + n_val]
    test_stems = sorted_images[n_train + n_val :]

    return {"train": train_stems, "val": val_stems, "test": test_stems}


# ---------------------------------------------------------------------------
# 3) Single-file VOC parse
# ---------------------------------------------------------------------------


def voc_xml_to_objects(xml_path: Path) -> tuple[int, int, list[dict]]:
    """Parse one VOC XML and return (image_width, image_height, objects).

    Each item in `objects` is:
        {"class_name": str, "bbox_xyxy": [xmin, ymin, xmax, ymax]}

    Every <object> in the XML is included (no <difficult> filtering — that's
    a downstream decision).
    """
    root = ET.parse(xml_path).getroot()

    width = int(root.findtext("size/width"))
    height = int(root.findtext("size/height"))

    objects: list[dict] = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        objects.append(
            {
                "class_name": obj.findtext("name"),
                "bbox_xyxy": [
                    int(float(bndbox.findtext("xmin"))),
                    int(float(bndbox.findtext("ymin"))),
                    int(float(bndbox.findtext("xmax"))),
                    int(float(bndbox.findtext("ymax"))),
                ],
            }
        )

    return width, height, objects


# ---------------------------------------------------------------------------
# 4) COCO assembly
# ---------------------------------------------------------------------------


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _find_image(image_dir: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image for stem {stem!r} under {image_dir}")


def build_coco_split(
    stems: list[str],
    image_dir: Path,
    label_dir: Path,
    class_to_id: dict[str, int],
) -> dict:
    """Assemble one COCO-format dict for a single split.

    Category ids are 0-indexed throughout (sunglass=0, hat=1, ...) to match
    classes.json. Note: this is non-standard for COCO — pycocotools, RF-DETR,
    etc. usually assume 1-indexed with 0 reserved for background. When wiring
    those up, configure them for 0-indexed categories (often a --no_background
    flag or `num_classes = len(classes)` rather than `len(classes) + 1`).
    """
    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1

    for image_id, stem in enumerate(stems, start=1):
        image_path = _find_image(image_dir, stem)
        w, h, objects = voc_xml_to_objects(label_dir / f"{stem}.xml")

        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": w,
                "height": h,
            }
        )

        for obj in objects:
            xmin, ymin, xmax, ymax = obj["bbox_xyxy"]
            box_w = xmax - xmin
            box_h = ymax - ymin
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": class_to_id[obj["class_name"]],
                    "bbox": [xmin, ymin, box_w, box_h],
                    "area": float(box_w * box_h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [{"id": idx, "name": name} for name, idx in class_to_id.items()]

    return {"images": images, "annotations": annotations, "categories": categories}


def write_coco_json(coco: dict, path: Path) -> None:
    """Write a COCO dict to disk. (Plumbing — leave it.)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco, indent=2))
    n_img = len(coco.get("images", []))
    n_ann = len(coco.get("annotations", []))
    print(f"  wrote {path.relative_to(REPO_ROOT)}  ({n_img} images, {n_ann} anns)")


# ---------------------------------------------------------------------------
# 5) Orchestration
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    # Standard VOC layout for the Colorful Fashion dataset. If your download
    # nests these one level deeper (e.g. args.raw / "ColorfulFashion"/...),
    # adjust here — the validation below will tell you what's missing.
    image_dir = (
        args.raw / "colorful_fashion_dataset_for_object_detection" / "JPEGImages"
    )
    label_dir = (
        args.raw / "colorful_fashion_dataset_for_object_detection" / "Annotations"
    )
    labels_file = (
        args.raw / "colorful_fashion_dataset_for_object_detection" / "labels.txt"
    )

    if not image_dir.is_dir() or not label_dir.is_dir() or not labels_file.is_file():
        sys.exit(
            "Expected the following to exist:\n"
            f"  image_dir   = {image_dir}\n"
            f"  label_dir   = {label_dir}\n"
            f"  labels_file = {labels_file}\n"
            "Run scripts/download_dataset.py first, then adjust the three "
            "paths in main() if your downloaded layout differs."
        )

    print("1) Discovering classes ...")
    classes = discover_classes(labels_file)
    class_to_id = write_classes_registry(classes, args.out / "classes.json")

    print("\n2) Building split ...")
    stems = sorted(p.stem for p in label_dir.glob("*.xml"))
    split = make_split(
        stems, seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac
    )
    (args.out).mkdir(parents=True, exist_ok=True)
    (args.out / "split.json").write_text(json.dumps(split, indent=2))
    for name, lst in split.items():
        print(f"  {name}: {len(lst)} images")

    print("\n3) Converting to COCO JSON ...")
    for name, lst in split.items():
        coco = build_coco_split(lst, image_dir, label_dir, class_to_id)
        write_coco_json(coco, args.out / "annotations" / f"{name}.json")

    print("\nDone. Outputs under:", args.out.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
