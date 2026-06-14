#!/usr/bin/env python3
"""Generate offline augmented copies of the train split.

Reads:
    data/processed/annotations/train.json      (frozen train split from prepare.py)
    data/raw/JPEGImages/*.jpg                  (source images)

Writes:
    data/processed/images/train_augmented/     (originals + augmented copies)
    data/processed/annotations/train_augmented.json

Output layout: each original image is copied through, then `--copies`
augmented variants are produced. With --copies=4 (default) every original
becomes 5 training samples (1 original + 4 augmented). The augmented
JSON keeps the frozen 0-indexed category_ids from the original train.json.

Run:  python -m dataset.augment
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_IMG_DIR = (
    REPO_ROOT / "data" / "raw" / "colorful_fashion_dataset_for_object_detection" / "JPEGImages"
)
DATA_DIR = REPO_ROOT / "data" / "processed"
TRAIN_ANN = DATA_DIR / "annotations" / "train.json"
OUT_IMG_DIR = DATA_DIR / "images" / "train_augmented"
OUT_ANN = DATA_DIR / "annotations" / "train_augmented.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--copies", type=int, default=4,
                   help="Augmented copies per original image (default: 4)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed — DO NOT change once frozen (same rule as the split).")
    p.add_argument("--jpeg-quality", type=int, default=92,
                   help="JPEG quality for augmented images (default: 92)")
    return p.parse_args()


def build_pipeline() -> A.Compose:
    """Photometric + light geometric augmentations, bbox-aware.

    `min_visibility=0.3` drops bboxes pushed mostly out of frame so the
    training data stays clean. Edit the list to tune — every teammate runs
    the same one because this file IS the augmentation spec for the project.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=(-10, 10), p=0.3),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_visibility=0.3,
        ),
    )


def main() -> int:
    args = parse_args()

    if not TRAIN_ANN.is_file():
        sys.exit(
            f"Missing {TRAIN_ANN}. Run `python -m dataset.prepare` first to "
            "produce the frozen train split."
        )

    # Seed the global RNGs that albumentations samples from.
    random.seed(args.seed)
    np.random.seed(args.seed)

    coco = json.loads(TRAIN_ANN.read_text())
    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ANN.parent.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline()

    new_images: list[dict] = []
    new_annotations: list[dict] = []
    next_image_id = 1
    next_ann_id = 1

    n_orig_images = len(images_by_id)
    print(f"Augmenting {n_orig_images} images × ({args.copies} aug + 1 orig) ...")

    for progress, (orig_id, info) in enumerate(sorted(images_by_id.items())):
        src_path = RAW_IMG_DIR / info["file_name"]
        if not src_path.exists():
            print(f"  [skip] missing source image: {src_path}")
            continue
        orig_anns = anns_by_image.get(orig_id, [])
        if not orig_anns:
            continue  # no point training on an image with no boxes

        # --- 1) Carry the original through unchanged ---
        shutil.copy2(src_path, OUT_IMG_DIR / info["file_name"])
        new_images.append({
            "id": next_image_id,
            "file_name": info["file_name"],
            "width": info["width"],
            "height": info["height"],
        })
        for ann in orig_anns:
            new_annotations.append({
                "id": next_ann_id,
                "image_id": next_image_id,
                "category_id": ann["category_id"],
                "bbox": list(ann["bbox"]),
                "area": float(ann["area"]),
                "iscrowd": 0,
            })
            next_ann_id += 1
        next_image_id += 1

        # --- 2) Generate `--copies` augmented variants ---
        image_np = np.array(Image.open(src_path).convert("RGB"))
        for copy_idx in range(args.copies):
            t = pipeline(
                image=image_np,
                bboxes=[a["bbox"] for a in orig_anns],
                category_ids=[a["category_id"] for a in orig_anns],
            )
            if not t["bboxes"]:
                # All bboxes were pushed below min_visibility — skip this copy.
                continue

            new_file = f"{src_path.stem}_aug{copy_idx}.jpg"
            Image.fromarray(t["image"]).save(OUT_IMG_DIR / new_file, quality=args.jpeg_quality)
            h, w = t["image"].shape[:2]
            new_images.append({
                "id": next_image_id,
                "file_name": new_file,
                "width": w,
                "height": h,
            })
            for bbox, cat_id in zip(t["bboxes"], t["category_ids"]):
                box_w, box_h = bbox[2], bbox[3]
                new_annotations.append({
                    "id": next_ann_id,
                    "image_id": next_image_id,
                    "category_id": cat_id,
                    "bbox": list(bbox),
                    "area": float(box_w * box_h),
                    "iscrowd": 0,
                })
                next_ann_id += 1
            next_image_id += 1

        if (progress + 1) % 100 == 0:
            print(f"  {progress + 1}/{n_orig_images} processed")

    out_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco["categories"],
    }
    OUT_ANN.write_text(json.dumps(out_coco, indent=2))
    print(
        f"\nDone. {len(new_images)} images, {len(new_annotations)} annotations.\n"
        f"  images -> {OUT_IMG_DIR.relative_to(REPO_ROOT)}\n"
        f"  coco   -> {OUT_ANN.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
