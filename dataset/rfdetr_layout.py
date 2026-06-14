"""Symlink processed data into the rfdetr-expected layout.

    data/rfdetr/
      train/_annotations.coco.json  + symlinks to augmented images
      valid/_annotations.coco.json  + symlinks to raw originals
      test/_annotations.coco.json   + symlinks to raw originals

Run: python -m dataset.rfdetr_layout
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_IMG = (
    REPO_ROOT / "data" / "raw" / "colorful_fashion_dataset_for_object_detection" / "JPEGImages"
)
AUG_IMG = REPO_ROOT / "data" / "processed" / "images" / "train_augmented"
ANN_DIR = REPO_ROOT / "data" / "processed" / "annotations"
OUT = REPO_ROOT / "data" / "rfdetr"

SPLITS = {
    "train": (AUG_IMG, ANN_DIR / "train_augmented.json"),
    "valid": (RAW_IMG, ANN_DIR / "val.json"),
    "test": (RAW_IMG, ANN_DIR / "test.json"),
}


def _bump_to_1indexed(coco: dict) -> dict:
    """rfdetr/pycocotools expect category_id >= 1; ours are 0-indexed. Bump."""
    for c in coco["categories"]:
        c["id"] = c["id"] + 1
    for a in coco["annotations"]:
        a["category_id"] = a["category_id"] + 1
    return coco


def main() -> int:
    for split, (img_src, ann_src) in SPLITS.items():
        if not ann_src.is_file():
            sys.exit(f"Missing {ann_src} — run dataset.prepare and dataset.augment first.")
        if not img_src.is_dir():
            sys.exit(f"Missing {img_src}")

        out_dir = OUT / split
        out_dir.mkdir(parents=True, exist_ok=True)

        coco = _bump_to_1indexed(json.loads(ann_src.read_text()))
        (out_dir / "_annotations.coco.json").write_text(json.dumps(coco))

        linked, missing = 0, 0
        for img in coco["images"]:
            src = img_src / img["file_name"]
            dst = out_dir / img["file_name"]
            if dst.exists() or dst.is_symlink():
                continue
            if not src.exists():
                missing += 1
                continue
            dst.symlink_to(src.resolve())
            linked += 1
        print(f"  {split}: {linked} symlinked, {missing} missing, {len(coco['images'])} total in COCO")

    print(f"\nLayout ready at {OUT.relative_to(REPO_ROOT)}  (categories 1-indexed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
