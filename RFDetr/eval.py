"""Evaluate fine-tuned RF-DETR Nano on the test split. Run: python -m RFDetr.eval"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRNano

from RFDetr.config import CHECKPOINT_DIR, REPO_ROOT, VIZ_DIR

DATASET_DIR = REPO_ROOT / "data" / "rfdetr"
TEST_DIR = DATASET_DIR / "test"
TEST_ANN = TEST_DIR / "_annotations.coco.json"
CHECKPOINT = CHECKPOINT_DIR / "checkpoint_best_ema.pth"
SCORE_THRESHOLD = 0.0          # for COCO eval — keep all preds
GRID_THRESHOLD = 0.5           # for visualization — only confident preds
GRID_N = 12                    # 4 cols x 3 rows
GRID_SEED = 0


def main() -> int:
    if not CHECKPOINT.is_file():
        sys.exit(f"Missing checkpoint at {CHECKPOINT}")
    if not TEST_ANN.is_file():
        sys.exit(f"Missing test annotations at {TEST_ANN}")

    coco_gt = COCO(str(TEST_ANN))
    cat_ids = sorted(coco_gt.getCatIds())
    cats = coco_gt.loadCats(cat_ids)
    num_classes = len(cat_ids)

    print(f"Loading {CHECKPOINT.name}  (num_classes={num_classes})")
    model = RFDETRNano(
        pretrain_weights=str(CHECKPOINT),
        num_classes=num_classes,
    )

    image_ids = coco_gt.getImgIds()
    predictions: list[dict] = []
    for i, image_id in enumerate(image_ids):
        info = coco_gt.loadImgs(image_id)[0]
        image_path = TEST_DIR / info["file_name"]
        if not image_path.exists():
            print(f"  [skip] missing {image_path}")
            continue
        det = model.predict(Image.open(image_path).convert("RGB"), threshold=SCORE_THRESHOLD)
        for cls_id, box, score in zip(det.class_id, det.xyxy, det.confidence):
            cat_id = _to_cat_id(int(cls_id), cat_ids)
            if cat_id is None:
                continue
            xmin, ymin, xmax, ymax = box.tolist()
            predictions.append({
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "score": float(score),
            })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(image_ids)} images  ({len(predictions)} detections so far)")

    if not predictions:
        sys.exit("No predictions produced.")

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    precision = coco_eval.eval["precision"]
    per_class = {}
    for k, cat in enumerate(cats):
        ap = precision[:, :, k, 0, -1]
        ap = ap[ap >= 0]
        per_class[cat["name"]] = float(ap.mean()) if ap.size else float("nan")

    out = {
        "evaluated_at": datetime.now().isoformat(),
        "model": "RF-DETR-Nano",
        "checkpoint": str(CHECKPOINT.relative_to(REPO_ROOT)),
        "test_split": str(TEST_ANN.relative_to(REPO_ROOT)),
        "num_test_images": len(image_ids),
        "num_predictions": len(predictions),
        "mAP_50_95": float(coco_eval.stats[0]),
        "mAP_50": float(coco_eval.stats[1]),
        "mAP_75": float(coco_eval.stats[2]),
        "mAP_small": float(coco_eval.stats[3]),
        "mAP_medium": float(coco_eval.stats[4]),
        "mAP_large": float(coco_eval.stats[5]),
        "mAR_max1": float(coco_eval.stats[6]),
        "mAR_max10": float(coco_eval.stats[7]),
        "mAR_max100": float(coco_eval.stats[8]),
        "per_class_mAP_50_95": per_class,
    }
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = VIZ_DIR / f"rfdetr_eval_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path.relative_to(REPO_ROOT)}")

    grid_path = VIZ_DIR / f"rfdetr_grid_{stamp}.png"
    _save_predictions_grid(model, coco_gt, cat_ids, cats, grid_path)
    print(f"Saved {grid_path.relative_to(REPO_ROOT)}")
    return 0


def _to_cat_id(cls_int: int, cat_ids: list[int]) -> int | None:
    """Model emits 0-indexed head positions; map to COCO cat_id (verified empirically)."""
    if 0 <= cls_int < len(cat_ids):
        return cat_ids[cls_int]
    return None


def _save_predictions_grid(model, coco_gt, cat_ids: list[int], cats: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name_by_cat_id = {c["id"]: c["name"] for c in cats}
    rng = random.Random(GRID_SEED)
    image_ids = rng.sample(coco_gt.getImgIds(), min(GRID_N, len(coco_gt.getImgIds())))

    cols = 4
    rows = (len(image_ids) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    box_ann = sv.BoxAnnotator(thickness=2)
    label_ann = sv.LabelAnnotator(text_thickness=1, text_scale=0.4)

    for ax, image_id in zip(axes, image_ids):
        info = coco_gt.loadImgs(image_id)[0]
        path = TEST_DIR / info["file_name"]
        image_np = np.array(Image.open(path).convert("RGB"))
        det = model.predict(Image.fromarray(image_np), threshold=GRID_THRESHOLD)
        labels = []
        for c, s in zip(det.class_id, det.confidence):
            cat_id = _to_cat_id(int(c), cat_ids)
            name = name_by_cat_id.get(cat_id, "?") if cat_id is not None else "?"
            labels.append(f"{name} {s:.2f}")
        out = box_ann.annotate(image_np.copy(), det)
        out = label_ann.annotate(out, det, labels)
        ax.imshow(out)
        ax.set_title(info["file_name"], fontsize=7)
        ax.axis("off")

    for ax in axes[len(image_ids):]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
