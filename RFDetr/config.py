"""Shared paths + model id for RF-DETR train/eval."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"

TRAIN_IMAGE_DIR = DATA_DIR / "images" / "train_augmented"
ORIG_IMAGE_DIR = (
    REPO_ROOT / "data" / "raw" / "colorful_fashion_dataset_for_object_detection" / "JPEGImages"
)

CLASSES_FILE = DATA_DIR / "classes.json"
TRAIN_ANN = DATA_DIR / "annotations" / "train_augmented.json"
VAL_ANN = DATA_DIR / "annotations" / "val.json"
TEST_ANN = DATA_DIR / "annotations" / "test.json"

CHECKPOINT_DIR = REPO_ROOT / "models" / "rfdetr"
VIZ_DIR = REPO_ROOT / "visualizations"

# Verify against https://huggingface.co/Roboflow if from_pretrained 404s.
MODEL_ID = "Roboflow/rf-detr-base"
