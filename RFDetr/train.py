"""Train RF-DETR Nano via the rfdetr package. Run: python -m RFDetr.train"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
from datetime import datetime, timezone

from rfdetr import RFDETRNano

from RFDetr.config import CHECKPOINT_DIR, REPO_ROOT, VIZ_DIR

DATASET_DIR = REPO_ROOT / "data" / "rfdetr"

EPOCHS = 15
BATCH_SIZE = 8
GRAD_ACCUM = 2
LR = 1e-4


def main() -> int:
    if not (DATASET_DIR / "train" / "_annotations.coco.json").is_file():
        raise SystemExit(
            f"Missing {DATASET_DIR}. Run `python -m dataset.rfdetr_layout` first."
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()
    model = RFDETRNano()
    model.train(
        dataset_dir=str(DATASET_DIR),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM,
        lr=LR,
        output_dir=str(CHECKPOINT_DIR),
        tensorboard=True,
        checkpoint_interval=1,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (VIZ_DIR / f"rfdetr_run_{stamp}.json").write_text(json.dumps({
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "model": "RFDETRNano",
        "hyperparams": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM,
            "lr": LR,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        },
        "dataset_dir": str(DATASET_DIR.relative_to(REPO_ROOT)),
        "output_dir": str(CHECKPOINT_DIR.relative_to(REPO_ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
