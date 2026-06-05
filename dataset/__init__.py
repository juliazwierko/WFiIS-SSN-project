"""Shared dataset tooling — download + prepare for the Colorful Fashion set.

Re-exports the functions teammates will most commonly import from training
code in sibling directories:

    from dataset import (
        discover_classes,
        make_split,
        voc_xml_to_objects,
        build_coco_split,
    )

Run-as-script entry points are kept on the submodules:
    python -m dataset.download
    python -m dataset.prepare
"""

from dataset.prepare import (
    build_coco_split,
    discover_classes,
    make_split,
    voc_xml_to_objects,
    write_classes_registry,
    write_coco_json,
)

__all__ = [
    "build_coco_split",
    "discover_classes",
    "make_split",
    "voc_xml_to_objects",
    "write_classes_registry",
    "write_coco_json",
]
