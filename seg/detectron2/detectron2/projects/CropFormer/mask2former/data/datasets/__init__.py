# Copyright (c) Facebook, Inc. and its affiliates.
# from . import (
#     register_ade20k_full,
#     register_ade20k_panoptic,
#     register_coco_stuff_10k,
#     register_mapillary_vistas,
#     register_coco_panoptic_annos_semseg,
#     register_ade20k_instance,
#     register_mapillary_vistas_panoptic,
#     register_entityv2_entity,
#     register_entityv2_instances,
#     register_entityv2_panoptic_350,
#     register_entityv2_semseg_150,
# )

import os
# === Disable automatic dataset registration globally ===
if os.getenv("DETECTRON2_DATASETS", "datasets").lower() == "none":
    print("⚠️ Skip importing all built-in dataset registration modules (DETECTRON2_DATASETS=none)")
else:
    from . import (
        register_ade20k_full,
        register_ade20k_panoptic,
        register_coco_stuff_10k,
        register_mapillary_vistas,
        register_coco_panoptic_annos_semseg,
        register_ade20k_instance,
        register_mapillary_vistas_panoptic,
        register_entityv2_entity,
        register_entityv2_instances,
        register_entityv2_panoptic_350,
        register_entityv2_semseg_150,
    )
