from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG")


def normalize_target_id(target_id) -> str:
    if target_id is None:
        raise ValueError("target_id must not be None")

    if isinstance(target_id, str):
        text = target_id.strip()
        if not text:
            raise ValueError("target_id must not be empty")
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return text
            return normalize_target_id(parsed)
        return text

    if isinstance(target_id, Sequence) and not isinstance(target_id, (bytes, bytearray)):
        if len(target_id) == 0:
            raise ValueError("target_id sequence must not be empty")
        return "_".join(str(item) for item in target_id)

    return str(target_id)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_target_inpaint_root(model_path, target_id) -> Path:
    return Path(model_path).expanduser().resolve() / "target_inpaint" / normalize_target_id(target_id)


def get_raw_mask_from_sam2_dir(model_path, target_id) -> Path:
    return get_target_inpaint_root(model_path, target_id) / "raw_mask_from_sam2"


def get_unseen_mask_ready_dir(model_path, target_id) -> Path:
    return get_target_inpaint_root(model_path, target_id) / "unseen_mask_ready"


def get_before_2dinpaint_root(model_path, target_id) -> Path:
    return get_target_inpaint_root(model_path, target_id) / "before_2dinpaint"


def get_before_2dinpaint_color_dir(model_path, target_id) -> Path:
    return get_before_2dinpaint_root(model_path, target_id) / "color"


def get_before_2dinpaint_depth_dir(model_path, target_id) -> Path:
    return get_before_2dinpaint_root(model_path, target_id) / "depth"


def get_before_2dinpaint_depth_original_dir(model_path, target_id) -> Path:
    return get_before_2dinpaint_depth_dir(model_path, target_id) / "depth_original"


def get_after_2dinpaint_root(model_path, target_id) -> Path:
    return get_target_inpaint_root(model_path, target_id) / "after_2dinpaint"


def get_after_2dinpaint_color_dir(model_path, target_id) -> Path:
    return get_after_2dinpaint_root(model_path, target_id) / "color"


def get_after_2dinpaint_depth_dir(model_path, target_id) -> Path:
    return get_after_2dinpaint_root(model_path, target_id) / "depth"


def get_after_2dinpaint_depth_vis_dir(model_path, target_id) -> Path:
    return get_after_2dinpaint_depth_dir(model_path, target_id) / "vis"


def get_ready_for_3dinpaint_root(model_path, target_id) -> Path:
    return get_target_inpaint_root(model_path, target_id) / "ready_for_3dinpaint"


def get_ready_for_3dinpaint_color_dir(model_path, target_id) -> Path:
    return get_ready_for_3dinpaint_root(model_path, target_id) / "color"


def get_ready_for_3dinpaint_depth_completed_dir(model_path, target_id) -> Path:
    return get_ready_for_3dinpaint_root(model_path, target_id) / "depth_completed"


def find_image_for_stem(directory, stem: str, extensions=DEFAULT_IMAGE_EXTENSIONS) -> Path:
    directory = Path(directory)
    for ext in extensions:
        candidate = directory / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image not found for frame {stem} in {directory}")
