from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SEGTRACK_ROOT = PROJECT_ROOT / "Segment-and-Track-Anything"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.inpaint_target_paths import (
    ensure_dir,
    get_raw_mask_from_sam2_dir,
    get_target_inpaint_root,
    normalize_target_id,
)
from utils.iterative_workflow import remove_path, write_json
from utils.pretrained_paths import deaot_checkpoint, segment_anything_checkpoint


@dataclass
class FrameRecord:
    stem: str
    removal_render_path: Path
    coarse_mask_path: Path
    coarse_mask: np.ndarray
    coarse_area: int
    bbox: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic inpaint-mask provider using virtual coarse masks and Segment-and-Track-Anything."
    )
    parser.add_argument("-m", "--model_path", required=True, help="Workspace model path.")
    parser.add_argument("--target_id", required=True, help="Target id or ids, e.g. 26 / 26_27 / [26,27].")
    parser.add_argument("--iteration", type=int, default=0, help="Virtual iteration tag.")
    parser.add_argument("--round_dir", type=str, default=None, help="Optional iterative round directory.")
    parser.add_argument(
        "--coarse_mask_source",
        type=str,
        default="virtual_objects_pred",
        choices=["virtual_objects_pred"],
        help="Source used to derive coarse target masks.",
    )
    parser.add_argument(
        "--prompt_frame_source",
        type=str,
        default="removal_render",
        choices=["removal_render"],
        help="Frame source on which SAM box prompting runs.",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA gpu id used by SAM/AOT.")
    parser.add_argument("--sam_model_type", type=str, default="vit_b", help="SAM backbone type.")
    parser.add_argument("--sam_checkpoint", type=str, default=None, help="Optional override for SAM checkpoint.")
    parser.add_argument("--aot_model", type=str, default="r50_deaotl", help="AOT model name.")
    parser.add_argument("--aot_checkpoint", type=str, default=None, help="Optional override for AOT checkpoint.")
    parser.add_argument(
        "--tracking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use AOT tracking to smooth masks across the virtual sequence.",
    )
    parser.add_argument(
        "--strict_tracking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail instead of falling back to SAM-only if AOT tracking cannot be initialized.",
    )
    parser.add_argument(
        "--clean_output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace the existing raw_mask_from_sam2 directory on each run.",
    )
    parser.add_argument(
        "--save_debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save coarse / independent / final debug artifacts.",
    )
    parser.add_argument("--points_per_side", type=int, default=16, help="SAM everything-generator setting.")
    parser.add_argument("--long_term_mem_gap", type=int, default=9999, help="AOT long-term memory gap.")
    parser.add_argument("--max_len_long_term", type=int, default=9999, help="AOT max long-term memory length.")
    parser.add_argument("--coarse_min_area", type=int, default=128, help="Minimum area for a coarse virtual mask component.")
    parser.add_argument("--coarse_dilate", type=int, default=7, help="Dilation kernel applied to coarse masks.")
    parser.add_argument("--bbox_expand_ratio", type=float, default=0.10, help="Relative padding added to prompt boxes.")
    parser.add_argument("--bbox_min_pad", type=int, default=12, help="Minimum pixel padding added to prompt boxes.")
    parser.add_argument("--prior_dilate", type=int, default=11, help="Extra dilation used when validating candidates.")
    parser.add_argument("--candidate_min_area", type=int, default=96, help="Minimum accepted SAM/AOT mask area.")
    parser.add_argument(
        "--candidate_min_inside_ratio",
        type=float,
        default=0.20,
        help="Minimum overlap ratio between a candidate mask and the coarse prior.",
    )
    parser.add_argument(
        "--candidate_min_bbox_iou",
        type=float,
        default=0.08,
        help="Minimum bbox IoU between a candidate mask and the prompt box.",
    )
    parser.add_argument(
        "--max_area_to_box_ratio",
        type=float,
        default=1.75,
        help="Reject candidates whose area is much larger than the prompt box.",
    )
    parser.add_argument("--final_close", type=int, default=5, help="Morphological close kernel for final masks.")
    parser.add_argument("--final_dilate", type=int, default=5, help="Optional final dilation applied to output masks.")
    parser.add_argument("--reference_refresh_gap", type=int, default=1, help="Add a new AOT reference every N frames.")
    return parser.parse_args()


def parse_target_ids(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            return parse_target_ids(json.loads(text))
        normalized = text.replace(",", " ").replace("_", " ")
        return [int(item) for item in normalized.split()]
    return [int(value)]


def resolve_provider_root(model_path: Path, target_tag: str, round_dir: str | None) -> Path:
    if round_dir is not None:
        return Path(round_dir).expanduser().resolve() / "mask_provider"
    return get_target_inpaint_root(model_path, target_tag) / "mask_provider_auto"


def ensure_sta_sys_path() -> None:
    for path in (SEGTRACK_ROOT, SEGTRACK_ROOT / "aot"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def resolve_checkpoint(path_value: str | None, default_path: str, root: Path, pretrained_default: Path | None = None) -> Path:
    if path_value:
        candidate = Path(path_value).expanduser()
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
    elif pretrained_default is not None:
        candidate = pretrained_default.resolve()
    else:
        candidate = (root / default_path).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")
    return candidate


def load_label_mask(mask_path: Path, target_ids: list[int]) -> np.ndarray:
    label_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if label_image is None:
        raise FileNotFoundError(f"Unable to read coarse label mask: {mask_path}")
    if label_image.ndim == 3:
        label_image = label_image[..., 0]
    return np.isin(label_image, target_ids).astype(np.uint8)


def morph_kernel(size: int) -> np.ndarray | None:
    if size <= 1:
        return None
    odd_size = size if size % 2 == 1 else size + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_size, odd_size))


def keep_main_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if int(binary.sum()) == 0:
        return binary

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    kept_any = False
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label_idx] = 1
            kept_any = True

    if kept_any:
        return cleaned

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cleaned[labels == largest_label] = 1
    return cleaned


def dilate_mask(mask: np.ndarray, size: int) -> np.ndarray:
    kernel = morph_kernel(size)
    if kernel is None:
        return (mask > 0).astype(np.uint8)
    return (cv2.dilate((mask > 0).astype(np.uint8), kernel) > 0).astype(np.uint8)


def close_mask(mask: np.ndarray, size: int) -> np.ndarray:
    kernel = morph_kernel(size)
    if kernel is None:
        return (mask > 0).astype(np.uint8)
    return (cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0).astype(np.uint8)


def make_bbox_from_mask(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.array([[int(xs.min()), int(ys.min())], [int(xs.max()), int(ys.max())]], dtype=np.int64)


def expand_bbox(bbox: np.ndarray, height: int, width: int, expand_ratio: float, min_pad: int) -> np.ndarray:
    x0, y0 = bbox[0]
    x1, y1 = bbox[1]
    box_w = max(1, x1 - x0 + 1)
    box_h = max(1, y1 - y0 + 1)
    pad_x = max(min_pad, int(round(box_w * expand_ratio)))
    pad_y = max(min_pad, int(round(box_h * expand_ratio)))
    return np.array(
        [
            [max(0, x0 - pad_x), max(0, y0 - pad_y)],
            [min(width - 1, x1 + pad_x), min(height - 1, y1 + pad_y)],
        ],
        dtype=np.int64,
    )


def bbox_area(bbox: np.ndarray | None) -> int:
    if bbox is None:
        return 0
    return max(0, int(bbox[1, 0] - bbox[0, 0] + 1)) * max(0, int(bbox[1, 1] - bbox[0, 1] + 1))


def bbox_iou(box_a: np.ndarray | None, box_b: np.ndarray | None) -> float:
    if box_a is None or box_b is None:
        return 0.0
    x0 = max(int(box_a[0, 0]), int(box_b[0, 0]))
    y0 = max(int(box_a[0, 1]), int(box_b[0, 1]))
    x1 = min(int(box_a[1, 0]), int(box_b[1, 0]))
    y1 = min(int(box_a[1, 1]), int(box_b[1, 1]))
    inter = max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)
    if inter == 0:
        return 0.0
    union = bbox_area(box_a) + bbox_area(box_b) - inter
    return float(inter) / float(max(1, union))


def make_rect_mask(shape: tuple[int, int], bbox: np.ndarray | None) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if bbox is None:
        return mask
    x0, y0 = bbox[0]
    x1, y1 = bbox[1]
    mask[y0 : y1 + 1, x0 : x1 + 1] = 1
    return mask


def overlap_ratio(mask: np.ndarray, prior_mask: np.ndarray) -> float:
    area = int(mask.sum())
    if area == 0:
        return 0.0
    overlap = int(np.logical_and(mask > 0, prior_mask > 0).sum())
    return float(overlap) / float(area)


def clean_candidate_mask(mask: np.ndarray, min_area: int, final_close: int, final_dilate: int) -> np.ndarray:
    cleaned = keep_main_components(mask, min_area)
    cleaned = close_mask(cleaned, final_close)
    cleaned = keep_main_components(cleaned, min_area)
    cleaned = dilate_mask(cleaned, final_dilate)
    cleaned = keep_main_components(cleaned, min_area)
    return cleaned


def candidate_is_reasonable(
    candidate_mask: np.ndarray,
    prior_mask: np.ndarray,
    prompt_bbox: np.ndarray | None,
    min_area: int,
    min_inside_ratio: float,
    min_bbox_iou: float,
    max_area_to_box_ratio: float,
) -> bool:
    area = int(candidate_mask.sum())
    if area < min_area:
        return False

    candidate_bbox = make_bbox_from_mask(candidate_mask)
    if candidate_bbox is None:
        return False

    prompt_area = bbox_area(prompt_bbox)
    if prompt_area > 0 and float(area) > float(prompt_area) * max_area_to_box_ratio:
        return False

    prior_area = int(prior_mask.sum())
    inside = overlap_ratio(candidate_mask, prior_mask) if prior_area > 0 else 0.0
    if prior_area > 0 and inside >= min_inside_ratio:
        return True

    return bbox_iou(candidate_bbox, prompt_bbox) >= min_bbox_iou


def select_mask(
    sam_mask: np.ndarray | None,
    tracked_mask: np.ndarray | None,
    coarse_mask: np.ndarray,
    prompt_bbox: np.ndarray | None,
    args: argparse.Namespace,
) -> np.ndarray:
    coarse_mask = (coarse_mask > 0).astype(np.uint8)
    prior_mask = coarse_mask if int(coarse_mask.sum()) > 0 else make_rect_mask(coarse_mask.shape, prompt_bbox)
    prior_mask = dilate_mask(prior_mask, args.prior_dilate)

    best_mask = None
    best_score = -1.0

    for candidate in (sam_mask, tracked_mask):
        if candidate is None:
            continue
        cleaned = clean_candidate_mask(candidate, args.candidate_min_area, args.final_close, 0)
        if not candidate_is_reasonable(
            cleaned,
            prior_mask,
            prompt_bbox,
            args.candidate_min_area,
            args.candidate_min_inside_ratio,
            args.candidate_min_bbox_iou,
            args.max_area_to_box_ratio,
        ):
            continue

        score = overlap_ratio(cleaned, prior_mask)
        if score > best_score:
            best_mask = cleaned
            best_score = score

    if best_mask is None:
        best_mask = coarse_mask if int(coarse_mask.sum()) > 0 else make_rect_mask(coarse_mask.shape, prompt_bbox)

    best_mask = close_mask(best_mask, args.final_close)
    best_mask = dilate_mask(best_mask, args.final_dilate)
    best_mask = keep_main_components(best_mask, args.candidate_min_area)
    return (best_mask > 0).astype(np.uint8)


def read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    Image.fromarray(((mask > 0).astype(np.uint8) * 255), mode="L").save(path)


def save_overlay(
    image_rgb: np.ndarray,
    coarse_mask: np.ndarray,
    independent_mask: np.ndarray,
    final_mask: np.ndarray,
    bbox: np.ndarray | None,
    path: Path,
) -> None:
    ensure_dir(path.parent)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = image_bgr.copy()

    overlay[coarse_mask > 0] = (overlay[coarse_mask > 0] * 0.55 + np.array([255, 0, 0]) * 0.45).astype(np.uint8)
    overlay[independent_mask > 0] = (overlay[independent_mask > 0] * 0.45 + np.array([0, 255, 0]) * 0.55).astype(np.uint8)
    overlay[final_mask > 0] = (overlay[final_mask > 0] * 0.35 + np.array([0, 0, 255]) * 0.65).astype(np.uint8)

    if bbox is not None:
        cv2.rectangle(overlay, tuple(bbox[0]), tuple(bbox[1]), (0, 255, 255), 2)

    cv2.imwrite(str(path), overlay)


def build_frame_records(args: argparse.Namespace, model_path: Path, target_ids: list[int]) -> list[FrameRecord]:
    if args.coarse_mask_source != "virtual_objects_pred":
        raise ValueError(f"Unsupported coarse_mask_source: {args.coarse_mask_source}")
    if args.prompt_frame_source != "removal_render":
        raise ValueError(f"Unsupported prompt_frame_source: {args.prompt_frame_source}")

    coarse_dir = model_path / "virtual" / f"ours_{args.iteration}" / "objects_pred"
    removal_dir = model_path / "virtual" / f"ours_object_removal/iteration_{args.iteration}" / "renders"

    if not coarse_dir.is_dir():
        raise FileNotFoundError(f"Virtual coarse-mask directory not found: {coarse_dir}")
    if not removal_dir.is_dir():
        raise FileNotFoundError(f"Virtual removal-render directory not found: {removal_dir}")

    render_paths = sorted(path for path in removal_dir.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not render_paths:
        raise FileNotFoundError(f"No rendered frames found under: {removal_dir}")

    records: list[FrameRecord] = []
    for render_path in render_paths:
        stem = render_path.stem
        coarse_mask_path = coarse_dir / f"{stem}.png"
        if not coarse_mask_path.is_file():
            raise FileNotFoundError(f"Coarse virtual label mask missing for frame {stem}: {coarse_mask_path}")

        coarse_mask = load_label_mask(coarse_mask_path, target_ids)
        coarse_mask = keep_main_components(coarse_mask, args.coarse_min_area)
        coarse_mask = dilate_mask(coarse_mask, args.coarse_dilate)

        height, width = coarse_mask.shape
        bbox = make_bbox_from_mask(coarse_mask)
        if bbox is not None:
            bbox = expand_bbox(bbox, height, width, args.bbox_expand_ratio, args.bbox_min_pad)

        records.append(
            FrameRecord(
                stem=stem,
                removal_render_path=render_path,
                coarse_mask_path=coarse_mask_path,
                coarse_mask=coarse_mask,
                coarse_area=int(coarse_mask.sum()),
                bbox=bbox,
            )
        )

    valid_boxes = [record.bbox for record in records if record.bbox is not None]
    if not valid_boxes:
        raise RuntimeError("No non-empty coarse virtual masks were found for the requested target ids.")

    valid_indices = [idx for idx, record in enumerate(records) if record.bbox is not None]
    for idx, record in enumerate(records):
        if record.bbox is not None:
            continue
        nearest_idx = min(valid_indices, key=lambda valid_idx: abs(valid_idx - idx))
        nearest_bbox = records[nearest_idx].bbox
        record.bbox = nearest_bbox.copy() if nearest_bbox is not None else None

    return records


def build_segmentor(args: argparse.Namespace):
    ensure_sta_sys_path()
    model_args = importlib.import_module("model_args")
    segmentor_module = importlib.import_module("tool.segmentor")

    sam_cfg = copy.deepcopy(model_args.sam_args)
    sam_cfg["gpu_id"] = args.gpu_id
    sam_cfg["model_type"] = args.sam_model_type
    sam_cfg["generator_args"]["points_per_side"] = args.points_per_side
    sam_cfg["sam_checkpoint"] = str(
        resolve_checkpoint(
            args.sam_checkpoint,
            sam_cfg["sam_checkpoint"],
            SEGTRACK_ROOT,
            pretrained_default=segment_anything_checkpoint(args.sam_model_type),
        )
    )

    return segmentor_module.Segmentor(sam_cfg)


def build_aot_tracker(args: argparse.Namespace):
    ensure_sta_sys_path()
    model_args = importlib.import_module("model_args")
    aot_tracker_module = importlib.import_module("aot_tracker")

    aot_cfg = copy.deepcopy(model_args.aot_args)
    aot_cfg["gpu_id"] = args.gpu_id
    aot_cfg["model"] = args.aot_model
    aot_cfg["long_term_mem_gap"] = args.long_term_mem_gap
    aot_cfg["max_len_long_term"] = args.max_len_long_term
    aot_cfg["model_path"] = str(
        resolve_checkpoint(
            args.aot_checkpoint,
            aot_cfg["model_path"],
            SEGTRACK_ROOT,
            pretrained_default=deaot_checkpoint(args.aot_model),
        )
    )

    return aot_tracker_module.get_aot(aot_cfg)


def run_independent_sam(
    records: list[FrameRecord],
    segmentor: Any,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    independent_masks: dict[str, np.ndarray] = {}
    for record in records:
        frame_rgb = read_rgb(record.removal_render_path)
        sam_mask = segmentor.segment_with_box(frame_rgb, record.bbox, reset_image=True)[0].astype(np.uint8)
        independent_masks[record.stem] = select_mask(
            sam_mask=sam_mask,
            tracked_mask=None,
            coarse_mask=record.coarse_mask,
            prompt_bbox=record.bbox,
            args=args,
        )
    return independent_masks


def tensor_to_binary_mask(mask_tensor: Any) -> np.ndarray:
    if mask_tensor is None:
        return np.zeros((1, 1), dtype=np.uint8)
    if hasattr(mask_tensor, "detach"):
        mask_np = mask_tensor.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask_tensor)
    mask_np = np.squeeze(mask_np)
    return (mask_np > 0).astype(np.uint8)


def run_tracking_direction(
    records: list[FrameRecord],
    order: list[int],
    anchor_index: int,
    base_masks: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    tracker = build_aot_tracker(args)
    anchor_record = records[anchor_index]
    anchor_frame = read_rgb(anchor_record.removal_render_path)
    anchor_mask = base_masks[anchor_record.stem].astype(np.uint8)
    tracker.add_reference_frame(anchor_frame, anchor_mask, obj_nums=1, frame_step=0)

    refined: dict[str, np.ndarray] = {}
    frame_step = 1
    for idx in order:
        record = records[idx]
        frame_rgb = read_rgb(record.removal_render_path)
        tracked_mask = tensor_to_binary_mask(tracker.track(frame_rgb))
        final_mask = select_mask(
            sam_mask=base_masks[record.stem],
            tracked_mask=tracked_mask,
            coarse_mask=record.coarse_mask,
            prompt_bbox=record.bbox,
            args=args,
        )
        refined[record.stem] = final_mask
        if args.reference_refresh_gap > 0 and (frame_step % args.reference_refresh_gap == 0):
            tracker.add_reference_frame(frame_rgb, final_mask.astype(np.uint8), obj_nums=1, frame_step=frame_step)
        frame_step += 1

    return refined


def save_outputs(
    records: list[FrameRecord],
    raw_mask_dir: Path,
    provider_root: Path,
    target_tag: str,
    args: argparse.Namespace,
    anchor_stem: str,
    independent_masks: dict[str, np.ndarray],
    final_masks: dict[str, np.ndarray],
    tracking_used: bool,
) -> Path:
    if args.clean_output and raw_mask_dir.exists():
        remove_path(raw_mask_dir)
    ensure_dir(raw_mask_dir)

    ensure_dir(provider_root)

    auto_root = provider_root / "auto_segtrack_bbox"
    if auto_root.exists():
        remove_path(auto_root)
    ensure_dir(auto_root)
    coarse_dir = ensure_dir(auto_root / "coarse_masks") if args.save_debug else auto_root / "coarse_masks"
    independent_dir = ensure_dir(auto_root / "independent_masks") if args.save_debug else auto_root / "independent_masks"
    final_dir = ensure_dir(auto_root / "final_masks") if args.save_debug else auto_root / "final_masks"
    overlay_dir = ensure_dir(auto_root / "overlays") if args.save_debug else auto_root / "overlays"

    frame_stats: list[dict[str, Any]] = []
    for record in records:
        independent_mask = independent_masks[record.stem]
        final_mask = final_masks[record.stem]

        save_binary_mask(final_mask, raw_mask_dir / f"{record.stem}.png")

        if args.save_debug:
            save_binary_mask(record.coarse_mask, coarse_dir / f"{record.stem}.png")
            save_binary_mask(independent_mask, independent_dir / f"{record.stem}.png")
            save_binary_mask(final_mask, final_dir / f"{record.stem}.png")
            frame_rgb = read_rgb(record.removal_render_path)
            save_overlay(
                frame_rgb,
                record.coarse_mask,
                independent_mask,
                final_mask,
                record.bbox,
                overlay_dir / f"{record.stem}.png",
            )

        frame_stats.append(
            {
                "frame": record.stem,
                "coarse_area": int(record.coarse_mask.sum()),
                "independent_area": int(independent_mask.sum()),
                "final_area": int(final_mask.sum()),
                "bbox": record.bbox.tolist() if record.bbox is not None else None,
            }
        )

    manifest = {
        "provider_type": "auto_segtrack_bbox",
        "target_id_tag": target_tag,
        "iteration": args.iteration,
        "tracking_requested": args.tracking,
        "tracking_used": tracking_used,
        "anchor_frame": anchor_stem,
        "frame_count": len(records),
        "raw_mask_dir": str(raw_mask_dir),
        "provider_root": str(provider_root),
        "settings": {
            "coarse_mask_source": args.coarse_mask_source,
            "prompt_frame_source": args.prompt_frame_source,
            "gpu_id": args.gpu_id,
            "sam_model_type": args.sam_model_type,
            "aot_model": args.aot_model,
            "coarse_min_area": args.coarse_min_area,
            "coarse_dilate": args.coarse_dilate,
            "bbox_expand_ratio": args.bbox_expand_ratio,
            "bbox_min_pad": args.bbox_min_pad,
            "prior_dilate": args.prior_dilate,
            "candidate_min_area": args.candidate_min_area,
            "candidate_min_inside_ratio": args.candidate_min_inside_ratio,
            "candidate_min_bbox_iou": args.candidate_min_bbox_iou,
            "max_area_to_box_ratio": args.max_area_to_box_ratio,
            "final_close": args.final_close,
            "final_dilate": args.final_dilate,
            "reference_refresh_gap": args.reference_refresh_gap,
        },
        "frames": frame_stats,
    }
    manifest_path = provider_root / "provider_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    target_ids = parse_target_ids(args.target_id)
    if not target_ids:
        raise ValueError("No valid target ids were provided.")
    target_tag = normalize_target_id(target_ids)

    provider_root = resolve_provider_root(model_path, target_tag, args.round_dir)
    raw_mask_dir = get_raw_mask_from_sam2_dir(model_path, target_tag)

    records = build_frame_records(args, model_path, target_ids)
    anchor_index = max(range(len(records)), key=lambda idx: records[idx].coarse_area)
    anchor_stem = records[anchor_index].stem

    segmentor = build_segmentor(args)
    independent_masks = run_independent_sam(records, segmentor, args)
    final_masks = dict(independent_masks)

    tracking_used = False
    if args.tracking:
        try:
            forward_order = list(range(anchor_index + 1, len(records)))
            backward_order = list(range(anchor_index - 1, -1, -1))
            if forward_order:
                final_masks.update(run_tracking_direction(records, forward_order, anchor_index, independent_masks, args))
            if backward_order:
                final_masks.update(run_tracking_direction(records, backward_order, anchor_index, independent_masks, args))
            tracking_used = True
        except Exception as exc:  # pragma: no cover - runtime fallback path
            if args.strict_tracking:
                raise
            warnings.warn(f"AOT tracking failed, falling back to SAM-only masks: {exc}", RuntimeWarning)

    manifest_path = save_outputs(
        records=records,
        raw_mask_dir=raw_mask_dir,
        provider_root=provider_root,
        target_tag=target_tag,
        args=args,
        anchor_stem=anchor_stem,
        independent_masks=independent_masks,
        final_masks=final_masks,
        tracking_used=tracking_used,
    )

    print(f"Saved automatic inpaint masks to: {raw_mask_dir}")
    print(f"Saved provider manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
