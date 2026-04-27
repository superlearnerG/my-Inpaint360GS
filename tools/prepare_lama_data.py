import argparse
import json
import os
import shutil
from distutils.dir_util import copy_tree
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.inpaint_target_paths import (
    ensure_dir,
    get_after_2dinpaint_color_dir,
    get_after_2dinpaint_depth_dir,
    get_before_2dinpaint_color_dir,
    get_before_2dinpaint_depth_dir,
    get_before_2dinpaint_depth_original_dir,
    get_raw_mask_from_sam2_dir,
    get_ready_for_3dinpaint_color_dir,
    get_ready_for_3dinpaint_depth_completed_dir,
    get_target_inpaint_root,
    get_unseen_mask_ready_dir,
)
from utils.iterative_workflow import (
    collect_lama_outputs_to_workspace,
    default_lama_data_name,
    sync_before_2dinpaint_to_lama_project,
)


def enlarge(input_dir, output_dir, expand_pixels=10, min_area=50):
    os.makedirs(output_dir, exist_ok=True)

    file_list = [f for f in os.listdir(input_dir) if f.endswith(".png")]

    for filename in tqdm(file_list, desc="Processing Masks", unit="file"):
        file_path = os.path.join(input_dir, filename)
        mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        mask[mask < 128] = 0
        mask[mask >= 128] = 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned_mask = np.zeros_like(mask)

        found_any = False
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned_mask[labels == i] = 255
                found_any = True

        if not found_any and num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cleaned_mask[labels == largest_label] = 255

        pil_mask = Image.fromarray(cleaned_mask).convert("L")
        expanded_mask = pil_mask.filter(ImageFilter.MaxFilter(size=2 * expand_pixels + 1))
        binary_mask = expanded_mask.point(lambda p: 255 if p > 128 else 0)

        output_path = os.path.join(output_dir, filename)
        binary_mask.save(output_path)

    print(f"✅ Saved cleaned unseen masks to: {output_dir}")


def resolve_raw_mask_input_dir(source_path, model_path, target_id):
    canonical_dir = get_raw_mask_from_sam2_dir(model_path, target_id)
    if canonical_dir.is_dir():
        return canonical_dir

    legacy_candidates = [
        Path(model_path).expanduser().resolve() / "[tmp]virtual_view_mask_from_sam2",
        Path(source_path).expanduser().resolve() / "[tmp]virtual_view_mask_from_sam2",
    ]

    for legacy_dir in legacy_candidates:
        if legacy_dir.is_dir():
            ensure_dir(canonical_dir)
            copy_tree(str(legacy_dir), str(canonical_dir))
            print(f"📁 Copied legacy SAM2 masks into target workspace: {canonical_dir}")
            return canonical_dir

    raise FileNotFoundError(
        "Virtual-view SAM2 mask directory not found. Expected target-scoped path "
        f"{canonical_dir}, or legacy paths: {legacy_candidates[0]}, {legacy_candidates[1]}"
    )


def iter_image_files(input_dir):
    valid_suffixes = {".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"}
    for path in sorted(Path(input_dir).iterdir()):
        if path.is_file() and path.suffix in valid_suffixes:
            yield path


def copy_inpaint2lama_virtual(args):
    source_path = args.source_path
    model_path = args.model_path

    if args.target_id is None:
        raise ValueError("--target_id is required when using --inpaint2lama")

    raw_mask_dir = resolve_raw_mask_input_dir(source_path, model_path, args.target_id)
    target_root = ensure_dir(get_target_inpaint_root(model_path, args.target_id))
    unseen_mask_ready_dir = ensure_dir(get_unseen_mask_ready_dir(model_path, args.target_id))
    enlarge(str(raw_mask_dir), str(unseen_mask_ready_dir))

    vanilla_virtual_image = Path(model_path) / f"virtual/ours_{args.iterations}"
    removed_virtual_image = Path(model_path) / f"virtual/ours_object_removal/iteration_{args.iterations}"

    before_color_dir = ensure_dir(get_before_2dinpaint_color_dir(model_path, args.target_id))
    before_depth_dir = ensure_dir(get_before_2dinpaint_depth_dir(model_path, args.target_id))
    before_depth_original_dir = ensure_dir(get_before_2dinpaint_depth_original_dir(model_path, args.target_id))

    print("📁 Copying virtual color images into before_2dinpaint/color ...")
    copy_tree(str(removed_virtual_image / "renders"), str(before_color_dir))

    print("📁 Copying original virtual depth into before_2dinpaint/depth/depth_original ...")
    copy_tree(str(vanilla_virtual_image / "depth"), str(before_depth_original_dir))

    print("📁 Copying removal virtual depth into before_2dinpaint/depth ...")
    copy_tree(str(removed_virtual_image / "depth"), str(before_depth_dir))

    print("📁 Copying unseen masks into before_2dinpaint package ...")
    for filename in os.listdir(unseen_mask_ready_dir):
        if filename.endswith(".png"):
            new_name = filename.replace(".png", "_mask.png")
            src_file = unseen_mask_ready_dir / filename
            shutil.copy2(src_file, before_depth_dir / new_name)
            shutil.copy2(src_file, before_color_dir / new_name)

    print(f"✅ Prepared target-scoped 2D inpaint inputs under: {target_root}")

    if args.sync_lama_project:
        data_name = args.lama_data_name or default_lama_data_name(
            args.round_index if args.round_index is not None else 0,
            args.target_id,
        )
        sync_info = sync_before_2dinpaint_to_lama_project(model_path, args.target_id, data_name)
        print(f"✅ Synced LaMa input data to project workspace: {data_name}")
        print(f"   color -> {sync_info['color_dst']}")
        print(f"   depth -> {sync_info['depth_dst']}")


def copy_lama2inpaint_virtual(args):
    model_path = args.model_path
    if args.target_id is None:
        raise ValueError("--target_id is required when organizing 2D inpaint outputs")

    target_root = ensure_dir(get_target_inpaint_root(model_path, args.target_id))
    after_color_dir = get_after_2dinpaint_color_dir(model_path, args.target_id)
    after_depth_dir = get_after_2dinpaint_depth_dir(model_path, args.target_id)
    ready_color_dir = ensure_dir(get_ready_for_3dinpaint_color_dir(model_path, args.target_id))
    ready_depth_completed_dir = ensure_dir(get_ready_for_3dinpaint_depth_completed_dir(model_path, args.target_id))

    if args.sync_lama_project:
        data_name = args.lama_data_name or default_lama_data_name(
            args.round_index if args.round_index is not None else 0,
            args.target_id,
        )
        sync_info = collect_lama_outputs_to_workspace(model_path, args.target_id, data_name)
        print(f"✅ Pulled LaMa outputs from project workspace: {data_name}")
        print(f"   color <- {sync_info['color_src']}")
        print(f"   depth <- {sync_info['depth_src']}")

    if not after_color_dir.is_dir():
        raise FileNotFoundError(f"2D inpaint color output not found: {after_color_dir}")
    if not after_depth_dir.is_dir():
        raise FileNotFoundError(f"2D inpaint depth output not found: {after_depth_dir}")

    for image_path in iter_image_files(after_color_dir):
        target_path = ready_color_dir / f"{image_path.stem}.JPG"
        with Image.open(image_path) as image:
            image.convert("RGB").save(target_path, format="JPEG", quality=95)

    for depth_path in sorted(after_depth_dir.glob("*.npy")):
        shutil.copy2(depth_path, ready_depth_completed_dir / depth_path.name)

    print(f"✅ Organized final 3D-inpaint inputs under: {target_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and organize target-scoped 2D inpaint data")
    parser.add_argument("-s", "--source_path", type=str, required=True, help="e.g. data/inpaint360/doppelherz")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="e.g. output/inpaint360/doppelherz")
    parser.add_argument("-r", "--resolution", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=None, help="Virtual/removal iteration tag. Defaults to train_distill iterations for legacy usage.")
    parser.add_argument("--target_id", type=str, default=None, help="Target object id used in target_inpaint/<target_id>/")
    parser.add_argument("--inpaint2lama", action="store_true", help="prepare target-scoped inputs for 2D inpainting")
    parser.add_argument("--sync_lama_project", action=argparse.BooleanOptionalAction, default=False, help="Sync prepared inputs to/from my-Inpaint360GS/LaMa project directories.")
    parser.add_argument("--lama_data_name", type=str, default=None, help="External LaMa data name, e.g. iter360_round000_26.")
    parser.add_argument("--round_index", type=int, default=None, help="Optional round index used when auto-generating lama_data_name.")
    args = parser.parse_args()

    if args.iterations is None:
        with open("config/object_distill/train_distill.json", "r") as file:
            config = json.load(file)
        args.iterations = config.get("iterations")

    if args.inpaint2lama:
        copy_inpaint2lama_virtual(args)
    else:
        copy_lama2inpaint_virtual(args)
