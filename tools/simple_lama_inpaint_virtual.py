import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.inpaint_target_paths import (
    ensure_dir,
    find_image_for_stem,
    get_after_2dinpaint_color_dir,
    get_after_2dinpaint_depth_dir,
    get_after_2dinpaint_depth_vis_dir,
    get_before_2dinpaint_color_dir,
    get_before_2dinpaint_depth_dir,
    get_before_2dinpaint_root,
)
from utils.pretrained_paths import configure_pretrained_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SimpleLaMa for target-scoped virtual-view color and depth data."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Model root containing target_inpaint/<target_id>/.",
    )
    parser.add_argument(
        "--target_id",
        required=True,
        help="Target object id used in target_inpaint/<target_id>/.",
    )
    parser.add_argument(
        "--mode",
        choices=("color", "depth", "both"),
        default="both",
        help="Which branch to process. Defaults to both.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed to simple_lama_inpainting.SimpleLama, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--mask_dilation",
        type=int,
        default=0,
        help="Optional extra dilation radius applied to *_mask.png before inpainting.",
    )
    return parser.parse_args()


def require_directory(path: Path) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}")
    return path


def init_lama(device: str):
    configure_pretrained_env(include_simple_lama=True)
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError as exc:
        raise ImportError(
            "simple_lama_inpainting is required. Install it in the environment "
            "used to run this script."
        ) from exc

    return SimpleLama(device=device)


def list_mask_paths(input_dir: Path):
    mask_paths = sorted(input_dir.glob("*_mask.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No *_mask.png files found in: {input_dir}")
    return mask_paths


def load_binary_mask(mask_path: Path, dilation_radius: int) -> Image.Image:
    if dilation_radius < 0:
        raise ValueError(f"--mask_dilation must be >= 0, got {dilation_radius}")

    with Image.open(mask_path) as mask:
        mask_np = np.asarray(mask.convert("L"), dtype=np.uint8)

    binary = np.where(mask_np > 0, 255, 0).astype(np.uint8)
    if dilation_radius > 0:
        kernel_size = 2 * dilation_radius + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        binary = cv2.dilate(binary, kernel, iterations=1)

    return Image.fromarray(binary, mode="L")


def ensure_size_match(image: Image.Image, mask: Image.Image, image_path: Path, mask_path: Path) -> None:
    if image.size != mask.size:
        raise RuntimeError(
            "Image/mask size mismatch: "
            f"{image_path} size={image.size}, {mask_path} size={mask.size}"
        )


def crop_output_to_input_size(
    output: Image.Image,
    input_size: tuple[int, int],
    input_path: Path,
) -> Image.Image:
    if output.size == input_size:
        return output

    input_width, input_height = input_size
    output_width, output_height = output.size
    if output_width < input_width or output_height < input_height:
        raise RuntimeError(
            "SimpleLaMa output is smaller than input: "
            f"{input_path} input={input_size}, output={output.size}"
        )

    # simple_lama_inpainting pads on the right/bottom to a modulo boundary,
    # so the valid original content stays in the top-left window.
    return output.crop((0, 0, input_width, input_height))


def compute_depth_range(depth_original: np.ndarray) -> tuple[float, float]:
    finite = depth_original[np.isfinite(depth_original)]
    if finite.size == 0:
        return 0.0, 1.0

    depth_min = float(finite.min())
    depth_max = float(finite.max())
    if depth_max <= depth_min:
        depth_max = depth_min + 1.0
    return depth_min, depth_max


def depth_to_rgb(depth: np.ndarray, depth_min: float, depth_max: float) -> Image.Image:
    depth = np.nan_to_num(depth, nan=depth_min, posinf=depth_max, neginf=depth_min).astype(np.float32)
    depth_norm = np.clip((depth - depth_min) / (depth_max - depth_min), 0.0, 1.0)
    depth_u8 = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)
    depth_rgb = np.repeat(depth_u8[..., None], 3, axis=2)
    return Image.fromarray(depth_rgb, mode="RGB")


def save_depth_outputs(output_dir: Path, vis_dir: Path, stem: str, output_image: Image.Image, depth_min: float, depth_max: float) -> None:
    output_rgb = np.asarray(output_image.convert("RGB"), dtype=np.uint8)
    output_gray = output_rgb[:, :, 0].astype(np.float32) / 255.0
    depth_completed = output_gray * (depth_max - depth_min) + depth_min
    np.save(output_dir / f"{stem}.npy", depth_completed.astype(np.float32))

    Image.fromarray(output_rgb, mode="RGB").save(vis_dir / f"{stem}.png")
    depth_jet = cv2.applyColorMap(output_rgb[:, :, 0], cv2.COLORMAP_JET)
    cv2.imwrite(str(vis_dir / f"{stem}_jet.png"), depth_jet)


def run_color_branch(simple_lama, model_path: str, target_id: str, mask_dilation: int) -> None:
    input_dir = require_directory(get_before_2dinpaint_color_dir(model_path, target_id))
    output_dir = ensure_dir(get_after_2dinpaint_color_dir(model_path, target_id))

    for mask_path in list_mask_paths(input_dir):
        stem = mask_path.stem[:-5]
        image_path = find_image_for_stem(input_dir, stem)

        with Image.open(image_path) as image:
            image = image.convert("RGB")
        mask = load_binary_mask(mask_path, mask_dilation)
        ensure_size_match(image, mask, image_path, mask_path)

        output = simple_lama(image, mask).convert("RGB")
        output = crop_output_to_input_size(output, image.size, image_path)
        output_path = output_dir / f"{stem}.png"
        output.save(output_path)
        print(f"[SimpleLaMa][color] {image_path.name} -> {output_path}")


def run_depth_branch(simple_lama, model_path: str, target_id: str, mask_dilation: int) -> None:
    input_dir = require_directory(get_before_2dinpaint_depth_dir(model_path, target_id))
    depth_original_dir = require_directory(input_dir / "depth_original")
    output_dir = ensure_dir(get_after_2dinpaint_depth_dir(model_path, target_id))
    vis_dir = ensure_dir(get_after_2dinpaint_depth_vis_dir(model_path, target_id))

    for mask_path in list_mask_paths(input_dir):
        stem = mask_path.stem[:-5]
        depth_path = input_dir / f"{stem}.npy"
        depth_original_path = depth_original_dir / f"{stem}.npy"
        if not depth_path.is_file():
            raise FileNotFoundError(f"Depth file not found: {depth_path}")
        if not depth_original_path.is_file():
            raise FileNotFoundError(f"Original depth file not found: {depth_original_path}")

        depth = np.load(depth_path)
        depth_original = np.load(depth_original_path)
        depth_min, depth_max = compute_depth_range(depth_original)

        image = depth_to_rgb(depth, depth_min, depth_max)
        mask = load_binary_mask(mask_path, mask_dilation)
        ensure_size_match(image, mask, depth_path, mask_path)

        output = simple_lama(image, mask).convert("RGB")
        output = crop_output_to_input_size(output, image.size, depth_path)
        save_depth_outputs(output_dir, vis_dir, stem, output, depth_min, depth_max)
        print(f"[SimpleLaMa][depth] {depth_path.name} -> {output_dir / (stem + '.npy')}")


def main() -> None:
    args = parse_args()
    simple_lama = init_lama(device=args.device)
    require_directory(get_before_2dinpaint_root(args.model_path, args.target_id))

    if args.mode in ("color", "both"):
        run_color_branch(simple_lama, args.model_path, args.target_id, args.mask_dilation)
    if args.mode in ("depth", "both"):
        run_depth_branch(simple_lama, args.model_path, args.target_id, args.mask_dilation)


if __name__ == "__main__":
    main()
