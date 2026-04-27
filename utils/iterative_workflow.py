from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence
from zipfile import ZIP_DEFLATED, ZipFile

from utils.inpaint_target_paths import (
    ensure_dir,
    get_after_2dinpaint_color_dir,
    get_after_2dinpaint_depth_dir,
    get_before_2dinpaint_color_dir,
    get_before_2dinpaint_depth_dir,
    normalize_target_id,
)
from utils.system_utils import searchForMaxIteration


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEG_TRACK_ASSETS_DIR = PROJECT_ROOT / "Segment-and-Track-Anything" / "assets"
LAMA_ROOT = PROJECT_ROOT / "LaMa"


def read_json(path: str | Path, default: Any | None = None) -> Any:
    path = Path(path)
    if not path.is_file():
        if default is not None:
            return default
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    return path


def normalize_id_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            return normalize_id_list(json.loads(text))
        return [int(item) for item in text.replace(",", " ").split()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [int(item) for item in value]
    return [int(value)]


def build_select_object_ids(target_id: Any, surrounding_ids: Any) -> list[int]:
    ordered = list(dict.fromkeys(normalize_id_list(target_id) + normalize_id_list(surrounding_ids)))
    return ordered


def get_iterative_root(model_path: str | Path) -> Path:
    return Path(model_path).expanduser().resolve() / "iterative_inpaint"


def get_rounds_root(iterative_root: str | Path) -> Path:
    return Path(iterative_root).expanduser().resolve() / "rounds"


def format_round_name(round_index: int, target_id: Any) -> str:
    target_tag = normalize_target_id(target_id)
    return f"{round_index:03d}_obj_{target_tag}"


def get_round_dir(iterative_root: str | Path, round_index: int, target_id: Any) -> Path:
    return get_rounds_root(iterative_root) / format_round_name(round_index, target_id)


def get_round_workspace(round_dir: str | Path) -> Path:
    return Path(round_dir).expanduser().resolve() / "workspace"


def get_round_meta_dir(round_dir: str | Path) -> Path:
    return Path(round_dir).expanduser().resolve() / "meta"


def get_round_meta_path(round_dir: str | Path) -> Path:
    return get_round_meta_dir(round_dir) / "round_meta.json"


def get_round_config_dir(round_dir: str | Path) -> Path:
    return Path(round_dir).expanduser().resolve() / "config"


def get_round_scene_in_dir(round_dir: str | Path) -> Path:
    return Path(round_dir).expanduser().resolve() / "scene_in"


def get_round_scene_out_dir(round_dir: str | Path) -> Path:
    return Path(round_dir).expanduser().resolve() / "scene_out"


def get_round_mask_provider_root(round_dir: str | Path) -> Path:
    return Path(round_dir).expanduser().resolve() / "mask_provider"


def get_round_mask_request_dir(round_dir: str | Path) -> Path:
    return get_round_mask_provider_root(round_dir) / "request"


def get_round_mask_request_images_dir(round_dir: str | Path) -> Path:
    return get_round_mask_request_dir(round_dir) / "images"


def get_round_mask_request_zip_path(round_dir: str | Path) -> Path:
    return get_round_mask_request_dir(round_dir) / "images.zip"


def get_round_mask_request_manifest_path(round_dir: str | Path) -> Path:
    return get_round_mask_request_dir(round_dir) / "request_manifest.json"


def get_round_lama_bridge_dir(round_dir: str | Path) -> Path:
    return Path(round_dir).expanduser().resolve() / "lama_bridge"


def get_round_lama_bridge_manifest_path(round_dir: str | Path) -> Path:
    return get_round_lama_bridge_dir(round_dir) / "bridge_manifest.json"


def get_round_scene_state_manifest_path(scene_state_dir: str | Path) -> Path:
    return Path(scene_state_dir).expanduser().resolve() / "state.json"


def remove_path(path: str | Path) -> None:
    path = Path(path)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def copy_tree(src_dir: str | Path, dst_dir: str | Path) -> Path:
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    return dst_dir


def copy_file(src_path: str | Path, dst_path: str | Path, prefer_hardlink: bool = False) -> Path:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    if not src_path.is_file():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    ensure_dir(dst_path.parent)
    if src_path.resolve() == dst_path.resolve():
        return dst_path
    if prefer_hardlink:
        try:
            if dst_path.exists() or dst_path.is_symlink():
                dst_path.unlink()
            os.link(src_path, dst_path)
            return dst_path
        except OSError:
            pass
    shutil.copy2(src_path, dst_path)
    return dst_path


def create_zip_from_dir(src_dir: str | Path, zip_path: str | Path) -> Path:
    src_dir = Path(src_dir)
    zip_path = Path(zip_path)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Zip source directory not found: {src_dir}")
    ensure_dir(zip_path.parent)
    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for file_path in sorted(src_dir.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(src_dir))
    return zip_path


def resolve_iteration(model_path: str | Path, iteration: int) -> int:
    if iteration == -1:
        return int(searchForMaxIteration(str(Path(model_path) / "point_cloud")))
    return int(iteration)


def _iteration_dir(model_path: str | Path, iteration: int) -> Path:
    model_path = Path(model_path).expanduser().resolve()
    return model_path / "point_cloud" / f"iteration_{iteration}"


def bootstrap_workspace_from_base_model(
    base_model_path: str | Path,
    workspace_model_path: str | Path,
    iteration: int,
    metadata: dict[str, Any] | None = None,
    prefer_hardlink: bool = False,
) -> dict[str, Any]:
    workspace_model_path = Path(workspace_model_path).expanduser().resolve()
    ensure_dir(workspace_model_path)

    resolved_iteration = resolve_iteration(base_model_path, iteration)
    source_iteration_dir = _iteration_dir(base_model_path, resolved_iteration)
    source_ply = source_iteration_dir / "point_cloud.ply"
    source_classifier = source_iteration_dir / "classifier.pth"

    target_iteration_dir = _iteration_dir(workspace_model_path, 0)
    ensure_dir(target_iteration_dir)
    copy_file(source_ply, target_iteration_dir / "point_cloud.ply", prefer_hardlink=prefer_hardlink)
    copy_file(source_classifier, target_iteration_dir / "classifier.pth")

    source_cfg_args = Path(base_model_path).expanduser().resolve() / "cfg_args"
    if source_cfg_args.is_file():
        copy_file(source_cfg_args, workspace_model_path / "cfg_args")

    manifest = {
        "source_type": "base_model",
        "source_model_path": str(Path(base_model_path).expanduser().resolve()),
        "source_iteration": resolved_iteration,
        "workspace_iteration": 0,
    }
    if metadata:
        manifest.update(metadata)

    write_json(workspace_model_path / "scene_state_bootstrap.json", manifest)
    return manifest


def bootstrap_workspace_from_snapshot(
    snapshot_dir: str | Path,
    workspace_model_path: str | Path,
    metadata: dict[str, Any] | None = None,
    prefer_hardlink: bool = False,
) -> dict[str, Any]:
    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    workspace_model_path = Path(workspace_model_path).expanduser().resolve()
    target_iteration_dir = _iteration_dir(workspace_model_path, 0)
    ensure_dir(target_iteration_dir)
    copy_file(snapshot_dir / "point_cloud.ply", target_iteration_dir / "point_cloud.ply", prefer_hardlink=prefer_hardlink)
    copy_file(snapshot_dir / "classifier.pth", target_iteration_dir / "classifier.pth")

    snapshot_cfg_args = snapshot_dir / "cfg_args"
    if snapshot_cfg_args.is_file():
        copy_file(snapshot_cfg_args, workspace_model_path / "cfg_args")

    manifest = {
        "source_type": "previous_round_snapshot",
        "source_snapshot_dir": str(snapshot_dir),
        "workspace_iteration": 0,
    }
    if metadata:
        manifest.update(metadata)

    write_json(workspace_model_path / "scene_state_bootstrap.json", manifest)
    return manifest


def save_scene_snapshot(
    snapshot_dir: str | Path,
    point_cloud_ply: str | Path,
    classifier_pth: str | Path,
    cfg_args_path: str | Path | None = None,
    state: dict[str, Any] | None = None,
    prefer_hardlink: bool = False,
) -> Path:
    snapshot_dir = ensure_dir(Path(snapshot_dir).expanduser().resolve())
    copy_file(point_cloud_ply, snapshot_dir / "point_cloud.ply", prefer_hardlink=prefer_hardlink)
    copy_file(classifier_pth, snapshot_dir / "classifier.pth")
    if cfg_args_path is not None and Path(cfg_args_path).is_file():
        copy_file(cfg_args_path, snapshot_dir / "cfg_args")
    write_json(get_round_scene_state_manifest_path(snapshot_dir), state or {})
    return snapshot_dir


def prepare_manual_mask_request(
    round_dir: str | Path,
    renders_dir: str | Path,
    sync_segment_track_assets: bool = True,
    extra_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    round_dir = Path(round_dir).expanduser().resolve()
    renders_dir = Path(renders_dir).expanduser().resolve()
    request_images_dir = get_round_mask_request_images_dir(round_dir)
    ensure_dir(get_round_mask_request_dir(round_dir))
    copy_tree(renders_dir, request_images_dir)
    zip_path = create_zip_from_dir(request_images_dir, get_round_mask_request_zip_path(round_dir))

    synced_zip_path = None
    if sync_segment_track_assets:
        ensure_dir(SEG_TRACK_ASSETS_DIR)
        synced_zip_path = copy_file(zip_path, SEG_TRACK_ASSETS_DIR / "images.zip")

    manifest = {
        "provider_type": "manual_sam2",
        "renders_dir": str(renders_dir),
        "request_images_dir": str(request_images_dir),
        "request_zip_path": str(zip_path),
        "synced_segment_track_assets_zip": str(synced_zip_path) if synced_zip_path else None,
    }
    if extra_manifest:
        manifest.update(extra_manifest)

    write_json(get_round_mask_request_manifest_path(round_dir), manifest)
    return manifest


def default_lama_data_name(round_index: int, target_id: Any, prefix: str = "iter360") -> str:
    target_tag = normalize_target_id(target_id).replace("/", "_")
    return f"{prefix}_round{round_index:03d}_{target_tag}"


def sync_before_2dinpaint_to_lama_project(model_path: str | Path, target_id: Any, data_name: str) -> dict[str, str]:
    color_src = get_before_2dinpaint_color_dir(model_path, target_id)
    depth_src = get_before_2dinpaint_depth_dir(model_path, target_id)
    color_dst = LAMA_ROOT / "data" / "color" / data_name
    depth_dst = LAMA_ROOT / "data" / "depth" / data_name
    copy_tree(color_src, color_dst)
    copy_tree(depth_src, depth_dst)
    return {
        "color_src": str(color_src),
        "depth_src": str(depth_src),
        "color_dst": str(color_dst),
        "depth_dst": str(depth_dst),
    }


def collect_lama_outputs_to_workspace(model_path: str | Path, target_id: Any, data_name: str) -> dict[str, str]:
    color_src = LAMA_ROOT / "output" / "color" / data_name
    depth_src = LAMA_ROOT / "output" / "depth" / data_name
    color_dst = get_after_2dinpaint_color_dir(model_path, target_id)
    depth_dst = get_after_2dinpaint_depth_dir(model_path, target_id)
    copy_tree(color_src, color_dst)
    copy_tree(depth_src, depth_dst)
    return {
        "color_src": str(color_src),
        "depth_src": str(depth_src),
        "color_dst": str(color_dst),
        "depth_dst": str(depth_dst),
    }


def resolve_support_ply(
    model_path: str | Path,
    iteration: int | str,
    support_view_name: str | None = None,
    explicit_support_ply: str | Path | None = None,
) -> Path:
    if explicit_support_ply:
        explicit_path = Path(explicit_support_ply).expanduser().resolve()
        if explicit_path.is_file():
            return explicit_path
        raise FileNotFoundError(f"Explicit support ply not found: {explicit_path}")

    if support_view_name:
        support_view_name = Path(str(support_view_name)).stem
        if support_view_name.isdigit():
            support_view_name = f"{int(support_view_name):05d}"

    iteration_tag = str(iteration)
    fusion_root = Path(model_path).expanduser().resolve() / "virtual" / f"ours_object_removal/iteration_{iteration_tag}"
    manifest_path = fusion_root / "fusion_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        fused_entries = manifest.get("fused_mask_col_dep_ply_files", [])
        if support_view_name:
            for entry in fused_entries:
                if Path(entry).stem == support_view_name:
                    candidate = Path(entry)
                    if candidate.is_file():
                        return candidate
        else:
            manifest_default = manifest.get("default_support_ply")
            if manifest_default:
                candidate = Path(manifest_default)
                if candidate.is_file():
                    return candidate

    fused_dir = fusion_root / "fused_mask_col_dep_ply"
    if not fused_dir.is_dir():
        raise FileNotFoundError(f"Support ply directory not found: {fused_dir}")

    if support_view_name:
        candidate = fused_dir / f"{support_view_name}.ply"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Support ply for view {support_view_name} not found under {fused_dir}")

    candidates = sorted(fused_dir.glob("*.ply"))
    if not candidates:
        raise FileNotFoundError(f"No support ply files found under: {fused_dir}")
    return candidates[len(candidates) // 2]
