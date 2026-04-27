from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.inpaint_target_paths import get_raw_mask_from_sam2_dir, normalize_target_id
from utils.iterative_workflow import (
    bootstrap_workspace_from_base_model,
    bootstrap_workspace_from_snapshot,
    build_select_object_ids,
    default_lama_data_name,
    get_iterative_root,
    get_round_config_dir,
    get_round_dir,
    get_round_lama_bridge_manifest_path,
    get_round_mask_request_manifest_path,
    get_round_meta_path,
    get_round_scene_in_dir,
    get_round_scene_out_dir,
    get_round_workspace,
    normalize_id_list,
    read_json,
    remove_path,
    save_scene_snapshot,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round-based iterative inpaint workflow for my-Inpaint360GS."
    )
    parser.add_argument(
        "command",
        choices=["init", "prepare-round", "run-mask-provider", "prepare-lama", "run-simple-lama", "finalize-round", "status"],
        help="Workflow action.",
    )
    parser.add_argument("-s", "--source_path", required=True, help="Scene source path.")
    parser.add_argument("-m", "--model_path", required=True, help="Base model path with distillation outputs.")
    parser.add_argument("--workflow_config", required=True, help="Iterative workflow JSON config.")
    parser.add_argument("--round_index", type=int, default=None, help="Round index for round-scoped actions.")
    parser.add_argument("--python_bin", default=sys.executable, help="Python executable used for stage scripts.")
    parser.add_argument("--force", action="store_true", help="Reset round workspace before prepare-round.")
    parser.add_argument(
        "--render_intermediate",
        action="store_true",
        help="Render intermediate removal/inpaint visualizations for non-final rounds. By default only the final round render is kept.",
    )
    parser.add_argument("--render_video", action="store_true", help="Forward render_video to removal/inpaint stage scripts.")
    parser.add_argument("--simple_lama_device", default="cuda", help="Device for tools/simple_lama_inpaint_virtual.py.")
    parser.add_argument("--mask_dilation", type=int, default=0, help="Mask dilation passed to simple LaMa helper.")
    parser.add_argument(
        "--storage_mode",
        choices=["full", "lite", "minimal"],
        default="full",
        help="Output retention mode. full preserves legacy artifacts; lite/minimal reduce intermediate storage.",
    )
    return parser.parse_args()


def load_workflow_config(path: str | Path) -> dict[str, Any]:
    config = read_json(path)
    config.setdefault("defaults", {})
    config.setdefault("mask_provider", {})
    config.setdefault("lama", {})
    rounds = config.get("rounds")
    if not rounds:
        raise ValueError(f"'rounds' must be a non-empty list in workflow config: {path}")
    return config


def iterative_root_for(model_path: str | Path) -> Path:
    return get_iterative_root(model_path)


def workflow_manifest_path(model_path: str | Path) -> Path:
    return iterative_root_for(model_path) / "workflow_manifest.json"


def ensure_workflow_initialized(args: argparse.Namespace, workflow: dict[str, Any]) -> Path:
    iterative_root = iterative_root_for(args.model_path)
    iterative_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_path": str(Path(args.source_path).expanduser().resolve()),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "workflow_config": str(Path(args.workflow_config).expanduser().resolve()),
        "round_count": len(workflow["rounds"]),
    }
    write_json(workflow_manifest_path(args.model_path), manifest)
    return iterative_root


def resolve_round_spec(workflow: dict[str, Any], round_index: int) -> dict[str, Any]:
    if round_index < 0 or round_index >= len(workflow["rounds"]):
        raise IndexError(f"round_index out of range: {round_index}")
    defaults = workflow.get("defaults", {})
    spec = dict(defaults)
    spec.update(workflow["rounds"][round_index])
    spec["target_id"] = normalize_id_list(spec.get("target_id"))
    spec["surrounding_ids"] = normalize_id_list(spec.get("surrounding_ids"))
    if not spec["target_id"]:
        raise ValueError(f"Round {round_index} has empty target_id")
    spec["select_obj_id"] = build_select_object_ids(spec["target_id"], spec["surrounding_ids"])
    spec.setdefault("removal_thresh", 0.7)
    spec.setdefault("lambda_dssim", 0.8)
    spec.setdefault("opacity_init", 0.1)
    spec.setdefault("lambda_lpips", 0.0005)
    spec.setdefault("finetune_iteration", 5000)
    spec.setdefault("circle_radius", -1.0)
    spec.setdefault("target_object_radius", -1.0)
    spec.setdefault("resolution", 1)

    mask_provider_cfg = dict(workflow.get("mask_provider", {}))
    if isinstance(spec.get("mask_provider"), dict):
        mask_provider_cfg.update(spec["mask_provider"])
    spec["mask_provider"] = mask_provider_cfg
    spec.setdefault("mask_provider_type", mask_provider_cfg.get("type", "manual_sam2"))
    spec.setdefault("sync_segment_track_assets", mask_provider_cfg.get("sync_segment_track_assets", True))

    spec.setdefault("lama_backend", workflow.get("lama", {}).get("backend", "external"))
    spec.setdefault(
        "lama_data_name",
        default_lama_data_name(
            round_index,
            spec["target_id"],
            prefix=workflow.get("lama", {}).get("data_name_prefix", "iter360"),
        ),
    )
    return spec


def round_paths(args: argparse.Namespace, spec: dict[str, Any], round_index: int) -> dict[str, Path]:
    iterative_root = ensure_workflow_initialized(args, load_workflow_config(args.workflow_config))
    round_dir = get_round_dir(iterative_root, round_index, spec["target_id"])
    return {
        "iterative_root": iterative_root,
        "round_dir": round_dir,
        "workspace": get_round_workspace(round_dir),
        "config_dir": get_round_config_dir(round_dir),
        "scene_in": get_round_scene_in_dir(round_dir),
        "scene_out": get_round_scene_out_dir(round_dir),
        "meta_path": get_round_meta_path(round_dir),
    }


def is_last_round(workflow: dict[str, Any], round_index: int) -> bool:
    return round_index == len(workflow["rounds"]) - 1


def round_meta_base(args: argparse.Namespace, round_index: int, spec: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "round_index": round_index,
        "target_id": spec["target_id"],
        "target_id_tag": normalize_target_id(spec["target_id"]),
        "surrounding_ids": spec["surrounding_ids"],
        "select_obj_id": spec["select_obj_id"],
        "workspace_model_path": str(paths["workspace"]),
        "scene_in_dir": str(paths["scene_in"]),
        "scene_out_dir": str(paths["scene_out"]),
        "storage_mode": getattr(args, "storage_mode", "full"),
        "removal_config_path": str(paths["config_dir"] / "object_removal.json"),
        "inpaint_config_path": str(paths["config_dir"] / "object_inpaint.json"),
        "lama_data_name": spec["lama_data_name"],
        "mask_provider_type": spec["mask_provider_type"],
        "lama_backend": spec["lama_backend"],
    }


def load_round_meta(paths: dict[str, Path], base_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    if paths["meta_path"].is_file():
        return read_json(paths["meta_path"])
    return dict(base_meta or {})


def save_round_meta(paths: dict[str, Path], meta: dict[str, Any]) -> None:
    write_json(paths["meta_path"], meta)


def build_round_configs(paths: dict[str, Path], spec: dict[str, Any]) -> tuple[Path, Path]:
    paths["config_dir"].mkdir(parents=True, exist_ok=True)
    removal_config = {
        "removal_thresh": spec["removal_thresh"],
        "select_obj_id": spec["select_obj_id"],
        "target_id": spec["target_id"],
        "surrounding_ids": spec["surrounding_ids"],
        "target_object_radius": spec["target_object_radius"],
        "circle_radius": spec["circle_radius"],
    }
    inpaint_config = {
        "removal_thresh": spec["removal_thresh"],
        "select_obj_id": spec["select_obj_id"],
        "images": "images_inpaint_unseen_virtual",
        "object_path": "inpaint_2d_unseen_mask_virtual",
        "lambda_dssim": spec["lambda_dssim"],
        "opacity_init": spec["opacity_init"],
        "lambda_lpips": spec["lambda_lpips"],
        "finetune_iteration": spec["finetune_iteration"],
        "target_id": spec["target_id"],
        "surrounding_ids": spec["surrounding_ids"],
        "target_object_radius": spec["target_object_radius"],
        "circle_radius": spec["circle_radius"],
    }
    removal_path = write_json(paths["config_dir"] / "object_removal.json", removal_config)
    inpaint_path = write_json(paths["config_dir"] / "object_inpaint.json", inpaint_config)
    return removal_path, inpaint_path


def ensure_round_workspace(
    args: argparse.Namespace,
    workflow: dict[str, Any],
    round_index: int,
    spec: dict[str, Any],
    paths: dict[str, Path],
    force: bool = False,
    storage_mode: str = "full",
) -> None:
    if force and paths["round_dir"].exists():
        remove_path(paths["round_dir"])
    paths["round_dir"].mkdir(parents=True, exist_ok=True)

    if paths["workspace"].exists():
        return

    prefer_hardlink = storage_mode == "minimal"
    if round_index == 0:
        bootstrap_manifest = bootstrap_workspace_from_base_model(
            args.model_path,
            paths["workspace"],
            iteration=int(workflow.get("base_iteration", -1)),
            metadata={"round_index": round_index},
            prefer_hardlink=prefer_hardlink,
        )
    else:
        prev_spec = resolve_round_spec(workflow, round_index - 1)
        prev_round_dir = get_round_dir(paths["iterative_root"], round_index - 1, prev_spec["target_id"])
        prev_scene_out = get_round_scene_out_dir(prev_round_dir)
        if not (prev_scene_out / "point_cloud.ply").is_file():
            raise FileNotFoundError(
                f"Previous round scene_out missing. Complete round {round_index - 1} first: {prev_scene_out}"
            )
        bootstrap_manifest = bootstrap_workspace_from_snapshot(
            prev_scene_out,
            paths["workspace"],
            metadata={"round_index": round_index, "previous_round": round_index - 1},
            prefer_hardlink=prefer_hardlink,
        )

    build_round_configs(paths, spec)
    save_scene_snapshot(
        paths["scene_in"],
        paths["workspace"] / "point_cloud" / "iteration_0" / "point_cloud.ply",
        paths["workspace"] / "point_cloud" / "iteration_0" / "classifier.pth",
        cfg_args_path=paths["workspace"] / "cfg_args",
        state=bootstrap_manifest,
        prefer_hardlink=prefer_hardlink,
    )


def cleanup_minimal_round(paths: dict[str, Path]) -> list[str]:
    removed: list[str] = []
    for rel_path in [
        "virtual",
        "target_inpaint",
        "point_cloud_object_removal",
        "point_cloud_object_inpaint_virtual",
        "point_cloud_vis",
    ]:
        path = paths["workspace"] / rel_path
        if path.exists() or path.is_symlink():
            remove_path(path)
            removed.append(str(path))
    return removed


def run_cmd(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command))
    env = os.environ.copy()
    project_root_str = str(PROJECT_ROOT)
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        env["PYTHONPATH"] = project_root_str + os.pathsep + existing_pythonpath
    else:
        env["PYTHONPATH"] = project_root_str
    subprocess.run(command, cwd=str(cwd or PROJECT_ROOT), env=env, check=True)


def append_bool_flag(command: list[str], flag_name: str, enabled: bool) -> None:
    command.append(f"--{flag_name}" if enabled else f"--no-{flag_name}")


def round_mask_provider_manifest_path(paths: dict[str, Path]) -> Path:
    return paths["round_dir"] / "mask_provider" / "provider_manifest.json"


def run_mask_provider(args: argparse.Namespace, spec: dict[str, Any], paths: dict[str, Path], meta: dict[str, Any]) -> dict[str, Any]:
    provider_type = spec["mask_provider_type"]
    provider_cfg = dict(spec.get("mask_provider") or {})

    if provider_type == "manual_sam2":
        request_manifest = read_json(get_round_mask_request_manifest_path(paths["round_dir"]), default={})
        meta["status"] = "mask_request_ready"
        meta["mask_request"] = request_manifest
        meta["mask_provider"] = {
            "provider_type": provider_type,
            "request_manifest": request_manifest,
        }
        return meta

    if provider_type == "auto_segtrack_bbox":
        target_tag = normalize_target_id(spec["target_id"])
        provider_cmd = [
            args.python_bin,
            "tools/auto_mask_provider_segtrack.py",
            "--model_path",
            str(paths["workspace"]),
            "--target_id",
            target_tag,
            "--iteration",
            "0",
            "--round_dir",
            str(paths["round_dir"]),
            "--coarse_mask_source",
            str(provider_cfg.get("coarse_mask_source", "virtual_objects_pred")),
            "--prompt_frame_source",
            str(provider_cfg.get("prompt_frame_source", "removal_render")),
            "--gpu_id",
            str(int(provider_cfg.get("gpu_id", 0))),
            "--sam_model_type",
            str(provider_cfg.get("sam_model_type", "vit_b")),
            "--aot_model",
            str(provider_cfg.get("aot_model", "r50_deaotl")),
            "--points_per_side",
            str(int(provider_cfg.get("points_per_side", 16))),
            "--long_term_mem_gap",
            str(int(provider_cfg.get("long_term_mem_gap", 9999))),
            "--max_len_long_term",
            str(int(provider_cfg.get("max_len_long_term", 9999))),
            "--coarse_min_area",
            str(int(provider_cfg.get("coarse_min_area", 128))),
            "--coarse_dilate",
            str(int(provider_cfg.get("coarse_dilate", 7))),
            "--bbox_expand_ratio",
            str(float(provider_cfg.get("bbox_expand_ratio", 0.10))),
            "--bbox_min_pad",
            str(int(provider_cfg.get("bbox_min_pad", 12))),
            "--prior_dilate",
            str(int(provider_cfg.get("prior_dilate", 11))),
            "--candidate_min_area",
            str(int(provider_cfg.get("candidate_min_area", 96))),
            "--candidate_min_inside_ratio",
            str(float(provider_cfg.get("candidate_min_inside_ratio", 0.20))),
            "--candidate_min_bbox_iou",
            str(float(provider_cfg.get("candidate_min_bbox_iou", 0.08))),
            "--max_area_to_box_ratio",
            str(float(provider_cfg.get("max_area_to_box_ratio", 1.75))),
            "--final_close",
            str(int(provider_cfg.get("final_close", 5))),
            "--final_dilate",
            str(int(provider_cfg.get("final_dilate", 5))),
            "--reference_refresh_gap",
            str(int(provider_cfg.get("reference_refresh_gap", 1))),
        ]

        if provider_cfg.get("sam_checkpoint"):
            provider_cmd.extend(["--sam_checkpoint", str(provider_cfg["sam_checkpoint"])])
        if provider_cfg.get("aot_checkpoint"):
            provider_cmd.extend(["--aot_checkpoint", str(provider_cfg["aot_checkpoint"])])

        append_bool_flag(provider_cmd, "tracking", bool(provider_cfg.get("tracking", True)))
        append_bool_flag(provider_cmd, "strict_tracking", bool(provider_cfg.get("strict_tracking", False)))
        append_bool_flag(provider_cmd, "clean_output", bool(provider_cfg.get("clean_output", True)))
        append_bool_flag(provider_cmd, "save_debug", bool(provider_cfg.get("save_debug", True)))

        run_cmd(provider_cmd)

        provider_manifest = read_json(round_mask_provider_manifest_path(paths), default={})
        meta["status"] = "masks_ready"
        meta["mask_provider"] = provider_manifest
        meta.pop("mask_request", None)
        return meta

    meta["status"] = "virtual_ready"
    meta["mask_provider"] = {
        "provider_type": provider_type,
        "note": "Populate target_inpaint/<target_id>/raw_mask_from_sam2 with the external mask provider.",
    }
    return meta


def run_prepare_round(args: argparse.Namespace, workflow: dict[str, Any], round_index: int) -> None:
    spec = resolve_round_spec(workflow, round_index)
    paths = round_paths(args, spec, round_index)
    ensure_round_workspace(args, workflow, round_index, spec, paths, force=args.force, storage_mode=args.storage_mode)
    removal_config_path, inpaint_config_path = build_round_configs(paths, spec)

    base_meta = round_meta_base(args, round_index, spec, paths)
    meta = load_round_meta(paths, base_meta)
    meta.update(base_meta)
    meta["status"] = "workspace_ready"
    save_round_meta(paths, meta)

    removal_cmd = [
        args.python_bin,
        "edit_object_removal.py",
        "-s",
        str(Path(args.source_path).expanduser().resolve()),
        "-m",
        str(paths["workspace"]),
        "--config_file",
        str(removal_config_path),
        "--inpaint_config_file",
        str(inpaint_config_path),
        "--iteration",
        "0",
        "--storage_mode",
        args.storage_mode,
    ]
    if not args.render_intermediate:
        removal_cmd.extend(["--skip_train", "--skip_test"])
    if args.render_intermediate and args.render_video:
        removal_cmd.append("--render_video")
    run_cmd(removal_cmd)

    virtual_cmd = [
        args.python_bin,
        "tools/virtual_pose.py",
        "-s",
        str(Path(args.source_path).expanduser().resolve()),
        "-m",
        str(paths["workspace"]),
        "--config_file",
        str(removal_config_path),
        "--inpaint_config_file",
        str(inpaint_config_path),
        "--iteration",
        "0",
        "--round_dir",
        str(paths["round_dir"]),
        "--mask_provider_type",
        spec["mask_provider_type"],
        "--storage_mode",
        args.storage_mode,
    ]
    if spec.get("sync_segment_track_assets", True):
        virtual_cmd.append("--sync_segment_track_assets")
    else:
        virtual_cmd.append("--no-sync_segment_track_assets")
    run_cmd(virtual_cmd)
    meta = run_mask_provider(args, spec, paths, meta)
    save_round_meta(paths, meta)


def run_prepare_lama(args: argparse.Namespace, workflow: dict[str, Any], round_index: int) -> None:
    spec = resolve_round_spec(workflow, round_index)
    paths = round_paths(args, spec, round_index)
    base_meta = round_meta_base(args, round_index, spec, paths)
    meta = load_round_meta(paths, base_meta)
    target_tag = normalize_target_id(spec["target_id"])
    raw_mask_dir = get_raw_mask_from_sam2_dir(paths["workspace"], target_tag)
    if not raw_mask_dir.is_dir():
        raise FileNotFoundError(
            f"Initial inpaint masks not found. Place them under: {raw_mask_dir}"
        )

    prepare_cmd = [
        args.python_bin,
        "tools/prepare_lama_data.py",
        "-s",
        str(Path(args.source_path).expanduser().resolve()),
        "-m",
        str(paths["workspace"]),
        "-r",
        str(spec["resolution"]),
        "--iterations",
        "0",
        "--target_id",
        target_tag,
        "--inpaint2lama",
    ]
    if spec["lama_backend"] == "external":
        prepare_cmd.extend(
            [
                "--sync_lama_project",
                "--lama_data_name",
                spec["lama_data_name"],
                "--round_index",
                str(round_index),
            ]
        )
    run_cmd(prepare_cmd)

    lama_manifest = {
        "backend": spec["lama_backend"],
        "data_name": spec["lama_data_name"],
        "prepare_command": " ".join(prepare_cmd),
    }
    if spec["lama_backend"] == "external":
        lama_manifest["external_commands"] = [
            f"cd {PROJECT_ROOT / 'LaMa'}",
            f"python bin/predict_color.py --data_name {spec['lama_data_name']}",
            f"python bin/predict_depth.py --data_name {spec['lama_data_name']}",
        ]
        print("LaMa external commands:")
        for command in lama_manifest["external_commands"]:
            print(command)

    write_json(get_round_lama_bridge_manifest_path(paths["round_dir"]), lama_manifest)
    meta["status"] = "lama_inputs_ready"
    meta["lama"] = lama_manifest
    save_round_meta(paths, meta)


def run_simple_lama(args: argparse.Namespace, workflow: dict[str, Any], round_index: int) -> None:
    spec = resolve_round_spec(workflow, round_index)
    paths = round_paths(args, spec, round_index)
    target_tag = normalize_target_id(spec["target_id"])
    cmd = [
        args.python_bin,
        "tools/simple_lama_inpaint_virtual.py",
        "--model_path",
        str(paths["workspace"]),
        "--target_id",
        target_tag,
        "--mode",
        "both",
        "--device",
        args.simple_lama_device,
        "--mask_dilation",
        str(args.mask_dilation),
    ]
    run_cmd(cmd)
    meta = load_round_meta(paths, round_meta_base(args, round_index, spec, paths))
    meta["status"] = "lama_outputs_ready"
    meta.setdefault("lama", {})
    meta["lama"]["backend"] = "simple"
    meta["lama"]["simple_lama_command"] = " ".join(cmd)
    save_round_meta(paths, meta)


def run_finalize_round(args: argparse.Namespace, workflow: dict[str, Any], round_index: int) -> None:
    spec = resolve_round_spec(workflow, round_index)
    paths = round_paths(args, spec, round_index)
    base_meta = round_meta_base(args, round_index, spec, paths)
    meta = load_round_meta(paths, base_meta)
    target_tag = normalize_target_id(spec["target_id"])

    collect_cmd = [
        args.python_bin,
        "tools/prepare_lama_data.py",
        "-s",
        str(Path(args.source_path).expanduser().resolve()),
        "-m",
        str(paths["workspace"]),
        "-r",
        str(spec["resolution"]),
        "--iterations",
        "0",
        "--target_id",
        target_tag,
    ]
    if spec["lama_backend"] == "external":
        collect_cmd.extend(
            [
                "--sync_lama_project",
                "--lama_data_name",
                spec["lama_data_name"],
                "--round_index",
                str(round_index),
            ]
        )
    run_cmd(collect_cmd)

    removal_config_path = Path(meta["removal_config_path"])
    inpaint_config_path = Path(meta["inpaint_config_path"])
    fusion_cmd = [
        args.python_bin,
        "edit_object_removal_plyfusion.py",
        "-s",
        str(Path(args.source_path).expanduser().resolve()),
        "-m",
        str(paths["workspace"]),
        "--config_file",
        str(removal_config_path),
        "--iteration",
        "0",
        "--storage_mode",
        args.storage_mode,
    ]
    if spec.get("support_view_name"):
        fusion_cmd.extend(["--support_view_name", str(spec["support_view_name"])])
    run_cmd(fusion_cmd)

    inpaint_cmd = [
        args.python_bin,
        "edit_object_inpaint.py",
        "-s",
        str(Path(args.source_path).expanduser().resolve()),
        "-m",
        str(paths["workspace"]),
        "--config_file",
        str(inpaint_config_path),
        "--iteration",
        "0",
        "--resolution",
        str(spec["resolution"]),
        "--storage_mode",
        args.storage_mode,
    ]
    if not args.render_intermediate and not is_last_round(workflow, round_index):
        inpaint_cmd.extend(["--skip_train", "--skip_test", "--skip_inpaint_render"])
    if args.render_video and (args.render_intermediate or is_last_round(workflow, round_index)):
        inpaint_cmd.append("--render_video")
    if spec.get("support_view_name"):
        inpaint_cmd.extend(["--support_view_name", str(spec["support_view_name"])])
    run_cmd(inpaint_cmd)

    point_cloud_ply = (
        paths["workspace"]
        / "point_cloud_object_inpaint_virtual"
        / f"iteration_{spec['finetune_iteration']}"
        / "point_cloud.ply"
    )
    classifier_pth = paths["workspace"] / "point_cloud" / "iteration_0" / "classifier.pth"
    save_scene_snapshot(
        paths["scene_out"],
        point_cloud_ply,
        classifier_pth,
        cfg_args_path=paths["workspace"] / "cfg_args",
        state={
            "round_index": round_index,
            "target_id": spec["target_id"],
            "finetune_iteration": spec["finetune_iteration"],
            "workspace_model_path": str(paths["workspace"]),
            "storage_mode": args.storage_mode,
        },
        prefer_hardlink=args.storage_mode == "minimal",
    )

    meta["status"] = "completed"
    meta["scene_out_snapshot"] = str(paths["scene_out"])
    meta["final_point_cloud_ply"] = str(paths["scene_out"] / "point_cloud.ply" if args.storage_mode == "minimal" else point_cloud_ply)
    if args.storage_mode == "minimal":
        meta["storage_cleanup"] = cleanup_minimal_round(paths)
    save_round_meta(paths, meta)


def run_status(args: argparse.Namespace, workflow: dict[str, Any]) -> None:
    iterative_root = ensure_workflow_initialized(args, workflow)
    print(f"Iterative root: {iterative_root}")
    for round_index in range(len(workflow["rounds"])):
        spec = resolve_round_spec(workflow, round_index)
        round_dir = get_round_dir(iterative_root, round_index, spec["target_id"])
        meta_path = get_round_meta_path(round_dir)
        if meta_path.is_file():
            meta = read_json(meta_path)
            status = meta.get("status", "unknown")
        else:
            status = "not_initialized"
        print(
            f"[round {round_index:03d}] target={normalize_target_id(spec['target_id'])} "
            f"surrounding={spec['surrounding_ids']} status={status}"
        )


def main() -> None:
    args = parse_args()
    workflow = load_workflow_config(args.workflow_config)

    if args.command == "init":
        ensure_workflow_initialized(args, workflow)
        print(f"Initialized workflow root: {iterative_root_for(args.model_path)}")
        return

    if args.command == "status":
        run_status(args, workflow)
        return

    if args.round_index is None:
        raise ValueError(f"--round_index is required for command: {args.command}")

    if args.command == "prepare-round":
        run_prepare_round(args, workflow, args.round_index)
    elif args.command == "run-mask-provider":
        spec = resolve_round_spec(workflow, args.round_index)
        paths = round_paths(args, spec, args.round_index)
        meta = load_round_meta(paths, round_meta_base(args, args.round_index, spec, paths))
        meta = run_mask_provider(args, spec, paths, meta)
        save_round_meta(paths, meta)
    elif args.command == "prepare-lama":
        run_prepare_lama(args, workflow, args.round_index)
    elif args.command == "run-simple-lama":
        run_simple_lama(args, workflow, args.round_index)
    elif args.command == "finalize-round":
        run_finalize_round(args, workflow, args.round_index)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
