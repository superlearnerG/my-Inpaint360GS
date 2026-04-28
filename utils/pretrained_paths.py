from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_ROOT = PROJECT_ROOT.parent / "pretrained_models"


def pretrained_root() -> Path:
    return PRETRAINED_ROOT


def pretrained_path(*parts: str) -> Path:
    return PRETRAINED_ROOT.joinpath(*parts)


def require_pretrained_file(*parts: str) -> Path:
    path = pretrained_path(*parts)
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
    return path


def require_pretrained_dir(*parts: str) -> Path:
    path = pretrained_path(*parts)
    if not path.is_dir():
        raise FileNotFoundError(f"Pretrained model directory not found: {path}")
    return path


def torch_home() -> Path:
    return pretrained_path("torch")


def configure_pretrained_env(include_simple_lama: bool = False) -> None:
    os.environ["INPAINT360GS_PRETRAINED_ROOT"] = str(PRETRAINED_ROOT)
    os.environ["TORCH_HOME"] = str(torch_home())
    if include_simple_lama and "LAMA_MODEL" not in os.environ:
        os.environ["LAMA_MODEL"] = str(require_simple_lama_torchscript())


def require_simple_lama_torchscript() -> Path:
    candidates = [
        pretrained_path("big-lama", "big-lama.pt"),
        pretrained_path("torch", "hub", "checkpoints", "big-lama.pt"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "SimpleLaMa torchscript checkpoint not found. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def require_external_lama_dir() -> Path:
    path = require_pretrained_dir("big-lama")
    if not (path / "config.yaml").is_file():
        raise FileNotFoundError(f"LaMa config.yaml not found under: {path}")
    if not (path / "models" / "best.ckpt").is_file():
        raise FileNotFoundError(f"LaMa best.ckpt not found under: {path / 'models'}")
    return path


def segment_anything_checkpoint(model_type: str = "vit_b") -> Path:
    checkpoint_by_model = {
        "vit_b": "sam_vit_b_01ec64.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_h": "sam_vit_h_4b8939.pth",
    }
    if model_type not in checkpoint_by_model:
        raise ValueError(f"Unsupported SAM model_type: {model_type}")
    return require_pretrained_file("segment-anything", checkpoint_by_model[model_type])


def deaot_checkpoint(model_name: str = "r50_deaotl") -> Path:
    checkpoint_by_model = {
        "r50_deaotl": "R50_DeAOTL_PRE_YTB_DAV.pth",
        "deaotb": "DeAOTB_PRE_YTB_DAV.pth",
        "deaotl": "DeAOTL_PRE_YTB_DAV.pth",
    }
    if model_name not in checkpoint_by_model:
        raise ValueError(f"Unsupported DeAOT model name: {model_name}")
    return require_pretrained_file("deaot", checkpoint_by_model[model_name])


def groundingdino_checkpoint() -> Path:
    return require_pretrained_file("groundingdino", "groundingdino_swint_ogc.pth")


def ast_checkpoint() -> Path:
    return require_pretrained_file("audio-spectrogram-transformer", "audio_mdl.pth")


def cropformer_checkpoint() -> Path:
    return require_pretrained_file("entityseg", "CropFormer_hornet_3x_03823a.pth")
