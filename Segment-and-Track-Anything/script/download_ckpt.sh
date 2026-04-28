#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PRETRAINED_ROOT="${PROJECT_ROOT}/../pretrained_models"

mkdir -p \
  "${PRETRAINED_ROOT}/deaot" \
  "${PRETRAINED_ROOT}/segment-anything" \
  "${PRETRAINED_ROOT}/groundingdino" \
  "${PRETRAINED_ROOT}/audio-spectrogram-transformer"

# download aot-ckpt
gdown '1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ' -O "${PRETRAINED_ROOT}/deaot/R50_DeAOTL_PRE_YTB_DAV.pth"

# download sam-ckpt
wget -O "${PRETRAINED_ROOT}/segment-anything/sam_vit_b_01ec64.pth" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# download grounding-dino ckpt
wget -O "${PRETRAINED_ROOT}/groundingdino/groundingdino_swint_ogc.pth" https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth

# download mit-ast-finetuned ckpt
wget -O "${PRETRAINED_ROOT}/audio-spectrogram-transformer/audio_mdl.pth" https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1
