# download aot-ckpt 
gdown  '1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ' -O ./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth

# download sam-ckpt
wget -P ./ckpt https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# download grounding-dino ckpt
wget -P ./ckpt https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth

# download mit-ast-finetuned ckpt
wget -O ./ast_master/pretrained_models/audio_mdl.pth https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1