
# "temperature=1.0,top_p=1.0"


# export HF_HOME=~/.cache/huggingface
# export HF_HUB_CACHE=$HF_HOME/hub
# export TRANSFORMERS_CACHE=$HF_HOME/transformers
# export HF_DATASETS_CACHE=$HF_HOME/datasets

# # force offline to excute the evaluation if the machine cannot access network
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1


# local siglip checkpoints path, leave blank here to be fullfilled by downloading from huggingface and putting on specific path
export SIGLIP_CACHE_DIR=''


# Due to the issue raised by VID that the sampling frames cannot meet the specified number,
# inter compress need to be disabled.
# export DYNAMIC_SEGMENTS='True'
# export INTER_COMPRESS='True'

# global token ration, local = 1 - global
export GLOBAL_RATIO=0.5
# intra or inter frame ota merge scale
export INTRA_SCALE=1.0
export INTER_SCALE=1.0

# siglip visual token num, 729 without inter-frame token reduction.
export VISUAL_TOKEN_NUM=180
# iter frame token keep ratio
export KEEP_RATIO=0.4


# longvideobench_val_v, mvbench, egoschema, videomme
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    accelerate launch --num_processes=8 --main_process_port=25000 \
    -m lmms_eval \
    --model llava_vid \
    --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,mm_spatial_pool_mode=average,max_frames_num=64 \
    --tasks mvbench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_vid \
    --output_path ./logs/vid-7b-OTA-Sinknorn/260323


