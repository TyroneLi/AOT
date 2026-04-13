
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
# export SIGLIP_CACHE_DIR='/data/base_model/VIT/siglip-so400m-patch14-384'

# if you want to have a try with dynamic segment across temporal clips to perform inter-frame compression
# The default set is False.
# export DYNAMIC_SEGMENTS='True'
export INTER_COMPRESS='True'

# global token ration, local = 1 - global
export GLOBAL_RATIO=0.5
# intra or inter frame ota merge scale
export INTRA_SCALE=1.0
export INTER_SCALE=1.0



# # For mvbench and egoschema
# siglip visual token num, max: 729 <-> no intra compress
export VISUAL_TOKEN_NUM=126
# iter frame token keep ratio
export KEEP_RATIO=0.3



# # # For videomme and longvideo
# # siglip visual token num, max: 729 <-> no intra compress
# export VISUAL_TOKEN_NUM=108
# # iter frame token keep ratio
# export KEEP_RATIO=0.1

# # if perform evaluation on egoschema, need to export this:
# export EGOSCHEMA_UNPAD='True'


# longvideobench_val_v, mvbench, egoschema, videomme
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    accelerate launch --num_processes=8 --main_process_port=25000 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
    --tasks longvideobench_val_v \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_onevision \
    --output_path ./logs/ov-7b-OTA-Sinknorn/260323

