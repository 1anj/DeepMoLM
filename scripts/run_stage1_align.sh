#!/bin/bash
# Stage 1: Molecule-Text Alignment
# Train only the StereoQFormer (projector). Freeze Vision Tower and LLM.

OUTPUT_DIR=${1:-"./checkpoints/stage1-align"}
DATA_MIXTURE=${3:-"3d_pretrain_mix"}
GPUS_PER_NODE=${4:-8}

MODEL_NAME=${MODEL_NAME:-"./checkpoints/Qwen2-VL-7B-Instruct"}
VISION_TOWER=${VISION_TOWER:-"./checkpoints/sam_clip_ckpt"}
PER_DEVICE_TRAIN_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=1

# Ensure PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.
export WANDB_MODE=offline
export WANDB_SERVICE_WAIT=1200
export WANDB_INIT_TIMEOUT=1200

echo "Starting Stage 1: Alignment..."
echo "Output Directory: $OUTPUT_DIR"
echo "Data Mixture: $DATA_MIXTURE"

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=29502 \
    scripts/train_molm.py \
    --model_name_or_path $MODEL_NAME \
    --chat_template qwen2 \
    --vision_tower $VISION_TOWER \
    --data_mixture $DATA_MIXTURE \
    --mm_vision_select_feature cls_patch \
    --mm_projector fusion \ 
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio dynamic \
    --bf16 True \
    --output_dir $OUTPUT_DIR/model \
    --num_train_epochs 5 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb 
