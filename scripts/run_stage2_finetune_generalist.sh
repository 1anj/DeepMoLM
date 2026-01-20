#!/bin/bash
# Stage 2: Visual Instruction Tuning
# Train StereoQFormer + LLM. Freeze Vision Tower.

PRETRAINED_ckpt=${1:-"./checkpoints/stage1-align/model"}
OUTPUT_DIR=${2:-"./checkpoints/stage2-finetune-generalist"}
DATA_MIXTURE=${3:-"3d_mol_mix"}
TEST_MIXTURE="3d_mol_test_mix"
GPUS_PER_NODE=${4:-8}

if [ -n "${5:-}" ]; then
    if [[ "$5" =~ ^[0-9]+$ ]]; then
        GPUS_PER_NODE="$5"
    else
        TEST_MIXTURE="$5"
    fi
fi
if [ -n "${6:-}" ]; then
    GPUS_PER_NODE="$6"
fi

MODEL_NAME=${MODEL_NAME:-"./checkpoints/Qwen2-VL-7B-Instruct"}
VISION_TOWER=${VISION_TOWER:-"./checkpoints/sam_clip_ckpt"}
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=1
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-"./eval_outputs"}

# Ensure PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.
export PYTHONPATH=$PYTHONPATH:.
export WANDB_MODE=offline
export WANDB_SERVICE_WAIT=1200
export WANDB_INIT_TIMEOUT=1200

echo "Starting Stage 2: Instruction Tuning..."
echo "Loading projector from: $PRETRAINED_ckpt"
echo "Output Directory: $OUTPUT_DIR"
echo "Data Mixture: $DATA_MIXTURE"
echo "Test Mixture: $TEST_MIXTURE"

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=29502 \
    scripts/train_molm.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PRETRAINED_ckpt \
    --chat_template qwen2 \
    --vision_tower $VISION_TOWER \
    --data_mixture $DATA_MIXTURE \
    --mm_vision_select_feature cls_patch \
    --mm_projector fusion \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio dynamic \
    --bf16 True \
    --output_dir $OUTPUT_DIR/model \
    --num_train_epochs 10 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --report_to wandb

echo "Starting evaluation..."
TEST_MIXTURE="$TEST_MIXTURE" python - <<'PY'
import os
from llava.data.builder import DATASETS, parse_mixture

names = parse_mixture(os.environ["TEST_MIXTURE"])
for name in names:
    cfg = DATASETS.get(name, {})
    data_path = cfg.get("data_path")
    if not data_path:
        continue
    print(f"{name}\t{data_path}")
PY \
| while IFS=$'\t' read -r name data_path; do
    output_file="${EVAL_OUTPUT_DIR}/${name}.jsonl"
    bash scripts/eval/run_eval.sh "$OUTPUT_DIR/model" "$data_path" "$output_file"
done
