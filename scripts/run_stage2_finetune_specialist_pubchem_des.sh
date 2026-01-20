#!/bin/bash
# Stage 2 Specialist: PubChem Description
# Fine-tune on a single dataset and evaluate on test/valid.

SPECIALIST_NAME="pubchem_des"
TRAIN_DATASET="e3fp_pubchem_des_train"
TEST_DATASET="e3fp_pubchem_des_test"
VALID_DATASET="e3fp_pubchem_des_valid"

PRETRAINED_ckpt=${1:-"./checkpoints/stage1-align/model"}
OUTPUT_DIR=${2:-"./checkpoints/stage2-finetune-specialist-${SPECIALIST_NAME}"}
GPUS_PER_NODE=${3:-8}
EXTRA_EVAL_ARGS=("${@:4}")

MODEL_NAME=${MODEL_NAME:-"./checkpoints/Qwen2-VL-7B-Instruct"}
VISION_TOWER=${VISION_TOWER:-"./checkpoints/sam_clip_ckpt"}
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=1
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-"./eval_outputs/specialist_${SPECIALIST_NAME}"}

# Ensure PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.
export WANDB_MODE=offline
export WANDB_SERVICE_WAIT=1200
export WANDB_INIT_TIMEOUT=1200

resolve_data_path() {
    DATASET_NAME="$1" python - <<'PY'
import os
from llava.data.builder import DATASETS

name = os.environ.get("DATASET_NAME", "")
print(DATASETS.get(name, {}).get("data_path", ""))
PY
}

run_eval_split() {
    local dataset_name="$1"
    local split="$2"
    local output_dir="$3"
    local data_path
    local question_file

    data_path=$(resolve_data_path "$dataset_name")
    if [ -z "$data_path" ]; then
        echo "Error: dataset path not found for $dataset_name"
        exit 1
    fi
    if [ ! -d "$data_path" ]; then
        echo "Error: dataset path not found: $data_path"
        exit 1
    fi

    if [ "$split" = "test" ]; then
        question_file="$data_path/test.json"
        if [ ! -f "$question_file" ]; then
            echo "Error: test.json not found in $data_path"
            exit 1
        fi
    else
        if [ -f "$data_path/validation.json" ]; then
            question_file="$data_path/validation.json"
        elif [ -f "$data_path/valid.json" ]; then
            question_file="$data_path/valid.json"
        else
            echo "Error: validation.json or valid.json not found in $data_path"
            exit 1
        fi
    fi

    mkdir -p "$output_dir"
    local output_file="$output_dir/${dataset_name}.jsonl"

    echo "Running evaluation ($split)..."
    echo "Model: $OUTPUT_DIR/model"
    echo "Data: $data_path ($question_file)"
    echo "Output: $output_file"

    torchrun --nproc_per_node=$GPUS_PER_NODE \
        scripts/eval/eval_molm_distributed.py \
        --model-path "$OUTPUT_DIR/model" \
        --data-path "$data_path" \
        --question-file "$question_file" \
        --answers-file "$output_file" \
        --temperature 0 \
        "${EXTRA_EVAL_ARGS[@]}"
}

echo "Starting Stage 2 Specialist: $SPECIALIST_NAME"
echo "Loading projector from: $PRETRAINED_ckpt"
echo "Output Directory: $OUTPUT_DIR"
echo "Train Dataset: $TRAIN_DATASET"

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=29502 \
    scripts/train_molm.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PRETRAINED_ckpt \
    --chat_template qwen2 \
    --vision_tower $VISION_TOWER \
    --data_mixture $TRAIN_DATASET \
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
    --save_steps 100 \
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

echo "Starting evaluation (test)..."
run_eval_split "$TEST_DATASET" "test" "$EVAL_OUTPUT_DIR/test"
# run_eval_split "$VALID_DATASET" "valid" "$EVAL_OUTPUT_DIR/valid"
