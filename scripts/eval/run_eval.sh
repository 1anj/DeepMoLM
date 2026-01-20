#!/bin/bash
MODEL_PATH=$1
DATA_ARG=$2
OUTPUT_ARG=$3
EXTRA_ARGS="${@:4}"
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: bash scripts/eval/run_eval.sh [MODEL_PATH] [DATA_PATH|MIXTURE] [OUTPUT_FILE|OUTPUT_DIR] [EXTRA_ARGS...]"
    echo "Example (single dataset): bash scripts/eval/run_eval.sh ./checkpoints/deepmolm-7b ./data/3d-mol-dataset ./eval_outputs/predictions.txt --load-4bit"
    echo "Example (mixture): bash scripts/eval/run_eval.sh ./checkpoints/deepmolm-7b 3d_mol_valid_mix ./eval_outputs --load-4bit"
    exit 1
fi

if [ -z "$DATA_ARG" ]; then
    DATA_ARG="3d_mol_valid_mix"
fi

if [ -e "$DATA_ARG" ]; then
    DATA_PATH="$DATA_ARG"
    OUTPUT_FILE="$OUTPUT_ARG"
    if [ -z "$OUTPUT_FILE" ]; then
        OUTPUT_FILE="./eval_outputs/predictions.txt"
    fi

    if [ -f "$DATA_PATH/validation.json" ]; then
        QUESTION_FILE="$DATA_PATH/validation.json"
    elif [ -f "$DATA_PATH/valid.json" ]; then
        QUESTION_FILE="$DATA_PATH/valid.json"
    elif [ -f "$DATA_PATH/test.json" ]; then
        QUESTION_FILE="$DATA_PATH/test.json"
    elif [ -f "$DATA_PATH/train.json" ]; then
        QUESTION_FILE="$DATA_PATH/train.json"
    else
        echo "Error: Could not find validation.json/test.json/train.json in $DATA_PATH"
        exit 1
    fi

    echo "Running evaluation..."
    echo "Model: $MODEL_PATH"
    echo "Data: $DATA_PATH ($QUESTION_FILE)"
    echo "Output: $OUTPUT_FILE"

    torchrun --nproc_per_node=$GPUS_PER_NODE \
        scripts/eval/eval_molm_distributed.py \
        --model-path $MODEL_PATH \
        --data-path $DATA_PATH \
        --question-file $QUESTION_FILE \
        --answers-file $OUTPUT_FILE \
        --temperature 0 \
        $EXTRA_ARGS

    echo "Evaluation complete."
    echo "Results saved to: $OUTPUT_FILE"
    echo "To compute metrics, assume you have 3D-MoLM reference code:"
    echo "python reference/3D-MoLM/read_generalist_results.py --file_path $OUTPUT_FILE --tokenizer_path $MODEL_PATH"
    exit 0
fi

MIXTURE_NAME="$DATA_ARG"
OUTPUT_DIR="$OUTPUT_ARG"
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="./eval_outputs"
fi
mkdir -p "$OUTPUT_DIR"

echo "Running mixture evaluation..."
echo "Model: $MODEL_PATH"
echo "Mixture: $MIXTURE_NAME"
echo "Output dir: $OUTPUT_DIR"

MIXTURE_NAME="$MIXTURE_NAME" python - <<'PY'
import os
from llava.data.builder import DATASETS, parse_mixture

names = parse_mixture(os.environ["MIXTURE_NAME"])
for name in names:
    cfg = DATASETS.get(name, {})
    data_path = cfg.get("data_path")
    if not data_path:
        continue
    print(f"{name}\t{data_path}")
PY \
| while IFS=$'\t' read -r name data_path; do
    if [ -z "$data_path" ]; then
        continue
    fi
    if [ ! -d "$data_path" ]; then
        echo "Error: dataset path not found: $data_path"
        exit 1
    fi
    if [ -f "$data_path/validation.json" ]; then
        question_file="$data_path/validation.json"
    elif [ -f "$data_path/valid.json" ]; then
        question_file="$data_path/valid.json"
    elif [ -f "$data_path/test.json" ]; then
        question_file="$data_path/test.json"
    elif [ -f "$data_path/train.json" ]; then
        question_file="$data_path/train.json"
    else
        echo "Error: no split json found in $data_path"
        exit 1
    fi
    output_file="${OUTPUT_DIR}/${name}.jsonl"
    torchrun --nproc_per_node=$GPUS_PER_NODE \
        scripts/eval/eval_molm_distributed.py \
        --model-path "$MODEL_PATH" \
        --data-path "$data_path" \
        --question-file "$question_file" \
        --answers-file "$output_file" \
        --temperature 0 \
        $EXTRA_ARGS
done
