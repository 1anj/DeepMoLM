"""
Distributed evaluation script for Qwen2-VL baseline.
Usage:
    torchrun --nproc_per_node=NUM_GPUS scripts/eval/eval_qwen_distributed.py \
        --model-path "./checkpoints/Qwen2-VL-7B-Instruct" \
        --data-path "$data_path" \
        --question-file "$question_file" \
        --answers-file "$output_file" \
        --temperature 0
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info



def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # Single GPU fallback
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl")
    
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def get_data_shard(questions, rank, world_size):
    """Split questions across ranks for distributed evaluation."""
    shard_size = len(questions) // world_size
    remainder = len(questions) % world_size
    
    # Distribute remainder evenly
    start_idx = rank * shard_size + min(rank, remainder)
    end_idx = start_idx + shard_size + (1 if rank < remainder else 0)
    
    return questions[start_idx:end_idx], start_idx


def eval_model(args):
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    
    if is_main:
        print(f"Distributed evaluation: {world_size} GPUs")
        print(f"Loading Qwen2-VL from: {args.model_path}")
    
    # Load Qwen2-VL model and processor
    model_path = os.path.expanduser(args.model_path)
    
    # Set device map for distributed
    device_map = {"": local_rank}
    device = torch.device(f"cuda:{local_rank}")
    
    # Load model
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            quantization_config=quantization_config
        )
    elif args.load_8bit:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            load_in_8bit=True
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_path)
    
    if is_main:
        print(f"Model loaded successfully on {world_size} GPU(s)")

    # Data
    data_root = Path(args.data_path).resolve()
    image_root = data_root / "images"
    if not image_root.exists():
        image_root = Path(args.question_file).resolve().parent / "images"

    # Load questions/tasks
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    if isinstance(questions, dict) and 'train' in questions: 
        questions = questions['train']
    elif isinstance(questions, dict) and 'test' in questions:
        questions = questions['test']

    total_questions = len(questions)
    
    # Shard data across ranks
    questions_shard, start_idx = get_data_shard(questions, rank, world_size)
    
    if is_main:
        print(f"Total questions: {total_questions}")
    print(f"[Rank {rank}] Processing {len(questions_shard)} questions (indices {start_idx} to {start_idx + len(questions_shard) - 1})")

    # Output file (each rank writes to a separate file)
    answers_file = os.path.expanduser(args.answers_file)
    base, ext = os.path.splitext(answers_file)
    shard_file = f"{base}_rank{rank}{ext}"
    os.makedirs(os.path.dirname(shard_file), exist_ok=True)
    ans_file = open(shard_file, "w")

    # Progress bar only on main rank
    iterator = tqdm(enumerate(questions_shard), total=len(questions_shard), disable=not is_main)

    for local_idx, line in iterator:
        global_idx = start_idx + local_idx
        
        # Extract task information
        task_type = line.get('task', 'Caption')
        target = line.get('enriched_output') or line.get('output') or line.get('smiles', '')
        instruction = line.get('instruction')
        smiles = line.get('smiles')

        # Prepare Image
        image = None
        image_name = line.get("image")
        if image_name:
            image_path = Path(image_name)
            if not image_path.is_absolute():
                image_path = image_root / image_path
            if not image_path.exists():
                alt_path = Path(args.question_file).resolve().parent / image_name
                if alt_path.exists():
                    image_path = alt_path
            if image_path.exists():
                image = Image.open(image_path).convert("RGB")
        
   
        # Prepare prompt
        prompt_template = instruction or args.prompt_template
        prompt_text = prompt_template.format(smiles) if "{}" in prompt_template else prompt_template
        
        # Construct Qwen2-VL messages format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": f"SMILES: {smiles}\n" + prompt_text
                    }
                ]
            }
        ]
        
        # Prepare inputs for the model
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(device)
        
        # Generate output
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                num_beams=args.num_beams,
            )
        
        # Trim generated_ids to remove input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode output
        outputs = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        # Write result
        res = {
            "prediction": [outputs],
            "target": [target],
            "task_type": [task_type],
            "global_idx": global_idx  # For ordering when merging
        }
        ans_file.write(json.dumps(res) + "\n")
        ans_file.flush()
    
    ans_file.close()
    
    if is_main:
        print(f"\n[Rank {rank}] Evaluation complete. Results written to {shard_file}")
        print(f"To merge results, run:")
        print(f"  python scripts/eval/merge_results.py --input-pattern '{base}_rank*{ext}' --output '{answers_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/Qwen2-VL-7B-Instruct")
    parser.add_argument("--data-path", type=str, default="data/3d-mol-dataset")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prompt_template", type=str, default="Describe the input molecule.")
    parser.add_argument("--load-8bit", action="store_true", help="Load the model in 8-bit (saves VRAM).")
    parser.add_argument("--load-4bit", action="store_true", help="Load the model in 4-bit (more VRAM savings).")
    parser.add_argument("--image-size", type=int, default=1024, help="Rendered SMILES image size (reduce to save VRAM).")
    args = parser.parse_args()
    eval_model(args)
