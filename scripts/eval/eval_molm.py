import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.data.mol_utils import Smiles2Img, get_3d_fingerprints
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, process_image
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if args.load_4bit and args.load_8bit:
        raise ValueError("Only one of --load-4bit or --load-8bit can be set.")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_name=model_name,
        model_base=args.model_base,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device_map=args.device_map,
    )
    if not (args.load_4bit or args.load_8bit):
        model_dtype = getattr(model.config, "model_dtype", "")
        if isinstance(model_dtype, str) and "bfloat16" in model_dtype:
            model = model.to(dtype=torch.bfloat16)

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

    # Output file
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    def prepare_molecule_fp(line, smiles):
        """
        Prepare E3FP molecular fingerprint for evaluation.
        
        E3FP fingerprints are categorical hash indices (integers), not continuous values.
        Format: (num_atoms, level+1) where each value is a hash index [0, mol_fp_bits-1] or -1 for padding.
        """
        mol_fp = line.get("molecule_fp")
        if mol_fp is None and smiles and line.get("coord_norm") is not None:
            mol_fp = get_3d_fingerprints(smiles, line["coord_norm"])
        
        # Return padding fallback if fingerprint is unavailable
        if mol_fp is None:
            # Minimal fallback: 1 atom with 4 levels, all padding
            return torch.full((1, 4), -1, dtype=torch.long, device=model.device)
        
        # Convert to tensor if not already (get_3d_fingerprints returns torch.long)
        if not isinstance(mol_fp, torch.Tensor):
            mol_fp = torch.as_tensor(mol_fp, dtype=torch.long)
        else:
            # Ensure correct dtype (should already be long from get_3d_fingerprints)
            if mol_fp.dtype in [torch.float16, torch.float32, torch.float64]:
                # Validate integer-valued before conversion
                if not torch.all((mol_fp == mol_fp.floor()) | (mol_fp == -1)):
                    print(f"Warning: molecule_fp contains non-integer values, using fallback")
                    return torch.full((1, 4), -1, dtype=torch.long, device=model.device)
                mol_fp = mol_fp.long()
            elif mol_fp.dtype != torch.long:
                mol_fp = mol_fp.long()
        
        # Validate shape: should be (num_atoms, 4) for E3FP with level=3
        if mol_fp.dim() == 1:
            # Single atom case: reshape to (1, num_levels)
            if mol_fp.numel() == 4:
                mol_fp = mol_fp.unsqueeze(0)  # (4,) -> (1, 4)
            else:
                print(f"Warning: unexpected 1D molecule_fp shape {mol_fp.shape}, using fallback")
                return torch.full((1, 4), -1, dtype=torch.long, device=model.device)
        elif mol_fp.dim() == 2:
            # Expected case: (num_atoms, num_levels)
            if mol_fp.shape[-1] != 4:
                print(f"Warning: molecule_fp has {mol_fp.shape[-1]} levels, expected 4, using fallback")
                return torch.full((1, 4), -1, dtype=torch.long, device=model.device)
        else:
            print(f"Warning: unexpected molecule_fp dimensionality {mol_fp.dim()}, using fallback")
            return torch.full((1, 4), -1, dtype=torch.long, device=model.device)
        
        return mol_fp.to(model.device)

    for line in tqdm(questions):
        # 3D-MoLM format: {"instruction":..., "output":..., "task":...}
        task_type = line.get('task', 'Caption')
        target = line.get('enriched_output') or line.get('output') or line.get('smiles', '')
        instruction = line.get('instruction')
        smiles = line.get('smiles')

        # Prepare Inputs
        # 1. Image
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
        if image is None and smiles:
            image = Smiles2Img(smiles, size=args.image_size)
        if image is None:
            image = Image.new('RGB', (args.image_size, args.image_size))
        
        # Wrap data_args to pass to process_image
        class DataArgs:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        
        data_args = DataArgs(
            image_processor=image_processor,
            image_aspect_ratio=args.image_aspect_ratio,
            image_grid_pinpoints=None
        )
        
        processed_image = process_image(image, data_args, None)
        
        if isinstance(processed_image, dict):
            processed_image = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in processed_image.items()}
        else:
            processed_image = processed_image.to(model.device) if torch.is_tensor(processed_image) else processed_image

        # 2. molecule_fp
        mol_fp = prepare_molecule_fp(line, smiles)

        # 3. Prompt Construction
        prompt_template = instruction or args.prompt_template
        prompt_text = prompt_template.format(smiles) if "{}" in prompt_template else prompt_template
        
        conv = conv_templates[args.conv_mode].copy()
        qs = f"{DEFAULT_IMAGE_TOKEN}\n{prompt_text}"
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=processed_image,
                molecule_fp=mol_fp,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Write result with format: {"prediction": [...], "target": [...], "task_type": [...]}
        res = {
            "prediction": [outputs],
            "target": [target],
            "task_type": [task_type]
        }
        ans_file.write(json.dumps(res) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/deepmolm")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="data/3d-mol-dataset")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prompt_template", type=str, default="Describe the input molecule.")
    parser.add_argument("--load-8bit", action="store_true", help="Load the model in 8-bit (saves VRAM).")
    parser.add_argument("--load-4bit", action="store_true", help="Load the model in 4-bit (more VRAM savings).")
    parser.add_argument("--device-map", type=str, default="auto", help="Device map for accelerate/transformers.")
    parser.add_argument("--conv-mode", type=str, default="qwen2", help="Conversation mode (template).")
    parser.add_argument("--image-size", type=int, default=1024, help="Rendered SMILES image size (reduce to save VRAM).")
    parser.add_argument("--image-aspect-ratio", type=str, default="dynamic", help="Image aspect ratio mode (dynamic, pad, square).")
    args = parser.parse_args()
    eval_model(args)
