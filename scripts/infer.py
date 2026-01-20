#!/usr/bin/env python
"""
Single molecule inference script combining features from llava/cli/infer.py and scripts/eval/eval_molm.py
Supports molecule image and fingerprint-based inference for the DeepMoLM model.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from PIL import Image
from termcolor import colored

try:
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle, CONVERSATION_MODE_MAPPING
    from llava.data.mol_utils import Smiles2Img, get_3d_fingerprints
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, process_image
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from rdkit import Chem
    import numpy as np
except ImportError as e:
    print(colored(f"Error importing llava modules: {e}", "red"))
    print(colored("\nPlease ensure you have installed the DeepMoLM package and its dependencies:", "yellow"))
    print(colored("  pip install -e .", "cyan"))
    print(colored("Or install required dependencies from pyproject.toml", "yellow"))
    sys.exit(1)


def read_sdf_coords(sdf_path):
    """
    Read 3D coordinates and SMILES from an SDF file.
    
    Args:
        sdf_path: Path to the SDF file
    
    Returns:
        tuple: (coords, smiles) where coords is numpy.ndarray of shape (num_atoms, 3) 
               and smiles is a string, or (None, None) if reading fails
    """
    try:
        sdf_path = Path(sdf_path)
        if not sdf_path.exists():
            print(colored(f"Warning: SDF file not found: {sdf_path}", "yellow"))
            return None, None
        
        # Read the first molecule from the SDF file
        suppl = Chem.SDMolSupplier(str(sdf_path))
        mol = next(iter(suppl), None)
        
        if mol is None:
            print(colored(f"Warning: Could not read molecule from SDF file: {sdf_path}", "yellow"))
            return None, None
        
        # Get the conformer (3D structure)
        if mol.GetNumConformers() == 0:
            print(colored(f"Warning: No conformer found in SDF file: {sdf_path}", "yellow"))
            return None, None
        
        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()
        
        # Extract coordinates
        coords = np.zeros((num_atoms, 3), dtype=np.float32)
        for i in range(num_atoms):
            pos = conf.GetAtomPosition(i)
            coords[i] = [pos.x, pos.y, pos.z]
        
        # Try to get SMILES from the molecule
        smiles = Chem.MolToSmiles(mol)
        
        print(colored(f"Successfully loaded {num_atoms} atoms from SDF file", "green"))
        if smiles:
            print(colored(f"Extracted SMILES: {smiles}", "green"))
        
        return coords, smiles
        
    except Exception as e:
        print(colored(f"Error reading SDF file {sdf_path}: {e}", "red"))
        return None, None


def prepare_molecule_fp(smiles, coord_norm=None, molecule_fp=None, device='cuda'):
    """
    Prepare E3FP molecular fingerprint for inference.
    
    Args:
        smiles: SMILES string of the molecule
        coord_norm: Normalized 3D coordinates
        molecule_fp: Pre-computed molecule fingerprint (optional)
        device: Target device for the fingerprint tensor
    
    Returns:
        torch.Tensor: E3FP fingerprint or fallback padding
    """
    mol_fp = molecule_fp
    if mol_fp is None and smiles and coord_norm is not None:
        mol_fp = get_3d_fingerprints(smiles, coord_norm)
    
    # Return padding fallback if fingerprint is unavailable
    if mol_fp is None:
        # Minimal fallback: 1 atom with 4 levels, all padding
        return torch.full((1, 4), -1, dtype=torch.long, device=device)
    
    # Convert to tensor if not already
    if not isinstance(mol_fp, torch.Tensor):
        mol_fp = torch.as_tensor(mol_fp, dtype=torch.long)
    else:
        # Ensure correct dtype
        if mol_fp.dtype in [torch.float16, torch.float32, torch.float64]:
            # Validate integer-valued before conversion
            if not torch.all((mol_fp == mol_fp.floor()) | (mol_fp == -1)):
                print(colored("Warning: molecule_fp contains non-integer values, using fallback", "yellow"))
                return torch.full((1, 4), -1, dtype=torch.long, device=device)
            mol_fp = mol_fp.long()
        elif mol_fp.dtype != torch.long:
            mol_fp = mol_fp.long()
    
    # Validate shape: should be (num_atoms, 4) for E3FP with level=3
    if mol_fp.dim() == 1:
        # Single atom case: reshape to (1, num_levels)
        if mol_fp.numel() == 4:
            mol_fp = mol_fp.unsqueeze(0)  # (4,) -> (1, 4)
        else:
            print(colored(f"Warning: unexpected 1D molecule_fp shape {mol_fp.shape}, using fallback", "yellow"))
            return torch.full((1, 4), -1, dtype=torch.long, device=device)
    elif mol_fp.dim() == 2:
        # Expected case: (num_atoms, num_levels)
        if mol_fp.shape[-1] != 4:
            print(colored(f"Warning: molecule_fp has {mol_fp.shape[-1]} levels, expected 4, using fallback", "yellow"))
            return torch.full((1, 4), -1, dtype=torch.long, device=device)
    else:
        print(colored(f"Warning: unexpected molecule_fp dimensionality {mol_fp.dim()}, using fallback", "yellow"))
        return torch.full((1, 4), -1, dtype=torch.long, device=device)
    
    return mol_fp.to(device)


def prepare_image(smiles=None, image_path=None, image_size=1024):
    """
    Prepare molecule image from SMILES or image file.
    
    Args:
        smiles: SMILES string to render
        image_path: Path to existing image file
        image_size: Size for rendered SMILES image
    
    Returns:
        PIL.Image: Molecule image
    """
    image = None
    
    # Load from file if provided
    if image_path:
        img_path = Path(image_path)
        if img_path.exists():
            image = Image.open(img_path).convert("RGB")
        else:
            print(colored(f"Warning: Image file not found: {image_path}", "yellow"))
    
    # Generate from SMILES if no image loaded
    if image is None and smiles:
        image = Smiles2Img(smiles, size=image_size)
    
    # Fallback to blank image
    if image is None:
        image = Image.new('RGB', (image_size, image_size))
    
    return image


def infer_model(args):
    """Main inference function."""
    # Initialize model
    print(colored("Loading model...", "green"))
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
    
    print(colored(f"Model loaded: {model_name}", "green"))
    
    # Prepare inputs
    print(colored("Preparing inputs...", "green"))
    
    # 0. Read SDF file first if provided (to get both coords and SMILES)
    coord_norm = None
    smiles_from_sdf = None
    
    if args.structure:
        coord_norm, smiles_from_sdf = read_sdf_coords(args.structure)
    elif args.coord_norm_file:
        with open(args.coord_norm_file, 'r') as f:
            coord_norm = json.load(f)
    
    # Use SMILES from SDF if not provided via --smiles argument
    smiles_to_use = args.smiles if args.smiles else smiles_from_sdf
    
    # 1. Prepare image
    image = prepare_image(
        smiles=smiles_to_use,
        image_path=args.image,
        image_size=args.image_size
    )
    
    # Wrap data_args for process_image
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
        processed_image = {k: (v.to(model.device) if torch.is_tensor(v) else v) 
                          for k, v in processed_image.items()}
    else:
        processed_image = (processed_image.to(model.device) 
                          if torch.is_tensor(processed_image) else processed_image)
    
    # 2. Prepare molecule fingerprint
    mol_fp_data = None
    if args.molecule_fp_file:
        with open(args.molecule_fp_file, 'r') as f:
            mol_fp_data = json.load(f)
    
    mol_fp = prepare_molecule_fp(
        smiles=smiles_to_use,
        coord_norm=coord_norm,
        molecule_fp=mol_fp_data,
        device=model.device
    )
    
    # 3. Prepare prompt
    prompt_text = args.text
    if "{}" in prompt_text and smiles_to_use:
        prompt_text = prompt_text.format(smiles_to_use)
    
    # Resolve "auto" conversation mode to concrete template
    conv_mode = args.conv_mode
    if conv_mode == "auto":
        # Check model name/path against the mapping
        resolved_mode = None
        model_name_lower = model_name.lower()
        model_path_lower = model_path.lower()
        
        for key, value in CONVERSATION_MODE_MAPPING.items():
            if key in model_name_lower or key in model_path_lower:
                resolved_mode = value
                break
        
        # Default to qwen2 if no match found (common for DeepMoLM models)
        if resolved_mode is None:
            resolved_mode = "qwen2"
            print(colored(f"No conversation mode mapping found for '{model_name}', defaulting to '{resolved_mode}'", "yellow"))
        else:
            print(colored(f"Auto-detected conversation mode '{resolved_mode}' for model '{model_name}'", "green"))
        
        conv_mode = resolved_mode
    
    conv = conv_templates[conv_mode].copy()
    qs = f"{DEFAULT_IMAGE_TOKEN}\n{prompt_text}"
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Tokenize
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(model.device)
    
    # Generate response
    print(colored("Generating response...", "green"))
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=processed_image,
            molecule_fp=mol_fp,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True
        )
    
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    # Print result
    print(colored("\n" + "="*80, "blue"))
    print(colored("Response:", "cyan", attrs=["bold"]))
    print(colored(output, "cyan"))
    print(colored("="*80 + "\n", "blue"))
    
    # Save to file if requested
    if args.output_file:
        output_path = os.path.expanduser(args.output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        result = {
            "smiles": smiles_to_use,
            "prompt": prompt_text,
            "response": output,
            "model_path": args.model_path,
        }
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(colored(f"Result saved to: {output_path}", "green"))
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Single molecule inference for DeepMoLM")
    
    # Model arguments
    parser.add_argument("--model-path", "-m", type=str, required=True,
                       help="Path to the pretrained model")
    parser.add_argument("--model-base", type=str, default=None,
                       help="Base model path for LoRA models")
    parser.add_argument("--load-8bit", action="store_true",
                       help="Load model in 8-bit (saves VRAM)")
    parser.add_argument("--load-4bit", action="store_true",
                       help="Load model in 4-bit (more VRAM savings)")
    parser.add_argument("--device-map", type=str, default="auto",
                       help="Device map for accelerate/transformers")
    
    # Input arguments
    parser.add_argument("--smiles", type=str, default=None,
                       help="SMILES string of the molecule")
    parser.add_argument("--image", type=str, default=None,
                       help="Path to molecule image file")
    parser.add_argument("--structure", type=str, default=None,
                       help="Path to molecule structure file (SDF format) for 3D coordinates")
    parser.add_argument("--molecule-fp-file", type=str, default=None,
                       help="Path to JSON file containing pre-computed molecule_fp")
    parser.add_argument("--coord-norm-file", type=str, default=None,
                       help="Path to JSON file containing normalized 3D coordinates")
    parser.add_argument("--text", "-t", type=str, default="Describe the input molecule.",
                       help="Input prompt text (use {} as SMILES placeholder)")
    
    # Generation arguments
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=None,
                       help="Nucleus sampling top-p")
    parser.add_argument("--num-beams", type=int, default=1,
                       help="Number of beams for beam search")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                       help="Maximum number of tokens to generate")
    
    # Configuration arguments
    parser.add_argument("--conv-mode", "-c", type=str, default="qwen2",
                       help="Conversation template mode")
    parser.add_argument("--image-size", type=int, default=1024,
                       help="Size for rendered SMILES images")
    parser.add_argument("--image-aspect-ratio", type=str, default="dynamic",
                       help="Image aspect ratio mode (dynamic, pad, square)")
    
    # Output arguments
    parser.add_argument("--output-file", "-o", type=str, default=None,
                       help="Path to save the output JSON result")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.smiles and not args.image and not args.structure:
        parser.error("At least one of --smiles, --image, or --structure must be provided")
    
    infer_model(args)


if __name__ == "__main__":
    main()
