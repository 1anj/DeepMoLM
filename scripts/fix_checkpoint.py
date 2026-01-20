import os
import sys
import argparse
import torch
import transformers
from transformers import AutoConfig

# Add current directory to path so we can import llava
sys.path.append(os.getcwd())

from llava.model.language_model.llava_llama import LlavaLlamaModel
from llava.model.language_model.llava_topdown_llama import LlavaTopDownLlamaModel

def fix_checkpoint(checkpoint_path):
    print(f"Fixing checkpoint at {checkpoint_path}...")
    
    checkpoint_path = os.path.abspath(checkpoint_path)
    # Load config to determine model class
    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    config.resume_path = checkpoint_path
    
    # Determine model class
    if hasattr(config, "architectures") and config.architectures:
        if "LlavaTopDownLlamaModel" in config.architectures:
            model_cls = LlavaTopDownLlamaModel
        elif "LlavaLlamaModel" in config.architectures:
            model_cls = LlavaLlamaModel
        else:
            # Fallback or try to eval
            try:
                model_cls = eval(config.architectures[0])
            except:
                print(f"Could not determine model class from {config.architectures}. Defaulting to LlavaLlamaModel.")
                model_cls = LlavaLlamaModel
    else:
        print("Architectures not found in config. Defaulting to LlavaLlamaModel.")
        model_cls = LlavaLlamaModel

    print(f"Loading model using {model_cls.__name__}...")
    try:
        # Load the model. This uses LlavaMetaModel.load_pretrained which handles the subfolders (llm/, mm_projector/)
        # We pass config object directly so it uses our resume_path
        model = model_cls.from_pretrained(config, device_map="cpu")
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Try without device_map if it failed (sometimes cpu offload issues?)
        print("Retrying with default settings...")
        model = model_cls.from_pretrained(config)

    print("Model loaded successfully. Saving combined weights to root...")
    
    # We want to use the standard PreTrainedModel.save_pretrained to save everything in one place
    # avoiding the custom LlavaMetaModel.save_pretrained which splits folders.
    transformers.PreTrainedModel.save_pretrained(model, checkpoint_path, safe_serialization=True)
    
    print(f"Fixed checkpoint saved to {checkpoint_path}. You should see model.safetensors now.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint directory (e.g. checkpoints/stage1-align/model/checkpoint-4200)")
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Path {args.checkpoint_path} does not exist.")
    else:
        fix_checkpoint(args.checkpoint_path)