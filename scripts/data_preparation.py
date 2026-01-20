import os
import io
import sys
import glob
import json
import tqdm
import argparse
import pickle
import numpy as np
import multiprocessing
import random
import torch
import shortuuid
from pathlib import Path
from functools import partial
from datasets import load_dataset
# Add project root to path
sys.path.insert(0, os.getcwd())

# Import RDKit
from rdkit import Chem
from rdkit.Chem import Draw
import logging
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("prepare_mol")

# ----------------- Helper Functions ----------------- #

def Smiles2Img(smis: str, size: int = 1024, savePath=None):
    """Render a SMILES string to a PIL image."""
    try:
        mol = Chem.MolFromSmiles(smis)
        if mol is None: return None
        img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(size, size))
        if savePath is not None:
            img.save(savePath)
        return img
    except Exception:
        return None

def get_value(data, keys, default=None):
    for k in keys:
        if k in data:
            return data[k]
    return default

# ----------------- Worker Process ----------------- #

worker_output_dir = None

def init_worker(output_dir_path):
    global worker_output_dir
    worker_output_dir = Path(output_dir_path)

def process_item_worker(item):
    """
    Worker function to process a single item.
    Saves image to disk and returns metadata dict.
    """
    try:
        # Extract SMILES
        smiles = get_value(item, ["smiles", "smi", "canonical_smiles", "SMILES"])
        
        if not smiles:
            return None

        # Generate unique ID
        # unique_id = shortuuid.uuid()
        unique_id = str(item['global_idx'])
        
        # Generate Image
        img_filename = f"{unique_id}.png"
        img_path = worker_output_dir / img_filename
        
        try:
            # We check if image already exists? No, mostly new. 
            # Or assume unrelated run.
            img = Smiles2Img(smiles, size=1024)
            if img is not None:
                img.save(img_path)
            else:
                return None # parsing failed
        except Exception as e:
            # logger.warning(f"Failed to generate image for {smiles}: {e}")
            return None
        
        # Construct Result
        # We start with a copy of the item to preserve original data
        result_data = item.copy()
        
        # Add LLaVA-compat fields
        result_data['id'] = unique_id
        result_data['image'] = img_filename
        
        # Clean up some types for JSON serialization if needed
        # (Datasets often returns native python types, but let's be safe if numpy)
        for k, v in result_data.items():
            if isinstance(v, (np.ndarray, np.generic)):
                result_data[k] = v.tolist()
            if isinstance(v, bytes):
                 # Skip bytes fields or decode? 
                 # Often better to remove raw bytes if not needed
                 try:
                     result_data[k] = v.decode('utf-8')
                 except:
                     del result_data[k]
        
        return result_data
        
    except Exception as e:
        # logger.error(f"Error processing item: {e}")
        return None

# ----------------- Main ----------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Root directory to search for Parquet files")
    parser.add_argument("--dataset_name", type=str, default="mol_dataset", help="Name of the output dataset directory")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Base directory to save the output")
    parser.add_argument("--num_workers", type=int, default=16) 
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process per file (for debugging)")
    args = parser.parse_args()

    root = Path(args.data_path).resolve()
    base_output = Path(args.output_dir).resolve()
    dataset_dir = base_output / args.dataset_name
    images_dir = dataset_dir / "images"
    
    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning {root} for parquet files...")
    parquet_files = list(root.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files.")
    
    if not parquet_files:
        print("No parquet files found. Exiting.")
        return

    # Data Accumulators
    split_data = {
        'train': [],
        'validation': [],
        'test': []
    }
    
    pool = multiprocessing.Pool(processes=args.num_workers, initializer=init_worker, initargs=(str(images_dir),))
    
    total_processed = 0
    
    for p_file in parquet_files:
        print(f"Processing {p_file}...")
        try:
            ds_loaded = load_dataset("parquet", data_files=str(p_file))
            
            # Infer splits
            if type(ds_loaded).__name__ == 'DatasetDict':
                ds_keys = list(ds_loaded.keys())
                ds_dict = ds_loaded
                
                # Heuristic override if defaulting to 'train'
                if len(ds_keys) == 1 and ds_keys[0] == 'train':
                    filename_lower = p_file.name.lower()
                    if 'val' in filename_lower:
                        real_split = 'validation'
                    elif 'test' in filename_lower:
                        real_split = 'test'
                    else:
                        real_split = 'train'
                    
                    if real_split != 'train':
                        ds_dict = {real_split: ds_loaded['train']}
                        ds_keys = [real_split]
            else:
                filename_lower = p_file.name.lower()
                if 'train' in filename_lower:
                     inferred_split = 'train'
                elif 'test' in filename_lower:
                    inferred_split = 'test'
                elif 'val' in filename_lower:
                    inferred_split = 'validation'
                else:
                    inferred_split = 'train'
                
                ds_dict = {inferred_split: ds_loaded}
                ds_keys = [inferred_split]
            
            for split_name in ds_keys:
                # Map to standard split names
                target_split = split_name
                if 'val' in split_name: target_split = 'validation'
                
                # Fallback for unknown splits? default to train
                if target_split not in split_data:
                    target_split = 'train'
                    
                print(f"  - Processing split: {split_name} -> {target_split}")
                ds = ds_dict[split_name]
                
                if args.max_samples:
                    ds = ds.select(range(min(len(ds), args.max_samples)))
                
                print(f"  - Loaded {len(ds)} rows.")
                
                # Inject split info
                # We do this in main process before sending to worker, effectively simple dict update
                # Actually, worker doesn't need split info unless we want it in JSON. 
                # Yes, we want it in JSON.
                
                # Inject split info and global index
                # Generator to avoid loading all into memory
                def item_generator_with_idx(dataset, start_idx, split):
                    for i, item in enumerate(dataset):
                        yield {**item, 'split': split, 'global_idx': start_idx + i}

                items = item_generator_with_idx(ds, total_processed, target_split)
                
                # Update total_processed validation for next batch
                current_batch_len = len(ds)
                
                # Process in parallel
                for i, result in enumerate(tqdm.tqdm(pool.imap_unordered(process_item_worker, items, chunksize=100), total=current_batch_len)):
                    if result is None:
                        continue
                    split_data[target_split].append(result)
                
                total_processed += current_batch_len
                        
        except Exception as e:
            print(f"Error processing file {p_file}: {e}")
            continue

    pool.close()
    pool.join()
    
    # Save JSONs
    print("Saving JSON files...")
    for split_name, data_list in split_data.items():
        if not data_list:
            continue
            
        output_json = dataset_dir / f"{split_name}.json"
        print(f"  - Saving {len(data_list)} records to {output_json}")
        with open(output_json, 'w') as f:
            json.dump(data_list, f, indent=2)
            
    print(f"Finished. Total processed: {total_processed}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
