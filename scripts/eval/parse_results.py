from transformers import AutoTokenizer
import json
import os
import re
import glob
import numpy as np
from collections import Counter
from help_funcs import caption_evaluate
import argparse


def load_results(path):
    """Load results from a directory or file.
    
    Supports:
    1. Directory containing rank files (e.g., *_rank0.jsonl, *_rank1.jsonl, ...)
    2. Single merged file (e.g., results.jsonl)
    3. File path with associated rank files (legacy support)
    
    Results are merged and sorted by global_idx when available.
    """
    dicts = []
    
    # Case 1: Path is a directory - find all rank files inside
    if os.path.isdir(path):
        rank_pattern = os.path.join(path, "*_rank*.jsonl")
        rank_files = sorted(glob.glob(rank_pattern))
        
        # Also check for non-rank jsonl files
        all_jsonl = glob.glob(os.path.join(path, "*.jsonl"))
        non_rank_files = [f for f in all_jsonl if "_rank" not in os.path.basename(f)]
        
        if rank_files:
            print(f"Found {len(rank_files)} rank files in directory {path}:")
            for rf in rank_files:
                print(f"  - {os.path.basename(rf)}")
            
            all_results = []
            for rank_file in rank_files:
                with open(rank_file, 'r', encoding='utf8') as f:
                    for line in f:
                        if line.strip():
                            all_results.append(json.loads(line.strip()))
            
            # Sort by global_idx if present to maintain original order
            if all_results and 'global_idx' in all_results[0]:
                all_results.sort(key=lambda x: x.get('global_idx', 0))
            
            dicts = all_results
            print(f"Loaded {len(dicts)} results from rank files")
        
        elif non_rank_files:
            # Load from merged jsonl file(s)
            print(f"Loading from {len(non_rank_files)} jsonl file(s) in {path}")
            for jsonl_file in non_rank_files:
                print(f"  - {os.path.basename(jsonl_file)}")
                with open(jsonl_file, 'r', encoding='utf8') as f:
                    for line in f:
                        if line.strip():
                            dicts.append(json.loads(line.strip()))
            print(f"Loaded {len(dicts)} results")
        
        else:
            raise FileNotFoundError(
                f"No jsonl files found in directory: {path}\n"
                f"Expected files matching: *_rank*.jsonl or *.jsonl"
            )
    
    # Case 2: Path is a file
    elif os.path.isfile(path):
        # Check for rank files with same base name
        base, ext = os.path.splitext(path)
        rank_pattern = f"{base}_rank*{ext}"
        rank_files = sorted(glob.glob(rank_pattern))
        
        if rank_files:
            print(f"Found {len(rank_files)} rank files:")
            for rf in rank_files:
                print(f"  - {rf}")
            
            all_results = []
            for rank_file in rank_files:
                with open(rank_file, 'r', encoding='utf8') as f:
                    for line in f:
                        if line.strip():
                            all_results.append(json.loads(line.strip()))
            
            if all_results and 'global_idx' in all_results[0]:
                all_results.sort(key=lambda x: x.get('global_idx', 0))
            
            dicts = all_results
            print(f"Loaded {len(dicts)} results from rank files")
        else:
            # Load from single file
            print(f"Loading from single file: {path}")
            with open(path, 'r', encoding='utf8') as f:
                for line in f:
                    if line.strip():
                        dicts.append(json.loads(line.strip()))
            print(f"Loaded {len(dicts)} results")
    
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    return dicts


def main(file_path, tokenizer_path):
    dicts = load_results(file_path)

    _prediction = []
    _targets = []
    _task_type = []
    for _dict in dicts:
        _prediction.extend(_dict['prediction'])
        _targets.extend(_dict['target'])
        _task_type.extend(_dict['task_type'])

    task_type_counter = Counter(_task_type)
    prediction = {}
    targets = {}

    prediction['HOMO'] = []
    prediction['LUMO'] = []
    prediction['HOMO-LUMO Gap'] = []
    prediction['SCF Energy'] = []

    prediction['Molecular Weight'] = []
    prediction['LogP'] = []
    prediction['Topological Polar Surface Area'] = []
    prediction['Complexity'] = []

    prediction['Description'] = []
    prediction['Caption'] = []

    targets['HOMO'] = []
    targets['LUMO'] = []
    targets['HOMO-LUMO Gap'] = []
    targets['SCF Energy'] = []

    targets['Molecular Weight'] = []
    targets['LogP'] = []
    targets['Topological Polar Surface Area'] = []
    targets['Complexity'] = []

    targets['Description'] = []
    targets['Caption'] = []

    pattern = r'-?\d+\.\d+'
    # pattern = r'(-?\d+\.\d+|-?\d+)'

    e_prediction = []
    e_target = []

    for i, t in enumerate(_task_type):
        if t in ['Description', 'Caption']:
            prediction[t].append(_prediction[i])
            targets[t].append(_targets[i])
        else:
            pre_matches = re.findall(pattern, _prediction[i])
            tar_matches = re.findall(pattern, _targets[i])
            if t in ['HOMO', 'LUMO', 'HOMO-LUMO Gap']:
                if len(pre_matches) > 0 and -20 < float(pre_matches[0]) < 20:
                    prediction[t].append(float(pre_matches[0]))
                    targets[t].append(float(tar_matches[0]))
                else:
                    e_prediction.append(_prediction[i])
                    e_target.append(_targets[i])
            elif t in ['SCF Energy']:
                if len(pre_matches) > 0 and -5 < float(pre_matches[0]) < 0:
                    prediction[t].append(float(pre_matches[0]))
                    targets[t].append(float(tar_matches[0]))
                else:
                    e_prediction.append(_prediction[i])
                    e_target.append(_targets[i])
            elif t in ['LogP']:
                if len(pre_matches) > 0 and -30 < float(pre_matches[0]) < 50:
                    prediction[t].append(float(pre_matches[0]))
                    targets[t].append(float(tar_matches[0]))
                else:
                    e_prediction.append(_prediction[i])
                    e_target.append(_targets[i])
            elif t in ['Topological Polar Surface Area']:
                if len(pre_matches) > 0 and 0 <= float(pre_matches[0]) < 2000:
                    prediction[t].append(float(pre_matches[0]))
                    targets[t].append(float(tar_matches[0]))
                else:
                    e_prediction.append(_prediction[i])
                    e_target.append(_targets[i])
            elif t in ['Complexity']:
                if len(pre_matches) > 0 and 0 <= float(pre_matches[0]) < 10000:
                    prediction[t].append(float(pre_matches[0]))
                    targets[t].append(float(tar_matches[0]))
                else:
                    e_prediction.append(_prediction[i])
                    e_target.append(_targets[i])
            elif t in ['Molecular Weight']:
                if len(pre_matches) > 0 and 0 < float(pre_matches[0]) < 4000:
                    prediction[t].append(float(pre_matches[0]))
                    targets[t].append(float(tar_matches[0]))
                else:
                    e_prediction.append(_prediction[i])
                    e_target.append(_targets[i])

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, padding_side='right')

    for key in prediction.keys():
        if key in ['Description', 'Caption']:
            if len(prediction[key]) > 0:
                print(f'{key}')
                bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = caption_evaluate(prediction[key], targets[key], tokenizer, 128)
            else:
                print(f'{key}    No samples')
        else:
            total_count = task_type_counter.get(key, 0)
            if total_count == 0:
                print(f'{key}    No samples')
                continue
            if len(prediction[key]) == 0:
                print(f'{key}    Valid Ratio:0.0000, MAE:N/A (no valid predictions)')
                continue
            valid_ratio = len(prediction[key]) / total_count
            error = np.array(prediction[key]) - np.array(targets[key])
            mae = np.mean(np.abs(error))
            print(f'{key}    Valid Ratio:{valid_ratio:.4f}, MAE:{mae:.3f}')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse results from distributed evaluation output.")
    parser.add_argument(
        "--file_path", 
        type=str, 
        default="./eval_outputs/specialist_pubchem_cap/test", 
        help="Path to directory containing rank result files (*_rank*.jsonl) or a single results file."
    )
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        default="./checkpoints/Llama-2-7b-hf", 
        help="The path to the tokenizer."
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args.file_path, args.tokenizer_path)