import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())
import json
import argparse
import pickle
import numpy as np
import multiprocessing
import lmdb
import selfies as sf
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Geometry import Point3D
from PIL import Image
import tqdm

# E3FP imports
try:
    from e3fp.pipeline import fprints_from_mol_verbose
except ImportError:
    from e3fp.pipeline import fprints_from_mol as fprints_from_mol_verbose
from e3fp.fingerprint.fprinter import signed_to_unsigned_int



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

def get_3d_fingerprints(smiles: str, coords: np.ndarray, bits: int = 4096, level: int = 3):
    """
    Generate E3FP per-atom fingerprints.
    Returns: ndarray [num_atoms, level+1]; -1 padding if e3fp fails.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.full((1, level + 1), -1, dtype=np.int32)

        coords = np.asarray(coords, dtype=np.float32)
        num_atoms = mol.GetNumAtoms()
        
        if coords.ndim != 2 or coords.shape[1] != 3:
            return np.full((1, level + 1), -1, dtype=np.int32)
            
        if len(coords) != num_atoms:
            # Try removing hydrogens
            mol_no_h = Chem.RemoveHs(mol, updateExplicitCount=True, sanitize=False)
            if mol_no_h is not None and mol_no_h.GetNumAtoms() == len(coords):
                mol = mol_no_h
                num_atoms = mol.GetNumAtoms()
            else:
                # Pad or truncate coords
                if len(coords) > num_atoms:
                    coords = coords[:num_atoms]
                else:
                    pad = np.zeros((num_atoms - len(coords), 3), dtype=np.float32)
                    coords = np.concatenate([coords, pad], axis=0)

        # Set conformer
        conf = Chem.Conformer(num_atoms)
        for i in range(num_atoms):
            if i < len(coords):
                x, y, z = coords[i]
            else:
                x, y, z = 0.0, 0.0, 0.0
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        mol.RemoveAllConformers()
        mol.AddConformer(conf)
        mol.SetProp("_Name", smiles)

        # Generate E3FP fingerprints
        params = {
            "bits": bits,
            "rdkit_invariants": True,
            "level": level,
            "all_iters": True,
            "exclude_floating": False
        }
        fprints_list, fingerprinter = fprints_from_mol_verbose(mol, fprint_params=params)

        num_levels = len(fingerprinter.level_shells.keys()) or (level + 1)
        fp_num_atom = max(1, len(fingerprinter.all_shells) // num_levels)

        fprints_all_atom = np.full((num_atoms, level + 1), -1, dtype=np.int32)
        for i, shell in enumerate(fingerprinter.all_shells):
            lvl = i // fp_num_atom
            if lvl > level or shell.center_atom >= num_atoms:
                continue
            fp_i = signed_to_unsigned_int(shell.identifier) % bits
            fprints_all_atom[shell.center_atom, lvl] = fp_i

        return fprints_all_atom
    except Exception as e:
        return np.full((1, level + 1), -1, dtype=np.int32)

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
        smiles = item.get("smiles") or item.get("smi") or item.get("canonical_smiles") or item.get("SMILES")
        
        if not smiles:
            return None

        # Generate unique ID from global index or cid
        unique_id = str(item.get('cid', item.get('global_idx')))
        
        # Generate Image
        img_filename = f"{unique_id}.png"
        img_path = worker_output_dir / img_filename
        
        try:
            # Check if image exists to avoid re-generating? 
            # For now, we overwrite or skip if strict.
            # But let's just generate.
            img = Smiles2Img(smiles, size=1024)
            if img is not None:
                img.save(img_path)
            else:
                return None # parsing failed
        except Exception as e:
            return None
        
        
        # Construct Result
        result_data = item.copy()
        
        # Process coordinates -> coord_norm (normalize)
        if 'coordinates' in result_data:
            coords_raw = result_data.pop('coordinates')
            # Handle list of arrays or list of lists
            if isinstance(coords_raw, list) and len(coords_raw) > 0:
                if isinstance(coords_raw[0], np.ndarray):
                    coords = coords_raw[0]  # Take first conformer if multiple
                else:
                    coords = np.array(coords_raw)
            else:
                coords = np.array(coords_raw)
            
            # Normalize coordinates (center at origin)
            if coords.size > 0:
                coords = coords - coords.mean(axis=0)
            result_data['coord_norm'] = coords
        else:
            coords = None
        
        # Generate Selfies
        if 'selfies' not in result_data:
            try:
                result_data['selfies'] = sf.encoder(smiles)
            except Exception:
                result_data['selfies'] = None

        # Generate molecule_fp
        if 'molecule_fp' not in result_data and coords is not None:
            try:
                fp_array = get_3d_fingerprints(smiles, coords)
                result_data['molecule_fp'] = fp_array  # Will be serialized later
            except Exception as e:
                result_data['molecule_fp'] = None

        # Add required fields
        result_data['id'] = unique_id
        result_data['image'] = img_filename
        result_data['instruction'] = 'Describe the input molecule.'
        
        # Rename description to output
        if 'description' in result_data:
            result_data['output'] = result_data.pop('description')
        
        # Rename enriched_description to enriched_output
        if 'enriched_description' in result_data:
            result_data['enriched_output'] = result_data.pop('enriched_description')
        
        # Remove atoms field
        if 'atoms' in result_data:
            del result_data['atoms']
        
        # Clean up types
        
        def make_serializable(obj):
            if isinstance(obj, (np.ndarray, np.generic)):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_serializable(x) for x in obj]
            if isinstance(obj, bytes):
                try:
                    return obj.decode('utf-8')
                except:
                    return None
            return obj

        # Process all fields
        result_data = make_serializable(result_data)
        
        return result_data
        
    except Exception as e:
        return None

# ----------------- Main ----------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_path", type=str, default="data/3d-pubchem-all.lmdb", help="Path to input LMDB file")
    parser.add_argument("--output_dir", type=str, default="data/processed/3d-pubchem", help="Directory to save output")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process (for debugging)")
    args = parser.parse_args()

    lmdb_path = Path(args.lmdb_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    images_dir = output_dir / "images"
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading from {lmdb_path}...")
    if not lmdb_path.exists():
        print(f"Error: {lmdb_path} does not exist.")
        return

    # Check approximate size
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        length = txn.stat()['entries']
    
    print(f"Total entries in LMDB: {length}")
    
    if args.max_samples:
        length = min(length, args.max_samples)
        print(f"Processing limited to {length} samples.")

    # Generator for Items
    def lmdb_generator():
        env_gen = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=False)
        with env_gen.begin() as txn:
            cursor = txn.cursor()
            count = 0
            for i, (key, value) in enumerate(cursor):
                if args.max_samples and count >= args.max_samples:
                    break
                try:
                    item = pickle.loads(value)
                    # item usually has 'cid', 'smiles', etc.
                    # We inject a global idx just in case
                    if 'global_idx' not in item:
                        item['global_idx'] = i
                    yield item
                    count += 1
                except Exception:
                    continue
        env_gen.close()

    # Parallel Processing
    init_args = (str(images_dir),)
    pool = multiprocessing.Pool(processes=args.num_workers, initializer=init_worker, initargs=init_args)
    
    results = []
    
    try:
        # We need to materialize the generator for tqdm to show progress effectively if we want exact bar,
        # but for large LMDB we might want to just iterate.
        # imap_unordered is good.
        
        # Note: imap_unordered consumes the iterable.
        # Since we use a generator, we can pass it directly.
        
        iterator = lmdb_generator()
        
        for result in tqdm.tqdm(pool.imap_unordered(process_item_worker, iterator, chunksize=100), total=length):
            if result is not None:
                results.append(result)
                
    except KeyboardInterrupt:
        print("Interrupted.")
        pool.terminate()
        pool.join()
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        
    pool.close()
    pool.join()
    
    # Save JSON
    output_json = output_dir / "all.json"
    print(f"Saving {len(results)} records to {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
        
    print("Done.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
