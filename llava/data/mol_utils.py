import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Geometry import Point3D
import logging
from typing import Optional
try:
    from e3fp.pipeline import fprints_from_mol_verbose
except ImportError:
    from e3fp.pipeline import fprints_from_mol as fprints_from_mol_verbose
from e3fp.fingerprint.fprinter import signed_to_unsigned_int

logger = logging.getLogger(__name__)


def Smiles2Img(smis: str, size: int = 224, savePath: Optional[str] = None):
    """Render a SMILES string to a PIL image."""
    mol = Chem.MolFromSmiles(smis)
    if mol is None:
        return None
    try:
        img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(size, size))
    except Exception as exc:
        logger.warning("Failed to render SMILES %s: %s", smis, exc)
        return None
    if savePath is not None:
        img.save(savePath)
    return img


def get_3d_fingerprints(smiles: str, coords: np.ndarray, bits: int = 4096, level: int = 3) -> torch.Tensor:
    """
    E3FP: generate per-atom fingerprints for levels 0..level.
    Returns: LongTensor [num_atoms, level+1]; zeros if e3fp fails.
    """

    fallback = torch.zeros(1, level + 1, dtype=torch.long)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return fallback

        coords = np.asarray(coords, dtype=np.float32)
        num_atoms = mol.GetNumAtoms()
        if coords.ndim != 2 or coords.shape[1] != 3:
            logger.warning(f"E3FP invalid coordinate shape for SMILES {smiles}: {coords.shape}")
            return fallback
        if len(coords) != num_atoms:
            # If coords correspond to heavy atoms only, drop hydrogens (incl. isotopic) and retry alignment.
            mol_no_h = Chem.RemoveHs(mol, updateExplicitCount=True, sanitize=False)
            if mol_no_h is not None and mol_no_h.GetNumAtoms() == len(coords):
                mol = mol_no_h
                num_atoms = mol.GetNumAtoms()
            else:
                if len(coords) > num_atoms:
                    coords = coords[:num_atoms]
                else:
                    pad = np.zeros((num_atoms - len(coords), 3), dtype=np.float32)
                    coords = np.concatenate([coords, pad], axis=0)

        conf = Chem.Conformer(num_atoms)
        for i, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        mol.RemoveAllConformers()
        mol.AddConformer(conf)
        mol.SetProp("_Name", smiles)

        params = {"bits": bits, "rdkit_invariants": True, "level": level, "all_iters": True, "exclude_floating": False}
        _, fingerprinter = fprints_from_mol_verbose(mol, fprint_params=params)

        num_levels = len(fingerprinter.level_shells) or (level + 1)
        fp_num_atom = max(1, len(fingerprinter.all_shells) // num_levels)

        fprints_all_atom = np.zeros((num_atoms, level + 1), dtype=np.int32)
        for i, shell in enumerate(fingerprinter.all_shells):
            lvl = i // fp_num_atom
            if lvl > level or shell.center_atom >= num_atoms:
                continue
            fp_i = signed_to_unsigned_int(shell.identifier) % bits
            fprints_all_atom[shell.center_atom, lvl] = fp_i

        return torch.tensor(fprints_all_atom, dtype=torch.long)
    except Exception as e:
        logger.error(f"E3FP failed for SMILES {smiles}: {e}")
        return fallback
