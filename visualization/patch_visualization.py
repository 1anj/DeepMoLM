import os
from typing import Optional
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import math

def Smiles2Img(smis: str, size: int = 224, savePath: Optional[str] = None):
    """
    Render a SMILES string to a PIL image.
    
    Args:
        smis: The SMILES string of the molecule.
        size: The width and height of the output image (square).
        savePath: If provided, saves the image to this path.
        
    Returns:
        PIL.Image object or None if conversion fails.
    """
    try:
        mol = Chem.MolFromSmiles(smis)
        if mol is None:
            print(f"Error: Invalid SMILES string '{smis}'")
            return None
            
        # Draw.MolsToGridImage returns a PIL image directly in recent RDKit versions
        # We use returnPNG=False to get a PIL object
        img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(size, size), returnPNG=False)
        
        if savePath is not None:
            img.save(savePath)
            print(f"Original molecule image saved to: {savePath}")
            
        return img
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def split_image_into_patches(image: Image.Image, patch_size: int = 16, output_dir: str = "patches"):
    """
    Splits a PIL image into n x patch_size x patch_size patches.
    
    Args:
        image: The source PIL Image.
        patch_size: The size of each square patch (default 16).
        output_dir: Directory to save the patches.
    """
    if image is None:
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    width, height = image.size
    
    # Calculate number of patches in x and y directions
    # If the image size isn't perfectly divisible by patch_size, we might ignore the edge or pad.
    # Here we assume we just crop what fits.
    n_patches_x = width // patch_size
    n_patches_y = height // patch_size
    
    count = 0
    print(f"Splitting image ({width}x{height}) into {patch_size}x{patch_size} patches...")
    
    for i in range(n_patches_y):
        for j in range(n_patches_x):
            # Define the box coordinates (left, upper, right, lower)
            left = j * patch_size
            upper = i * patch_size
            right = left + patch_size
            lower = upper + patch_size
            
            bbox = (left, upper, right, lower)
            patch = image.crop(bbox)
            
            # Save the patch
            patch_filename = f"patch_{i}_{j}.png"
            patch_path = os.path.join(output_dir, patch_filename)
            patch.save(patch_path)
            count += 1

    print(f"Successfully saved {count} patches to '{output_dir}'.")

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Define the molecule
    smiles_string = "OC(=O)CN(CCc1ccc(Cl)cc1)C(=O)[C@](C)(Cc2c[nH]c3ccccc23)NC(=O)OCC45CC6CC(C4)CC(C6)C5"
    
    # 2. Generate the image
    img_size = 1024
    original_img_path = "molecule_original.png"
    
    print(f"Processing ({smiles_string})...")
    
    molecule_image = Smiles2Img(smiles_string, size=img_size, savePath=original_img_path)
    
    # 3. Split into patches
    if molecule_image:
        split_image_into_patches(molecule_image, patch_size=256, output_dir="molecule_patches")