#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SELFIES to PNG Converter
=======================

A command-line utility to render SELFIES strings as 2D molecular images with
molecular formulas and proper subscript formatting.

This script demonstrates how to:
- Parse SELFIES strings and decode them to SMILES
- Generate 2D molecular coordinates using RDKit
- Create publication-quality molecular images
- Add custom legends with molecular formulas
- Handle font rendering and subscripts

Example Usage:
    python selfies2png.py "[C][C][O]"  # Ethanol
    python selfies2png.py "[C][C][Branch1][C][C][C][C]" -o isobutane.png  # Isobutane
    python selfies2png.py "[C][=C][C][=C][C][=C][Ring1][=Branch1]" --size 800  # Benzene, larger image

Author: Hunter Heidenreich
Website: https://hunterheidenreich.com
"""

import argparse
import hashlib
import sys
from pathlib import Path

# SELFIES import
import selfies as sf

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, rdMolDescriptors

# PIL imports for image manipulation
from PIL import Image, ImageDraw, ImageFont

# Constants for image configuration
DEFAULT_IMAGE_SIZE = 500
LEGEND_HEIGHT_RATIO = 0.08  # Legend height as ratio of image size
LEGEND_Y_OFFSET_RATIO = 0.02  # Y offset as ratio of image size
LEGEND_X_OFFSET_RATIO = 0.02  # X offset as ratio of image size
SUBSCRIPT_Y_OFFSET_RATIO = 0.006  # Subscript offset as ratio of image size

# Font size ratios based on image size
REGULAR_FONT_RATIO = 0.028  # Regular font size as ratio of image size
SUBSCRIPT_FONT_RATIO = 0.02  # Subscript font size as ratio of image size

# Font paths for different operating systems
FONT_PATHS = [
    "/System/Library/Fonts/Arial.ttf",  # macOS
    "/usr/share/fonts/truetype/arial.ttf",  # Linux
    "C:/Windows/Fonts/arial.ttf",  # Windows
]


def _calculate_dynamic_sizes(image_size: int):
    """Calculate dynamic sizing values based on image size."""
    return {
        'legend_height': int(image_size * LEGEND_HEIGHT_RATIO),
        'legend_y_offset': int(image_size * LEGEND_Y_OFFSET_RATIO),
        'legend_x_offset': int(image_size * LEGEND_X_OFFSET_RATIO),
        'subscript_y_offset': int(image_size * SUBSCRIPT_Y_OFFSET_RATIO),
        'regular_font_size': int(image_size * REGULAR_FONT_RATIO),
        'subscript_font_size': int(image_size * SUBSCRIPT_FONT_RATIO),
    }


def _load_fonts(regular_size: int, subscript_size: int):
    """Load system fonts for text rendering, with fallback to default font."""
    font_regular = None
    font_small = None

    for font_path in FONT_PATHS:
        try:
            font_regular = ImageFont.truetype(font_path, regular_size)
            font_small = ImageFont.truetype(font_path, subscript_size)
            break
        except (OSError, IOError):
            continue

    if font_regular is None:
        font_regular = ImageFont.load_default()
        font_small = ImageFont.load_default()

    return font_regular, font_small


def create_molecule_image(
    mol: Chem.Mol, selfies_string: str, size: int = DEFAULT_IMAGE_SIZE
) -> Image.Image:
    """
    Creates a molecule image with a legend showing molecular formula and SELFIES string.

    Args:
        mol: RDKit molecule object (already validated)
        selfies_string: Original SELFIES string for legend display
        size: Image size in pixels (square image)

    Returns:
        PIL Image object with molecule structure and formatted legend
    """
    # Calculate dynamic sizes based on image size
    sizes = _calculate_dynamic_sizes(size)
    
    rdDepictor.Compute2DCoords(mol)
    molecular_formula = rdMolDescriptors.CalcMolFormula(mol)

    mol_img = Draw.MolToImage(mol, size=(size, size))
    if mol_img.mode != "RGBA":
        mol_img = mol_img.convert("RGBA")

    # Calculate required legend height based on content - be more generous
    legend_height = _calculate_legend_height(molecular_formula, selfies_string, sizes, size)
    
    # Add extra safety margin to ensure text doesn't get cut off
    legend_height = max(legend_height, int(size * 0.15))  # At least 15% of image height
    
    # Calculate if we need extra width for the SELFIES text
    temp_img = Image.new("RGBA", (1, 1), "white")
    temp_draw = ImageDraw.Draw(temp_img)
    font_regular, _ = _load_fonts(sizes['regular_font_size'], sizes['subscript_font_size'])
    
    selfies_label = "SELFIES: "
    label_width = int(temp_draw.textlength(selfies_label, font=font_regular))
    selfies_width = int(temp_draw.textlength(selfies_string, font=font_regular))
    total_selfies_width = label_width + selfies_width + (sizes['legend_x_offset'] * 2)
    
    # Make image wider if needed to accommodate the full SELFIES string
    final_width = max(size, total_selfies_width)
    
    total_height = size + legend_height
    final_img = Image.new("RGBA", (final_width, total_height), "white")
    
    # Center the molecule image if we made the canvas wider
    mol_x_offset = (final_width - size) // 2
    final_img.paste(mol_img, (mol_x_offset, 0))

    draw = ImageDraw.Draw(final_img)
    font_regular, font_small = _load_fonts(sizes['regular_font_size'], sizes['subscript_font_size'])

    _draw_molecular_formula(draw, molecular_formula, font_regular, font_small, sizes, size, mol_x_offset)
    _draw_selfies_legend(draw, selfies_string, font_regular, sizes, molecular_formula, size, final_width)

    return final_img


def _draw_molecular_formula(
    draw: ImageDraw.Draw, formula: str, font_regular, font_small, sizes: dict, image_size: int, mol_x_offset: int = 0
) -> int:
    """Draw molecular formula with proper subscript formatting."""
    y_pos = image_size + sizes['legend_y_offset']
    x_pos = sizes['legend_x_offset'] + mol_x_offset

    draw.text((x_pos, y_pos), "Formula: ", fill="black", font=font_regular)
    x_pos += int(draw.textlength("Formula: ", font=font_regular))

    for char in formula:
        if char.isdigit():
            draw.text(
                (x_pos, y_pos + sizes['subscript_y_offset']), char, fill="black", font=font_small
            )
            x_pos += int(draw.textlength(char, font=font_small))
        else:
            draw.text((x_pos, y_pos), char, fill="black", font=font_regular)
            x_pos += int(draw.textlength(char, font=font_regular))

    return x_pos


def _calculate_legend_height(formula: str, selfies: str, sizes: dict, image_size: int) -> int:
    """Calculate the required height for the legend based on content."""
    # Create a temporary draw object to measure text
    temp_img = Image.new("RGBA", (1, 1), "white")
    temp_draw = ImageDraw.Draw(temp_img)
    font_regular, _ = _load_fonts(sizes['regular_font_size'], sizes['subscript_font_size'])
    
    # Calculate if SELFIES will fit on the same line as formula
    formula_text = f"Formula: {formula}"
    separator = " | SELFIES: "
    total_prefix_width = int(temp_draw.textlength(formula_text, font=font_regular)) + int(temp_draw.textlength(separator, font=font_regular))
    available_width_same_line = image_size - sizes['legend_x_offset'] - total_prefix_width - sizes['legend_x_offset']
    selfies_width = int(temp_draw.textlength(selfies, font=font_regular))
    
    base_height = sizes['legend_y_offset'] * 2 + sizes['regular_font_size']
    line_height = sizes['regular_font_size'] + 6  # Line spacing to match drawing function
    
    if selfies_width <= available_width_same_line:
        # Single line layout - add extra padding
        return base_height + 15
    else:
        # SELFIES goes on new line - calculate width available for SELFIES line
        selfies_label_width = int(temp_draw.textlength("SELFIES: ", font=font_regular))
        available_width_new_line = image_size - sizes['legend_x_offset'] - selfies_label_width - sizes['legend_x_offset']
        
        if selfies_width <= available_width_new_line:
            # SELFIES fits on one new line
            return base_height + line_height + 15
        else:
            # SELFIES needs wrapping - parse into tokens for accurate calculation
            tokens = []
            current_token = ""
            
            for char in selfies:
                if char == '[':
                    if current_token:
                        tokens.append(current_token)
                        current_token = ""
                    current_token = char
                elif char == ']':
                    current_token += char
                    tokens.append(current_token)
                    current_token = ""
                else:
                    current_token += char
            
            if current_token:
                tokens.append(current_token)
            
            # Simulate line wrapping
            lines_needed = 1
            current_line_width = 0
            
            for token in tokens:
                token_width = int(temp_draw.textlength(token, font=font_regular))
                if current_line_width + token_width > available_width_new_line and current_line_width > 0:
                    lines_needed += 1
                    current_line_width = token_width
                else:
                    current_line_width += token_width
            
            # Add extra height for the second line (SELFIES line) plus wrapped lines plus extra padding
            total_height = base_height + line_height + (line_height * (lines_needed - 1)) + 25  # Extra generous padding
            return total_height


def _draw_wrapped_text(
    draw: ImageDraw.Draw, text: str, start_x: int, start_y: int, max_width: int, font, sizes: dict
) -> None:
    """Draw text with word wrapping at bracket boundaries for SELFIES strings."""
    # For SELFIES, we want to break at bracket boundaries to maintain readability
    # Split SELFIES into tokens (brackets and their contents)
    tokens = []
    current_token = ""
    
    for char in text:
        if char == '[':
            if current_token:
                tokens.append(current_token)
                current_token = ""
            current_token = char
        elif char == ']':
            current_token += char
            tokens.append(current_token)
            current_token = ""
        else:
            current_token += char
    
    if current_token:
        tokens.append(current_token)
    
    # Now draw tokens, wrapping as needed
    x_pos = start_x
    y_pos = start_y
    line_height = sizes['regular_font_size'] + 4  # More generous line spacing
    
    for token in tokens:
        token_width = int(draw.textlength(token, font=font))
        
        # If this token would exceed the line, wrap to next line
        if x_pos + token_width > start_x + max_width and x_pos > start_x:
            y_pos += line_height
            x_pos = start_x
        
        draw.text((x_pos, y_pos), token, fill="black", font=font)
        x_pos += token_width


def _draw_selfies_legend(
    draw: ImageDraw.Draw, selfies: str, font_regular, sizes: dict, formula: str, 
    original_mol_size: int, final_image_width: int
) -> None:
    """Add SELFIES string to the image legend with proper text wrapping."""
    y_pos = original_mol_size + sizes['legend_y_offset']
    
    # ALWAYS put SELFIES on its own line - no more trying to fit on same line as formula
    selfies_y_pos = y_pos + sizes['regular_font_size'] + 6
    selfies_label = "SELFIES: "
    label_width = int(draw.textlength(selfies_label, font=font_regular))
    
    draw.text((sizes['legend_x_offset'], selfies_y_pos), selfies_label, fill="black", font=font_regular)
    
    # Use wrapped text drawing for ALL SELFIES strings
    selfies_x_pos = sizes['legend_x_offset'] + label_width
    available_width_for_selfies = final_image_width - selfies_x_pos - sizes['legend_x_offset']
    _draw_wrapped_text(
        draw, selfies, selfies_x_pos, selfies_y_pos, 
        available_width_for_selfies, font_regular, sizes
    )


def selfies_to_png(
    selfies_string: str, output_file: str, size: int = DEFAULT_IMAGE_SIZE
) -> None:
    """
    Convert a SELFIES string to a PNG image with molecular formula legend.

    Args:
        selfies_string: Valid SELFIES string representing a molecule
        output_file: Path where the PNG image will be saved
        size: Square image dimensions in pixels

    Raises:
        ValueError: If SELFIES string is invalid or size is non-positive
        IOError: If file cannot be saved to the specified location
    """
    if not selfies_string or not selfies_string.strip():
        raise ValueError("SELFIES string cannot be empty")

    if size <= 0:
        raise ValueError(f"Image size must be positive, got: {size}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Decode SELFIES to SMILES
    try:
        smiles_string = sf.decoder(selfies_string.strip())
    except Exception as e:
        raise ValueError(
            f"Invalid SELFIES string: '{selfies_string}'. "
            f"SELFIES decoding error: {e}"
        )

    if not smiles_string:
        raise ValueError(
            f"SELFIES string '{selfies_string}' decoded to empty SMILES. "
            f"Please check the SELFIES syntax."
        )

    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        raise ValueError(
            f"Invalid SELFIES string: '{selfies_string}'. "
            f"Decoded to SMILES '{smiles_string}' which is not valid. "
            f"Please check the syntax and try again."
        )

    img = create_molecule_image(mol, selfies_string.strip(), size)

    try:
        img.save(output_file, "PNG", optimize=True)
        print(f"Image successfully saved to: {output_file}")
    except Exception as e:
        raise IOError(f"Failed to save image to '{output_file}': {e}")


def create_safe_filename(selfies_string: str) -> str:
    """
    Generate a filesystem-safe filename from a SELFIES string using MD5 hash.

    Args:
        selfies_string: The input SELFIES string

    Returns:
        A safe filename ending with .png
    """
    clean_selfies = selfies_string.strip()
    hasher = hashlib.md5(clean_selfies.encode("utf-8"))
    return f"{hasher.hexdigest()}.png"


def main() -> None:
    """Command-line interface for the SELFIES to PNG converter."""
    parser = argparse.ArgumentParser(
        description="Convert SELFIES strings to publication-quality PNG images with molecular formulas.",
        epilog="""
Examples:
  %(prog)s "[C][C][O]"                                    # Ethanol with auto-generated filename
  %(prog)s "[C][C][Branch1][C][C][C][C]"                  # Isobutane with auto-generated filename
  %(prog)s "[C][=C][C][=C][C][=C][Ring1][=Branch1]" -o benzene.png   # Benzene with custom filename  
  %(prog)s "[C][C][O]" --size 800                         # Ethanol with larger image size

Common SELFIES patterns:
  [C][C][O]                                  - Ethanol
  [C][C][Branch1][C][=O][O]                  - Acetic acid
  [C][=C][C][=C][C][=C][Ring1][=Branch1]     - Benzene
  [C][C][Branch1][C][C][C]                   - Isobutane
  [N][C][Branch1][C][=O][C][=C][C][=C][C][=C][Ring1][=Branch1]  - Benzamide
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "selfies",
        type=str,
        help="SELFIES string of the molecule to visualize (e.g., '[C][C][O]' for ethanol)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        metavar="FILE",
        help="Output PNG filename. If not provided, generates a unique filename "
        "based on the SELFIES string hash. Extension .png will be added if missing.",
    )

    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        metavar="PIXELS",
        help=f"Square image size in pixels (default: {DEFAULT_IMAGE_SIZE}). "
        f"Typical values: 300 (small), 500 (medium), 800 (large).",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging",
    )

    args = parser.parse_args()

    if args.verbose:
        print(f"Input SELFIES: {args.selfies}")
        print(f"Image size: {args.size}x{args.size} pixels")

    if args.output:
        output_filename = (
            args.output
            if args.output.lower().endswith(".png")
            else f"{args.output}.png"
        )
        if args.verbose:
            print(f"Using custom filename: {output_filename}")
    else:
        output_filename = create_safe_filename(args.selfies)
        if args.verbose:
            print(f"Generated filename: {output_filename}")

    try:
        selfies_to_png(args.selfies, output_filename, args.size)

        if args.verbose:
            # Decode and show the SMILES for reference
            try:
                decoded_smiles = sf.decoder(args.selfies.strip())
                print(f"Decoded SMILES: {decoded_smiles}")
            except Exception:
                pass
            print("Conversion completed successfully!")

    except ValueError as e:
        print(f"Input Error: {e}", file=sys.stderr)
        print("Tip: Check your SELFIES string syntax", file=sys.stderr)
        sys.exit(1)

    except IOError as e:
        print(f"File Error: {e}", file=sys.stderr)
        print("Tip: Check file permissions and disk space", file=sys.stderr)
        sys.exit(2)

    except ImportError as e:
        print(f"Dependencies Error: {e}", file=sys.stderr)
        print(
            "Tip: Install required packages with 'pip install rdkit pillow selfies'",
            file=sys.stderr,
        )
        sys.exit(3)

    except Exception as e:
        print(f"Unexpected Error: {e}", file=sys.stderr)
        print("Tip: Please report this issue if it persists", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()
