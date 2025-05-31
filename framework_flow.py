"""
Multi-Crystal Spectrum Set Dataset Generation Pipeline

This module provides functionality for generating rotated crystal structures and their annotations.
It creates a dataset of crystal structures with various rotations and corresponding visualizations.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for parallel processing
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dataset_generation")

# Constants
ELEMENT_COLORS: Dict[str, str] = {
    'Ag': 'silver',
    'Au': 'gold',
    'Pb': 'darkgrey',
    'S': 'yellow',
    'Zn': 'blue',
    'O': 'red'
}

ATOMIC_MASSES: Dict[str, float] = {
    'Ag': 107.87,
    'Au': 196.97,
    'Pb': 207.2,
    'S': 32.06,
    'Zn': 65.38,
    'O': 16.00
}

AVOGADRO: float = 6.022e23  # atoms/mol

@dataclass
class CrystalStructure:
    """Represents a crystal structure with its properties."""
    n_atoms: int
    comment: str
    atoms: List[str]
    coords: np.ndarray

class DatasetGenerationError(Exception):
    """Base exception for dataset generation errors."""
    pass

class FileError(DatasetGenerationError):
    """Raised when there's an error with file operations."""
    pass

class ValidationError(DatasetGenerationError):
    """Raised when there's an error validating the structure."""
    pass

def load_xyz(file_path: Union[str, Path]) -> CrystalStructure:
    """Load an XYZ file and return its contents as a CrystalStructure object.

    Args:
        file_path: Path to the XYZ file

    Returns:
        CrystalStructure object containing the file contents

    Raises:
        FileError: If there's an error reading the file
        ValidationError: If the file format is invalid
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            raise ValidationError(f"Invalid XYZ file format in {file_path}: not enough lines")
        
        try:
            n_atoms = int(lines[0].strip())
        except ValueError:
            raise ValidationError(f"Invalid number of atoms in {file_path}: {lines[0]}")
        
        comment = lines[1].strip()
        atoms = []
        coords = []
        
        for i, line in enumerate(lines[2:], start=2):
            parts = line.split()
            if len(parts) < 4:
                logger.warning(f"Skipping invalid line {i} in {file_path}: {line}")
                continue
            atoms.append(parts[0])
            try:
                coords.append(list(map(float, parts[1:4])))
            except ValueError:
                raise ValidationError(f"Invalid coordinates at line {i} in {file_path}: {line}")
        
        if len(atoms) != n_atoms:
            raise ValidationError(f"Number of atoms mismatch in {file_path}: declared {n_atoms}, found {len(atoms)}")
        
        return CrystalStructure(n_atoms, comment, atoms, np.array(coords))
    except Exception as e:
        if isinstance(e, (FileError, ValidationError)):
            raise
        raise FileError(f"Error reading XYZ file {file_path}: {e}")

def write_xyz(file_path: Union[str, Path], structure: CrystalStructure) -> None:
    """Write a CrystalStructure object to an XYZ file.

    Args:
        file_path: Path where to save the XYZ file
        structure: CrystalStructure object to write

    Raises:
        FileError: If there's an error writing the file
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"{structure.n_atoms}\n")
            f.write(f"{structure.comment}\n")
            for atom, coord in zip(structure.atoms, structure.coords):
                coord_str = " ".join(f"{x:.8f}" for x in coord)
                f.write(f"{atom} {coord_str}\n")
    except Exception as e:
        raise FileError(f"Error writing XYZ file {file_path}: {e}")

def rotate_coords(coords: np.ndarray, angle_deg: float, axis: np.ndarray) -> np.ndarray:
    """Rotate coordinates by a given angle around an axis.

    Args:
        coords: Array of coordinates to rotate
        angle_deg: Rotation angle in degrees
        axis: Rotation axis vector

    Returns:
        Rotated coordinates
    """
    angle_rad = np.deg2rad(angle_deg)
    rotation = Rotation.from_rotvec(axis * angle_rad)
    return rotation.apply(coords)

def create_3d_plot(structure: CrystalStructure) -> plt.Figure:
    """Create a 3D scatter plot of the crystal structure.

    Args:
        structure: CrystalStructure object to visualize

    Returns:
        Matplotlib figure containing the plot
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for element in set(structure.atoms):
        indices = [i for i, a in enumerate(structure.atoms) if a == element]
        element_coords = structure.coords[indices]
        color = ELEMENT_COLORS.get(element, 'black')
        ax.scatter(element_coords[:, 0], element_coords[:, 1], element_coords[:, 2],
                   s=10, color=color)
    
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    
    return fig

def fibonacci_sphere(samples: int) -> np.ndarray:
    """Generate points uniformly distributed on a unit sphere using Fibonacci lattice.

    Args:
        samples: Number of points to generate

    Returns:
        Array of shape (samples, 3) containing the points
    """
    points = []
    offset = 2.0 / samples
    increment = np.pi * (3.0 - np.sqrt(5.0))
    
    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y * y)
        phi = ((i % samples) * increment) % (2 * np.pi)
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append([x, y, z])
    
    return np.array(points)

def compute_properties(structure: CrystalStructure) -> Tuple[float, Tuple[float, float, float], float, float]:
    """Compute physical properties of the crystal structure.

    Args:
        structure: CrystalStructure object to analyze

    Returns:
        Tuple containing:
        - Cell Volume (Å³)
        - Lattice Parameters (a, b, c)
        - Average NN Distance (Å)
        - Density (g/cm³)
    """
    if structure.coords.shape[0] == 0:
        return 0.0, (0.0, 0.0, 0.0), 0.0, 0.0

    x_min, x_max = np.min(structure.coords[:,0]), np.max(structure.coords[:,0])
    y_min, y_max = np.min(structure.coords[:,1]), np.max(structure.coords[:,1])
    z_min, z_max = np.min(structure.coords[:,2]), np.max(structure.coords[:,2])
    
    a = x_max - x_min
    b = y_max - y_min
    c = z_max - z_min
    volume = a * b * c  # in Å³
    
    dists = np.linalg.norm(structure.coords[:, None, :] - structure.coords[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    avg_nn = np.mean(np.min(dists, axis=1))
    
    total_mass = sum(ATOMIC_MASSES.get(atom, 0) for atom in structure.atoms) / AVOGADRO
    volume_cm3 = volume * 1e-24  # 1 Å³ = 1e-24 cm³
    density = total_mass / volume_cm3 if volume_cm3 > 0 else 0
    
    return volume, (a, b, c), avg_nn, density

def generate_annotation(
    material: str,
    structure: str,
    crystal: CrystalStructure,
    out_annotation_file: Union[str, Path]
) -> None:
    """Generate an annotation file for the crystal structure.

    Args:
        material: Material name
        structure: Structure name
        crystal: CrystalStructure object
        out_annotation_file: Path where to save the annotation

    Raises:
        FileError: If there's an error writing the annotation file
    """
    volume, lattice_params, avg_nn, density = compute_properties(crystal)
    a, b, c = lattice_params
    
    if '[' in crystal.comment:
        rotation_axis = crystal.comment.split('[')[1].split(']')[0]
        rotation_axis = [round(float(x), 3) for x in rotation_axis.split()]
        rotation_axis = ', '.join(str(x) for x in rotation_axis)
        rotation_axis = f"[{rotation_axis}]"
    else:
        rotation_axis = 'Original Structure'
    
    annotation = (
        f"Material: {material}\n"
        f"Structure Prototype: {structure}\n"
        f"Atomic Count: {crystal.n_atoms}\n"
        f"Cell Volume: {volume:.2f} Å³\n"
        f"Lattice Parameters: a = {a:.2f} Å, b = {b:.2f} Å, c = {c:.2f} Å\n"
        f"Average NN Distance: {avg_nn:.2f} Å\n"
        f"Rotation Axis: {rotation_axis}\n"
        f"Density: {density:.2f} g/cm³\n"
        "\n"
        "Summary:\n"
        f"This {material} crystal structure with prototype {structure} contains {crystal.n_atoms} atoms "
        f"and has a cell volume of {volume:.2f} Å³. Its lattice is characterized by parameters a = {a:.2f} Å, "
        f"b = {b:.2f} Å, and c = {c:.2f} Å, suggesting an orthogonal geometry. The average nearest neighbor "
        f"distance is {avg_nn:.2f} Å, and the density is estimated to be {density:.2f} g/cm³.\n"
    )
    
    try:
        with open(out_annotation_file, 'w', encoding='utf-8') as f:
            f.write(annotation)
    except Exception as e:
        raise FileError(f"Error writing annotation file {out_annotation_file}: {e}")

def process_single_rotation(
    material: str,
    structure: str,
    angle: float,
    idx: int,
    axis: np.ndarray,
    crystal: CrystalStructure,
    out_xyz_dir: Union[str, Path],
    out_img_dir: Union[str, Path],
    out_ann_dir: Union[str, Path]
) -> str:
    """Process a single rotation of the crystal structure.

    Args:
        material: Material name
        structure: Structure name
        angle: Rotation angle in degrees
        idx: Rotation index
        axis: Rotation axis
        crystal: Original CrystalStructure object
        out_xyz_dir: Directory for XYZ files
        out_img_dir: Directory for image files
        out_ann_dir: Directory for annotation files

    Returns:
        Status message

    Raises:
        FileError: If there's an error saving files
    """
    rotated_coords = rotate_coords(crystal.coords, angle, axis)
    rotated_crystal = CrystalStructure(
        crystal.n_atoms,
        f"Rotated by {angle:.2f} degrees about axis {axis}",
        crystal.atoms,
        rotated_coords
    )
    
    # Write rotated XYZ file
    out_xyz_file = Path(out_xyz_dir) / f"{material}_{structure}_rot{idx}.xyz"
    write_xyz(out_xyz_file, rotated_crystal)
    
    # Save rotated image
    fig = create_3d_plot(rotated_crystal)
    out_img_file = Path(out_img_dir) / f"{material}_{structure}_rot{idx}.jpg"
    fig.savefig(out_img_file, dpi=64, pad_inches=0)
    plt.close(fig)
    
    # Generate annotation
    ann_file = Path(out_ann_dir) / f"{material}_{structure}_rot{idx}_annotation.txt"
    generate_annotation(material, structure, rotated_crystal, ann_file)
    
    return f"Processed {material} {structure} rotation {idx}"

def process_file(
    material: str,
    structure: str,
    angle: float,
    num_axes: int,
    input_dir: Union[str, Path] = "initial_data",
    output_dir: Union[str, Path] = "multimodal_dataset_2"
) -> None:
    """Process a single XYZ file by generating rotated copies.

    Args:
        material: Material name
        structure: Structure name
        angle: Rotation angle in degrees
        num_axes: Number of rotation axes to generate
        input_dir: Input directory containing original XYZ files
        output_dir: Output directory for generated files

    Raises:
        FileError: If there's an error with file operations
        ValidationError: If the input file is invalid
    """
    file_name = f"{material}_{structure}.xyz"
    input_file = Path(input_dir) / material / structure / file_name
    
    if not input_file.exists():
        logger.error(f"File not found: {input_file}")
        return
    
    try:
        crystal = load_xyz(input_file)
    except Exception as e:
        logger.error(f"Error loading XYZ file {input_file}: {e}")
        return
    
    out_xyz_dir = Path(output_dir) / material / structure / "xyz"
    out_img_dir = Path(output_dir) / material / structure / "images"
    out_ann_dir = Path(output_dir) / material / structure / "annotations"
    
    for dir_path in [out_xyz_dir, out_img_dir, out_ann_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save original structure
    orig_xyz_file = out_xyz_dir / f"{material}_{structure}_rot0.xyz"
    write_xyz(orig_xyz_file, crystal)
    
    fig_orig = create_3d_plot(crystal)
    orig_img_file = out_img_dir / f"{material}_{structure}_rot0.jpg"
    fig_orig.savefig(orig_img_file, dpi=64, pad_inches=0)
    plt.close(fig_orig)
    
    orig_ann_file = out_ann_dir / f"{material}_{structure}_rot0_annotation.txt"
    generate_annotation(material, structure, crystal, orig_ann_file)
    
    logger.info(f"Saved original {material} {structure} as rot0")
    
    # Process rotations in parallel
    axes = fibonacci_sphere(num_axes)
    with ProcessPoolExecutor() as executor:
        futures = []
        for idx, axis in enumerate(axes, start=1):
            futures.append(
                executor.submit(
                    process_single_rotation,
                    material,
                    structure,
                    angle,
                    idx,
                    axis,
                    crystal,
                    out_xyz_dir,
                    out_img_dir,
                    out_ann_dir
                )
            )
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {material} {structure}"):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in processing rotation: {e}")

def main(angle: float, num_axes: int) -> None:
    """Main execution function.

    Args:
        angle: Rotation angle in degrees
        num_axes: Number of rotation axes to generate
    """
    materials = ['Au', 'Ag', 'PbS', 'ZnO']
    structures = ['R6', 'R7', 'R8', 'R9', 'R10']
    
    for material in materials:
        for structure in structures:
            try:
                process_file(material, structure, angle, num_axes)
            except Exception as e:
                logger.error(f"Error processing {material} {structure}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate rotated XYZ files, images, and annotation text files using uniformly sampled rotation axes."
    )
    parser.add_argument(
        "--angle",
        type=float,
        required=True,
        help="Rotation angle in degrees to apply about each axis"
    )
    parser.add_argument(
        "--num_axes",
        type=int,
        required=True,
        help="Number of uniformly distributed rotation axes to sample (plus the original file)"
    )
    args = parser.parse_args()
    
    main(args.angle, args.num_axes)
