"""
Multi-Crystal Spectrum Set Generation Pipeline

This module provides functionality for generating crystal structures using LLMs.
It supports multiple models through OpenAI and OpenRouter APIs.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import random
import requests
from PIL import Image
from openai import OpenAI
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crystal_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("crystal_pipeline")

# Type aliases
JSONDict = Dict[str, Any]
SampleDict = Dict[str, Union[str, bool, Path]]
XYZContent = str

@dataclass
class ModelConfig:
    """Configuration for an LLM model."""
    name: str
    use_openrouter: bool
    max_tokens: int = 6500
    temperature: float = 0.2

class CrystalGenerationError(Exception):
    """Base exception for crystal generation errors."""
    pass

class FileError(CrystalGenerationError):
    """Raised when there's an error with file operations."""
    pass

class APIError(CrystalGenerationError):
    """Raised when there's an error with the API call."""
    pass

class ValidationError(CrystalGenerationError):
    """Raised when there's an error validating the structure."""
    pass

def save_xyz_file(xyz_content: XYZContent, file_path: Union[str, Path]) -> None:
    """Save XYZ content to a file.

    Args:
        xyz_content: The XYZ content to save
        file_path: Path where to save the file

    Raises:
        FileError: If there's an error saving the file
    """
    try:
        with open(file_path, 'w') as f:
            f.write(xyz_content)
        logger.info(f"XYZ file saved to: {file_path}")
    except Exception as e:
        raise FileError(f"Error saving XYZ file: {e}")

def convert_structured_to_xyz(structured_output: JSONDict) -> XYZContent:
    """Convert structured output to XYZ format.

    Args:
        structured_output: Dictionary containing structured output

    Returns:
        XYZ formatted string

    Raises:
        ValidationError: If the structured output is invalid
    """
    try:
        num_atoms = structured_output.get("num_atoms")
        atoms = structured_output.get("atoms", [])
        xyz_lines = [str(num_atoms), ""]
        
        for atom in atoms:
            element = atom.get("element")
            try:
                x = float(atom.get("x"))
                y = float(atom.get("y"))
                z = float(atom.get("z"))
                xyz_lines.append(f"{element} {x:.2f} {y:.2f} {z:.2f}")
            except (ValueError, TypeError):
                xyz_lines.append(f"{element} {atom.get('x')} {atom.get('y')} {atom.get('z')}")
        
        return "\n".join(xyz_lines)
    except Exception as e:
        raise ValidationError(f"Error converting structured output to XYZ: {e}")

def format_xyz_to_2decimals(xyz_str: XYZContent) -> XYZContent:
    """Format XYZ coordinates to 2 decimal places.

    Args:
        xyz_str: Input XYZ string

    Returns:
        Formatted XYZ string
    """
    lines = xyz_str.strip().splitlines()
    if len(lines) < 3:
        return xyz_str
        
    formatted_lines = [lines[0], lines[1]]
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            formatted_lines.append(line)
        else:
            element = parts[0]
            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                formatted_lines.append(f"{element} {x:.2f} {y:.2f} {z:.2f}")
            except ValueError:
                formatted_lines.append(line)
    return "\n".join(formatted_lines)

def parse_xyz(xyz_str: XYZContent) -> List[Tuple[str, float, float, float]]:
    """Parse XYZ format string into a list of atoms with coordinates.

    Args:
        xyz_str: XYZ format string

    Returns:
        List of tuples containing (element, x, y, z)

    Raises:
        ValidationError: If the XYZ format is invalid
    """
    try:
        lines = xyz_str.strip().splitlines()
        if len(lines) < 2:
            raise ValidationError("Invalid XYZ format: not enough lines")
            
        try:
            n_declared = int(lines[0].strip())
        except ValueError:
            raise ValidationError(f"Invalid XYZ format: first line should be number of atoms, got: {lines[0]}")
        
        n_actual = len(lines) - 2
        if n_declared != n_actual:
            logger.warning(
                f"Mismatch in declared number of atoms ({n_declared}) and actual number of lines ({n_actual}). Using available lines."
            )
            
        atoms = []
        for i, line in enumerate(lines[2:2+n_actual]):
            parts = line.split()
            if len(parts) < 4:
                logger.warning(f"Invalid atom format at line {i+3}: {line}")
                continue
            try:
                element = parts[0]
                x, y, z = map(float, parts[1:4])
                atoms.append((element, x, y, z))
            except ValueError:
                logger.warning(f"Invalid coordinates at line {i+3}: {line}")
                continue
        return atoms
    except Exception as e:
        raise ValidationError(f"Error parsing XYZ: {e}")

def is_valid_structure_xyz(xyz_str: XYZContent, min_distance: float = 0.5) -> Tuple[bool, Optional[str]]:
    """Check if every pair of atoms in the XYZ file is at least min_distance apart.

    Args:
        xyz_str: XYZ format string
        min_distance: Minimum allowed distance between atoms

    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    try:
        atoms = parse_xyz(xyz_str)
        if not atoms:
            return False, "Invalid XYZ format or no atoms found"
        
        for i in range(len(atoms)):
            for j in range(i+1, len(atoms)):
                dx = atoms[i][1] - atoms[j][1]
                dy = atoms[i][2] - atoms[j][2]
                dz = atoms[i][3] - atoms[j][3]
                d = (dx**2 + dy**2 + dz**2) ** 0.5
                if d < min_distance:
                    return False, f"Atoms too close: {atoms[i][0]} and {atoms[j][0]} at distance {d:.2f}Å"
        return True, None
    except Exception as e:
        return False, str(e)

def compute_rmsd_xyz(target_xyz_str: XYZContent, generated_xyz_str: XYZContent) -> Tuple[Optional[float], Optional[str]]:
    """Compute the root mean square deviation between two XYZ structures.

    Args:
        target_xyz_str: Target XYZ structure
        generated_xyz_str: Generated XYZ structure

    Returns:
        Tuple of (rmsd, error_message_if_any)
    """
    try:
        atoms_target = parse_xyz(target_xyz_str)
        atoms_generated = parse_xyz(generated_xyz_str)
        
        if not atoms_target:
            return None, "Invalid target XYZ format"
        if not atoms_generated:
            return None, "Invalid generated XYZ format"
        
        n_target = len(atoms_target)
        n_generated = len(atoms_generated)
        
        if n_target == n_generated:
            for a_t, a_g in zip(atoms_target, atoms_generated):
                if a_t[0] != a_g[0]:
                    return None, f"Element mismatch: {a_t[0]} vs {a_g[0]}"
        
        squared_diff = 0.0
        for i in range(n_target):
            a_t = atoms_target[i]
            if i < n_generated:
                a_g = atoms_generated[i]
            else:
                a_g = (a_t[0], 0.0, 0.0, 0.0)
            dx = a_t[1] - a_g[1]
            dy = a_t[2] - a_g[2]
            dz = a_t[3] - a_g[3]
            squared_diff += dx*dx + dy*dy + dz*dz

        rmsd = (squared_diff / n_target) ** 0.5
        return rmsd, None
    except Exception as e:
        return None, str(e)

class MultiModelCrystalGenerationPipeline:
    """Pipeline for generating crystal structures using LLMs."""

    def __init__(
        self,
        api_keys: Dict[str, str],
        model_config: ModelConfig,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None
    ) -> None:
        """Initialize the crystal generation pipeline.

        Args:
            api_keys: Dictionary containing API keys
            model_config: Configuration for the model to use
            site_url: Optional URL for the site
            site_name: Optional name of the site
        """
        self.model_config = model_config
        self.api_keys = api_keys
        self._openai_client = None

        if model_config.use_openrouter:
            self.headers = {
                "Authorization": f"Bearer {api_keys['openrouter']}",
                "Content-Type": "application/json"
            }

    @property
    def openai_client(self) -> Optional[OpenAI]:
        """Lazy-loaded OpenAI client."""
        if not self._openai_client and not self.model_config.use_openrouter and 'openai' in self.api_keys:
            self._openai_client = OpenAI(api_key=self.api_keys['openai'])
        return self._openai_client

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image as base64.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded string

        Raises:
            FileError: If there's an error encoding the image
        """
        if not os.path.exists(image_path):
            raise FileError(f"Image file not found: {image_path}")
            
        try:
            with Image.open(image_path) as img:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            raise FileError(f"Error encoding image {image_path}: {e}")

    def _read_xyz(self, xyz_path: Union[str, Path]) -> XYZContent:
        """Read XYZ file.

        Args:
            xyz_path: Path to the XYZ file

        Returns:
            XYZ content as string

        Raises:
            FileError: If there's an error reading the file
        """
        if not os.path.exists(xyz_path):
            raise FileError(f"XYZ file not found: {xyz_path}")
            
        try:
            with open(xyz_path, 'r') as f:
                return f.read()
        except Exception as e:
            raise FileError(f"Error reading XYZ file {xyz_path}: {e}")

    def create_messages(
        self,
        image_path: Union[str, Path],
        input_xyz_paths: Dict[str, Union[str, Path]],
        material: str,
        rotation: str
    ) -> List[Dict[str, Any]]:
        """Create messages for the model.

        Args:
            image_path: Path to the image file
            input_xyz_paths: Dictionary of input XYZ file paths
            material: Material name
            rotation: Rotation identifier

        Returns:
            List of message dictionaries

        Raises:
            ValidationError: If there's an error validating the input
        """
        try:
            image_b64 = self._encode_image(image_path)
            
            input_xyz_contents = ""
            for struct, path in input_xyz_paths.items():
                xyz_content = self._read_xyz(path)
                if not parse_xyz(xyz_content):
                    raise ValidationError(f"Invalid XYZ content in {path}")
                    
                xyz_content = format_xyz_to_2decimals(xyz_content)
                input_xyz_contents += f"### {struct} structure\n{xyz_content}\n\n"

            system_content = (
                "You are a materials science expert specializing in crystal structure generation. "
                "Your task is to generate the complete XYZ file for a crystal structure based on the provided input structures and an image. "
                "Ensure the output is valid (no atom closer than 0.5 Å) and output it as JSON:\n"
                '{ "num_atoms": ..., "atoms": [ { "element": "...", "x": "...", "y": "...", "z": "..." }, ... ] }'
            )

            user_text = (
                f"Material: {material}\n"
                f"Rotation: {rotation}\n\n"
                f"{input_xyz_contents}\n"
                "Also, refer to the image to generate the final rotated structure."
            )

            return [
                {"role": "system", "content": system_content},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "low"}}
                ]}
            ]
        except Exception as e:
            raise ValidationError(f"Error creating messages: {e}")

    def generate(
        self,
        image_path: Union[str, Path],
        input_xyz_paths: Dict[str, Union[str, Path]],
        material: str,
        rotation: str
    ) -> XYZContent:
        """Generate crystal structure.

        Args:
            image_path: Path to the image file
            input_xyz_paths: Dictionary of input XYZ file paths
            material: Material name
            rotation: Rotation identifier

        Returns:
            Generated XYZ content

        Raises:
            APIError: If there's an error with the API call
            ValidationError: If there's an error validating the structure
        """
        messages = self.create_messages(image_path, input_xyz_paths, material, rotation)
        last_error = None
        
        for attempt in range(3):
            try:
                logger.info(f"[{self.model_config.name}] Generation attempt {attempt+1}")
                
                if self.model_config.use_openrouter:
                    response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers=self.headers,
                        json={
                            "model": self.model_config.name,
                            "messages": messages,
                            "max_tokens": self.model_config.max_tokens,
                            "temperature": self.model_config.temperature,
                            "response_format": {"type": "json_object"}
                        },
                        timeout=120
                    )
                    
                    if response.status_code != 200:
                        raise APIError(f"API error: {response.status_code}, {response.text}")
                        
                    raw_content = response.json()['choices'][0]['message']['content']
                else:
                    if not self.openai_client:
                        raise APIError("OpenAI client not configured")
                        
                    response = self.openai_client.chat.completions.create(
                        model=self.model_config.name,
                        messages=messages,
                        max_tokens=self.model_config.max_tokens,
                        temperature=self.model_config.temperature,
                        response_format={"type": "json_object"}
                    )
                    raw_content = response.choices[0].message.content

                try:
                    structured_output = json.loads(raw_content)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON received: {e}")
                    json_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', raw_content)
                    if json_match:
                        json_content = json_match.group(1)
                        json_content = re.sub(r'(\w+)(?=\s*:)', r'"\1"', json_content)
                        json_content = re.sub(r'\'', r'"', json_content)
                        structured_output = json.loads(json_content)
                    else:
                        raise
                
                xyz_content = convert_structured_to_xyz(structured_output)
                
                valid, reason = is_valid_structure_xyz(xyz_content)
                if not valid:
                    logger.warning(f"[{self.model_config.name}] Generated invalid structure: {reason}")
                    if attempt == 2:
                        return xyz_content
                    continue
                
                return xyz_content
                
            except Exception as e:
                last_error = e
                logger.warning(f"[{self.model_config.name}] Attempt {attempt+1} failed: {e}")
                time.sleep(5 * (attempt + 1))
        
        raise APIError(f"{self.model_config.name} failed to generate a structure after retries: {last_error}")

def process_model_crystal(
    model_config: ModelConfig,
    sample: SampleDict,
    api_keys: Dict[str, str],
    output_dir: Union[str, Path] = "crystal_multi_model_results"
) -> Dict[str, Any]:
    """Process a single crystal generation task.

    Args:
        model_config: Model configuration
        sample: Sample dictionary
        api_keys: API keys
        output_dir: Output directory

    Returns:
        Dictionary containing results
    """
    model_name = model_config.name
    logger.info(f"Processing {model_name} for {sample['material']} rotation {sample['rotation']}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_output_dir = output_path / model_name.replace('/', '_')
    model_output_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    result = {
        "model": model_name,
        "material": sample["material"],
        "rotation": sample["rotation"],
        "validity": False,
        "min_distance": None,
        "rmsd": None,
        "match_rate": None,
        "generated_xyz": None,
        "error": None,
        "processing_time": None
    }
    
    try:
        pipeline = MultiModelCrystalGenerationPipeline(api_keys, model_config)
        
        generated_xyz = pipeline.generate(
            image_path=sample["target_image"],
            input_xyz_paths=sample["input_xyz"],
            material=sample["material"],
            rotation=sample["rotation"]
        )
        
        with open(sample["target_xyz"], 'r') as f:
            target_xyz = f.read()
        
        validity, reason = is_valid_structure_xyz(generated_xyz)
        rmsd, rmsd_error = compute_rmsd_xyz(target_xyz, generated_xyz)
        match_rate = max(0, 100 * (1 - rmsd)) if rmsd is not None else None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        xyz_filename = f"{sample['material']}_rot{sample['rotation']}_{timestamp}.xyz"
        xyz_path = model_output_dir / xyz_filename
        save_xyz_file(generated_xyz, xyz_path)
        
        result.update({
            "validity": validity,
            "min_distance": reason if not validity else None,
            "rmsd": rmsd,
            "rmsd_error": rmsd_error,
            "match_rate": match_rate,
            "generated_xyz": generated_xyz,
            "generated_xyz_path": str(xyz_path),
            "processing_time": time.time() - start_time
        })
        
    except Exception as e:
        logger.error(f"Error with model {model_name}: {e}", exc_info=True)
        result.update({
            "error": str(e),
            "processing_time": time.time() - start_time
        })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"result_{sample['material']}_rot{sample['rotation']}_{timestamp}.json"
    result_path = model_output_dir / result_filename
    
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

def build_crystal_generation_sample(
    dataset_base_dir: Union[str, Path],
    material: str = "Au"
) -> SampleDict:
    """Build a sample for crystal generation.

    Args:
        dataset_base_dir: Base directory containing the dataset
        material: Material name

    Returns:
        Sample dictionary

    Raises:
        FileError: If required files are not found
        ValidationError: If the sample data is invalid
    """
    logger.info(f"Building sample for {material}")
    
    base_path = Path(dataset_base_dir)
    if not base_path.exists():
        raise FileError(f"Dataset directory not found: {dataset_base_dir}")
    
    material_dir = base_path / material
    if not material_dir.exists():
        raise FileError(f"Material directory not found: {material_dir}")
    
    target_folder = material_dir / "R9"
    if not target_folder.exists():
        raise FileError(f"Target rotation folder not found: {target_folder}")
        
    xyz_target_dir = target_folder / "xyz"
    images_target_dir = target_folder / "images"
    
    if not xyz_target_dir.exists() or not images_target_dir.exists():
        raise FileError(f"Target xyz or images directory not found")
    
    xyz_files = list(xyz_target_dir.glob("*.xyz"))
    if not xyz_files:
        raise ValidationError(f"No target xyz files found in R9 for material {material}")
    
    target_xyz_file = random.choice(xyz_files)
    logger.info(f"Selected target file: {target_xyz_file}")
    
    match = re.search(r'rot(\d+)', target_xyz_file.name)
    if not match:
        match = re.search(r'_(\d+)\.', target_xyz_file.name)
        if not match:
            raise ValidationError(f"Rotation number not found in target file name: {target_xyz_file}")
    
    rotation = match.group(1)
    logger.info(f"Detected rotation: {rotation}")
    
    target_xyz_path = target_xyz_file
    
    possible_extensions = [".jpg", ".png", ".jpeg"]
    target_image_path = None
    
    for ext in possible_extensions:
        image_file = target_xyz_file.with_suffix(ext)
        if image_file.exists():
            target_image_path = image_file
            break
    
    if not target_image_path:
        raise FileError(f"No matching image found for {target_xyz_file}")
    
    input_structures = ["R6", "R7", "R8", "R10"]
    input_xyz = {}
    missing_inputs = []
    
    for struct in input_structures:
        struct_folder = material_dir / struct
        if not struct_folder.exists():
            missing_inputs.append(struct)
            continue
            
        xyz_dir = struct_folder / "xyz"
        if not xyz_dir.exists():
            missing_inputs.append(struct)
            continue
        
        filename_patterns = [
            f"{material}_{struct}_rot{rotation}.xyz",
            f"{material}_rot{rotation}_{struct}.xyz",
            f"{struct}_rot{rotation}.xyz"
        ]
        
        found = False
        for pattern in filename_patterns:
            file_path = xyz_dir / pattern
            if file_path.exists():
                input_xyz[struct] = file_path
                found = True
                break
        
        if not found:
            for f in xyz_dir.glob("*.xyz"):
                if f"rot{rotation}" in f.name:
                    input_xyz[struct] = f
                    found = True
                    break
        
        if not found:
            missing_inputs.append(struct)
    
    if len(missing_inputs) > 0:
        logger.warning(f"Missing input structures: {', '.join(missing_inputs)}")
        if len(input_xyz) < 2:
            raise ValidationError(f"Not enough input structures found (need at least 2, found {len(input_xyz)})")
    
    sample = {
        "material": material,
        "rotation": rotation,
        "target_xyz": target_xyz_path,
        "target_image": target_image_path,
        "input_xyz": input_xyz
    }
    
    logger.info(f"Sample built successfully for {material} rotation {rotation}")
    return sample

def run_parallel_crystal_generation(
    models_to_test: List[ModelConfig],
    api_keys: Dict[str, str],
    dataset_base_dir: Union[str, Path],
    material: str,
    num_tests: int = 10
) -> List[Dict[str, Any]]:
    """Run parallel crystal generation tests.

    Args:
        models_to_test: List of model configurations
        api_keys: API keys
        dataset_base_dir: Base directory containing the dataset
        material: Material name
        num_tests: Number of tests to run per model

    Returns:
        List of results dictionaries
    """
    logger.info(f"Starting parallel crystal generation for {len(models_to_test)} models, {num_tests} tests per model")
    
    results = []
    samples = []
    
    for _ in range(num_tests):
        try:
            sample = build_crystal_generation_sample(dataset_base_dir, material=material)
            samples.append(sample)
        except Exception as e:
            logger.error(f"Failed to build sample: {e}")
    
    logger.info(f"Built {len(samples)} test samples")
    
    if not samples:
        logger.error("No valid samples could be built, aborting test")
        return []
    
    max_workers = min(len(models_to_test), 5)
    logger.info(f"Using {max_workers} parallel workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        for model_config in models_to_test:
            for sample in samples:
                future = executor.submit(process_model_crystal, model_config, sample, api_keys)
                futures[future] = (model_config.name, sample['rotation'])
        
        for future in as_completed(futures):
            model_name, rotation = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    status = 'Passed' if result.get('validity') else 'Failed'
                    logger.info(f"{status} {model_name} (rot {rotation}): RMSD={result.get('rmsd')}, Match={result.get('match_rate')}%")
            except Exception as e:
                logger.error(f"Error processing {model_name} (rot {rotation}): {e}")
                results.append({
                    "model": model_name,
                    "rotation": rotation,
                    "error": str(e)
                })
    
    return results

def main() -> None:
    """Main execution function."""
    api_keys = {
        'openrouter': os.environ.get('OPENROUTER_API_KEY'),
    }
    
    if not api_keys['openrouter']:
        logger.error("No API keys found. Set OPENROUTER_API_KEY environment variable.")
        exit(1)

    models_to_test = [
        ModelConfig(name="anthropic/claude-3.5-sonnet", use_openrouter=True)
    ]

    dataset_base_dir = Path("data/MultiCrystalSpectrumSet")
    
    if not dataset_base_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_base_dir}")
        exit(1)
    
    materials = ["ZnO"]
    
    for material in materials:
        logger.info(f"Testing material: {material}")
        results = run_parallel_crystal_generation(models_to_test, api_keys, dataset_base_dir, material, num_tests=1)
        
        success_count = sum(1 for r in results if r.get("validity") and r.get("rmsd") is not None)
        logger.info(f"Completed testing for {material}: {success_count}/{len(results)} successful generations")

if __name__ == "__main__":
    main()