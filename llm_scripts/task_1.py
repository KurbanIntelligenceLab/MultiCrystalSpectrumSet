"""
Multi-Crystal Spectrum Set Annotation Pipeline

This module provides functionality for evaluating LLM-generated annotations on crystal structures.
It supports multiple models through OpenAI and OpenRouter APIs.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import random
import requests
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from openai import OpenAI
from rouge_score import rouge_scorer
from tqdm import tqdm

# Type aliases
JSONDict = Dict[str, Any]
AnnotationMetrics = Dict[str, Union[float, Dict[str, Any]]]
SampleDict = Dict[str, Union[str, bool, Path]]

@dataclass
class ModelConfig:
    """Configuration for an LLM model."""
    name: str
    use_openrouter: bool
    max_tokens: int = 1000
    temperature: float = 0.2

class AnnotationError(Exception):
    """Base exception for annotation-related errors."""
    pass

class ImageEncodingError(AnnotationError):
    """Raised when there's an error encoding an image."""
    pass

class XYZFileError(AnnotationError):
    """Raised when there's an error reading an XYZ file."""
    pass

class APIError(AnnotationError):
    """Raised when there's an error with the API call."""
    pass

class MultiModelAnnotationPipeline:
    """Pipeline for evaluating LLM-generated annotations on crystal structures."""

    def __init__(
        self,
        api_keys: Dict[str, str],
        model_config: ModelConfig,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None
    ) -> None:
        """Initialize the annotation pipeline.

        Args:
            api_keys: Dictionary containing API keys for different services
            model_config: Configuration for the model to use
            site_url: Optional URL for the site
            site_name: Optional name of the site
        """
        self.model_config = model_config
        self.api_keys = api_keys

        if model_config.use_openrouter:
            self.headers = {
                "Authorization": f"Bearer {api_keys['openrouter']}",
                "Content-Type": "application/json"
            }
        else:
            self.client = OpenAI(api_key=api_keys['openai'])

        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Convert image to base64 string.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded string of the image

        Raises:
            ImageEncodingError: If there's an error encoding the image
        """
        try:
            with Image.open(image_path) as img:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            raise ImageEncodingError(f"Error encoding image {image_path}: {e}")

    def _format_xyz_data(self, xyz_path: Union[str, Path]) -> str:
        """Read and format XYZ file content.

        Args:
            xyz_path: Path to the XYZ file

        Returns:
            Formatted XYZ content

        Raises:
            XYZFileError: If there's an error reading the XYZ file
        """
        try:
            with open(xyz_path, 'r') as f:
                xyz_content = f.read()
                # Skip first two lines
                xyz_content = xyz_content.split('\n', 2)[2]
            return xyz_content
        except Exception as e:
            raise XYZFileError(f"Error reading XYZ file {xyz_path}: {e}")

    def create_messages(
        self,
        image_path: Union[str, Path],
        xyz_path: Union[str, Path],
        material: str,
        structure: str,
        is_original: bool
    ) -> List[Dict[str, Any]]:
        """Create messages for the LLM API.

        Args:
            image_path: Path to the image file
            xyz_path: Path to the XYZ file
            material: Material name
            structure: Structure name
            is_original: Whether this is the original structure

        Returns:
            List of message dictionaries for the API
        """
        image_b64 = self._encode_image(image_path)
        xyz_content = self._format_xyz_data(xyz_path)

        system_content = (
            "You are a materials science expert specializing in analyzing crystal structures. "
            "Your task is to generate a detailed annotation for the provided structure based on both visual and atomic coordinate data. "
            "The annotation must strictly follow the format below:\n\n"
            "Material: [Material]\n"
            "Structure Prototype: [Structure]\n"
            "Atomic Count: [Total atoms]\n"
            "Cell Volume: [Cell volume] Å³\n"
            "Lattice Parameters: a = [a] Å, b = [b] Å, c = [c] Å\n"
            "Average NN Distance: [NN distance] Å\n"
            "Rotation Axis: [Rotation Axis]\n"
            "Density: [Density] g/cm³\n\n"
            "Summary:\n"
            "This [Material] crystal structure with prototype [Structure] contains [Total atoms] atoms and has a cell volume of [Cell volume] Å³. "
            "Its lattice is characterized by parameters a = [a] Å, b = [b] Å, and c = [c] Å, suggesting an orthogonal geometry. "
            "The average nearest neighbor distance is [NN distance] Å, and the density is estimated to be [Density] g/cm³.\n\n"
            "Only output the annotation as specified above without any additional text."
        )

        user_text = (
            f"Analyze this crystal structure.\n"
            f"Material: {material}\n"
            f"Structure Prototype: {structure}\n"
            f"This is {'the original structure (no rotation)' if is_original else 'a rotated structure'}.\n\n"
            "Here is the XYZ structural data:\n"
            "```\n"
            f"{xyz_content}\n"
            "```\n\n"
            "Based on the structural data and the image, generate the annotation following the exact format provided."
        )

        return [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "low"}}
                ]
            }
        ]

    def generate_annotation(
        self,
        image_path: Union[str, Path],
        xyz_path: Union[str, Path],
        material: str,
        structure: str,
        is_original: bool,
        max_retries: int = 3
    ) -> str:
        """Generate an annotation using the selected model.

        Args:
            image_path: Path to the image file
            xyz_path: Path to the XYZ file
            material: Material name
            structure: Structure name
            is_original: Whether this is the original structure
            max_retries: Maximum number of retry attempts

        Returns:
            Generated annotation text

        Raises:
            APIError: If there's an error with the API call
        """
        messages = self.create_messages(image_path, xyz_path, material, structure, is_original)

        for attempt in range(max_retries):
            try:
                if self.model_config.use_openrouter:
                    payload = {
                        "model": self.model_config.name,
                        "messages": messages,
                        "max_tokens": self.model_config.max_tokens,
                        "temperature": self.model_config.temperature
                    }
                    response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers=self.headers,
                        json=payload
                    )
                    if response.status_code != 200:
                        raise APIError(f"OpenRouter API Error: {response.text}")
                    response_json = response.json()
                    if 'error' in response_json:
                        raise APIError(f"OpenRouter API Error: {response_json['error']}")
                    return response_json['choices'][0]['message']['content']
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_config.name,
                        messages=messages,
                        max_tokens=self.model_config.max_tokens,
                        temperature=self.model_config.temperature
                    )
                    return response.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    raise APIError(f"Failed to generate annotation after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

    def extract_properties(self, text: str) -> Dict[str, Any]:
        """Extract numerical properties from an annotation text.

        Args:
            text: Annotation text to parse

        Returns:
            Dictionary of extracted properties
        """
        props = {}
        try:
            patterns = {
                'atomic_count': (r'Atomic Count:\s*(\d+)', int),
                'cell_volume': (r'Cell Volume:\s*([\d.]+)\s*Å³', float),
                'lattice_a': (r'Lattice Parameters:\s*a\s*=\s*([\d.]+)\s*Å', float),
                'lattice_b': (r'b\s*=\s*([\d.]+)\s*Å', float),
                'lattice_c': (r'c\s*=\s*([\d.]+)\s*Å', float),
                'avg_nn': (r'Average NN Distance:\s*([\d.]+)\s*Å', float),
                'density': (r'Density:\s*([\d.]+)\s*g/cm³', float),
                'material': (r'Material:\s*(\S+)', str),
                'structure': (r'Structure Prototype:\s*(\S+)', str)
            }

            for key, (pattern, converter) in patterns.items():
                match = re.search(pattern, text)
                if match:
                    props[key] = converter(match.group(1))

        except Exception as e:
            print(f"Error extracting properties: {e}")
        return props

    def compute_metrics(self, reference: str, candidate: str) -> AnnotationMetrics:
        """Compute text similarity and numerical differences between annotations.

        Args:
            reference: Reference annotation text
            candidate: Candidate annotation text

        Returns:
            Dictionary containing computed metrics
        """
        try:
            smoothing = SmoothingFunction().method1
            bleu_score = sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothing)
            rouge_scores = self.rouge_scorer.score(reference, candidate)

            ref_props = self.extract_properties(reference)
            cand_props = self.extract_properties(candidate)

            numerical_metrics = {}
            for key in ['atomic_count', 'cell_volume', 'lattice_a', 'lattice_b', 'lattice_c', 'avg_nn', 'density']:
                if key in ref_props and key in cand_props and ref_props[key] is not None and cand_props[key] is not None:
                    diff = abs(ref_props[key] - cand_props[key])
                    perc = diff / ref_props[key] * 100 if ref_props[key] != 0 else None
                    numerical_metrics[key] = {
                        'reference': ref_props[key],
                        'candidate': cand_props[key],
                        'absolute_error': diff,
                        'percent_error': perc
                    }
                else:
                    numerical_metrics[key] = None

            material_match = (ref_props.get('material', '').lower() == cand_props.get('material', '').lower())
            structure_match = (ref_props.get('structure', '').lower() == cand_props.get('structure', '').lower())
            numerical_metrics['material_match'] = material_match
            numerical_metrics['structure_match'] = structure_match

            return {
                'bleu': bleu_score,
                'rouge1': rouge_scores['rouge1'].fmeasure,
                'rouge2': rouge_scores['rouge2'].fmeasure,
                'rougeL': rouge_scores['rougeL'].fmeasure,
                'numerical': numerical_metrics,
                'reference_values': ref_props,
                'generated_values': cand_props
            }
        except Exception as e:
            raise AnnotationError(f"Error computing metrics: {e}")

def build_samples_from_dataset(base_dir: Union[str, Path]) -> List[SampleDict]:
    """Build sample list from dataset directory structure.

    Args:
        base_dir: Base directory containing the dataset

    Returns:
        List of sample dictionaries
    """
    samples = []
    base_path = Path(base_dir)

    for material in base_path.iterdir():
        if not material.is_dir():
            continue
        for structure in material.iterdir():
            if not structure.is_dir():
                continue

            annotations_dir = structure / "annotations"
            images_dir = structure / "images"
            xyz_dir = structure / "xyz"

            if not annotations_dir.exists():
                continue

            for annotation_file in annotations_dir.glob("*_annotation.txt"):
                base_name = annotation_file.stem.replace("_annotation", "")
                image_file = images_dir / f"{base_name}.jpg"
                xyz_file = xyz_dir / f"{base_name}.xyz"
                is_original = "rot0" in base_name

                sample = {
                    'annotation': annotation_file,
                    'image_path': image_file,
                    'xyz_path': xyz_file,
                    'material': material.name,
                    'structure': structure.name,
                    'is_original': is_original
                }
                samples.append(sample)

    return samples

def process_samples(
    pipeline: MultiModelAnnotationPipeline,
    samples: List[SampleDict],
    output_dir: Union[str, Path],
    split_name: str = "test"
) -> List[Dict[str, Any]]:
    """Process a list of samples through the annotation pipeline.

    Args:
        pipeline: Annotation pipeline instance
        samples: List of samples to process
        output_dir: Directory to save results
        split_name: Name of the data split

    Returns:
        List of results dictionaries
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results = []

    for sample in tqdm(samples, desc=f"Processing samples for {split_name}"):
        try:
            generated_annotation = pipeline.generate_annotation(
                sample['image_path'],
                sample['xyz_path'],
                sample['material'],
                sample['structure'],
                sample['is_original']
            )

            with open(sample['annotation'], 'r', encoding='utf-8') as f:
                reference_annotation = f.read()

            metrics = pipeline.compute_metrics(reference_annotation, generated_annotation)
            result = {
                "material": sample['material'],
                "structure": sample['structure'],
                "is_original": sample['is_original'],
                "reference_annotation": reference_annotation,
                "generated_annotation": generated_annotation,
                "metrics": metrics
            }
            results.append(result)
        except Exception as e:
            print(f"Error processing sample {sample}: {e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = pipeline.model_config.name.replace('/', '_')
    output_file = output_path / f"annotations_{model_name_safe}_{split_name}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def select_subset(samples: List[SampleDict], count_per_group: int) -> List[SampleDict]:
    """Select a subset of samples from each material-structure group.

    Args:
        samples: List of all samples
        count_per_group: Maximum number of samples to keep per group

    Returns:
        Subset of samples
    """
    groups: Dict[Tuple[str, str], List[SampleDict]] = {}
    for sample in samples:
        key = (sample['material'], sample['structure'])
        groups.setdefault(key, []).append(sample)

    subset_samples = []
    for group in groups.values():
        if len(group) <= count_per_group:
            subset_samples.extend(group)
        else:
            subset_samples.extend(random.sample(group, count_per_group))
    
    return subset_samples

def main() -> None:
    """Main execution function."""
    api_keys = {
        'openrouter': os.environ.get('OPENROUTER_API_KEY'),
    }

    models_to_test = [
        ModelConfig(name="deepseek/deepseek-chat", use_openrouter=True),
        ModelConfig(name="deepseek/deepseek-r1", use_openrouter=True),
        ModelConfig(name="x-ai/grok-2-vision-1212", use_openrouter=True),
        ModelConfig(name="openai/gpt-4o", use_openrouter=True),
        ModelConfig(name="openai/o1", use_openrouter=True),
        ModelConfig(name="google/gemma-3-27b-it", use_openrouter=True),
        ModelConfig(name="google/gemini-2.0-flash-001", use_openrouter=True),
        ModelConfig(name="anthropic/claude-3.5-haiku", use_openrouter=True),
        ModelConfig(name="anthropic/claude-3.5-sonnet", use_openrouter=True)
    ]

    dataset_base_dir = Path("data/MultiCrystalSpectrumSet")
    samples = build_samples_from_dataset(dataset_base_dir)
    print(f"Found {len(samples)} total samples.")

    subset_samples = select_subset(samples, 10)
    print(f"Using {len(subset_samples)} samples for evaluation.")

    for model_config in models_to_test:
        print(f"\nProcessing with model: {model_config.name}")
        pipeline = MultiModelAnnotationPipeline(
            api_keys=api_keys,
            model_config=model_config
        )

        model_output_dir = Path("annotation_results") / model_config.name.replace('/', '_')
        try:
            results = process_samples(pipeline, subset_samples, model_output_dir, split_name="test")
        except Exception as e:
            print(f"Error processing samples: {e}")

if __name__ == "__main__":
    main()