# MultiCrystalSpectrumSet Framework

This repository contains the implementation and dataset of:

**Beyond Atomic Geometry Representations in Materials Science: A Human-in-the-Loop Multimodal Framework** by Can Polat, Erchin Serpedin, Mustafa Kurban, and Hasan Kurban.

*Presented at ICML 2025*

## Overview

The MultiCrystalSpectrumSet Framework is a comprehensive toolkit for generating, processing, and analyzing crystal structures with multimodal representations. It combines atomic geometry data with visual representations and natural language annotations to create a rich dataset for materials science research.

## Dataset

The MultiCrystalSpectrumSet is available at: [Framework](https://figshare.com/s/6c930e6e9bb0779eb53b)

The dataset includes:
- XYZ files containing atomic coordinates
- Visual representations of crystal structures
- Natural language annotations describing structural properties
- Rotation variants of crystal structures

## Installation

1. Clone the repository:

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
MultiCrystalSpectrumSet/
├── data/
│   └── MultiCrystalSpectrumSet/  # Dataset directory
├── llm_scripts/
│   ├── task_1.py                 # Annotation generation pipeline
│   └── task_2.py                 # Crystal structure generation pipeline
├── generate_dataset.py           # Dataset generation script
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## Usage

### Generating the Dataset

To generate the dataset with rotated crystal structures:

```bash
python generate_dataset.py --angle <rotation_angle> --num_axes <number_of_rotation_axes>
```

### Running the Annotation Pipeline

To generate annotations for crystal structures:

```bash
python llm_scripts/task_1.py
```

### Running the Structure Generation Pipeline

To generate crystal structures:

```bash
python llm_scripts/task_2.py
```

## Features

- **Multimodal Representation**: Combines atomic coordinates, visual representations, and natural language descriptions
- **Rotation Generation**: Creates multiple rotated variants of crystal structures
- **Property Computation**: Calculates physical properties like cell volume, lattice parameters, and density
- **LLM Integration**: Uses language models for generating and validating annotations
- **Parallel Processing**: Efficient handling of multiple structures and rotations

## Citation

If you use this framework or dataset in your research, please cite:

```bibtex
@article{polat2025beyond,
  title={Beyond Atomic Geometry Representations in Materials Science: A Human-in-the-Loop Multimodal Framework},
  author={Polat, Can and Kurban, Hasan and Serpedin, Erchin and Kurban, Mustafa},
  journal={arXiv preprint arXiv:2506.00302},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
Data released under CC-BY 4.0 license.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and feedback, please open an issue in the repository or contact the authors.
