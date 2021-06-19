# Torch Pitch Shift

Pitch-shift audio clips quickly with PyTorch (CUDA supported), with additional utilities to search for efficient pitch-shift transformations!

[View on PyPI](https://pypi.org/project/torch-pitch-shift/)

[![Publish to PyPI](https://github.com/KentoNishi/torch_pitch_shift/actions/workflows/publish.yaml/badge.svg)](https://github.com/KentoNishi/torch_pitch_shift/actions/workflows/publish.yaml)

## About

This package includes two main features:
* Pitch-shift audio clips quickly to using PyTorch (with CUDA support)
* Calculate efficient pitch-shift targets (useful for augmentation, where speed is more important than precise pitch-shifts)

## Installation
```bash
pip install torch_pitch_shift
```

## Usage

### Example
Check out [example.py](./example.py) to see `torch_pitch_shift` in action!

## Documentation
Documentation is built into the class and function docstrings. If anyone wants to properly document the package, please feel free to contribute!
