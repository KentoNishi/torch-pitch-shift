# Torch Pitch Shift

Pitch-shift audio clips quickly with PyTorch (CUDA supported)! Additional utilities for searching efficient transformations are included.

[View on PyPI](https://pypi.org/project/torch-pitch-shift/) / [View Documentation](https://kentonishi.github.io/torch-pitch-shift/)

[![Publish to PyPI](https://github.com/KentoNishi/torch-pitch-shift/actions/workflows/publish.yaml/badge.svg)](https://github.com/KentoNishi/torch-pitch-shift/actions/workflows/publish.yaml)
[![Run tests](https://github.com/KentoNishi/torch-pitch-shift/actions/workflows/test.yaml/badge.svg)](https://github.com/KentoNishi/torch-pitch-shift/actions/workflows/test.yaml)
[![PyPI version](https://img.shields.io/pypi/v/torch-pitch-shift.svg?style=flat)](https://pypi.org/project/torch-pitch-shift/)
[![Number of downloads from PyPI per month](https://img.shields.io/pypi/dm/torch-pitch-shift.svg?style=flat)](https://pypi.org/project/torch-pitch-shift/)
![Python version support](https://img.shields.io/pypi/pyversions/torch-pitch-shift)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/ambv/black)

## About

This package includes two main features:
* Pitch-shift audio clips quickly using PyTorch (with CUDA support)
* Calculate efficient pitch-shift targets (useful for augmentation, where speed is more important than precise pitch-shifts)

> Also check out [torch-time-stretch](https://github.com/KentoNishi/torch-time-stretch), a sister project for time-stretching.

## Installation
```bash
pip install torch-pitch-shift
```

## Usage

### Example

Check out [example.py](https://github.com/KentoNishi/torch-pitch-shift/blob/master/example.py) to see `torch-pitch-shift` in action!

## Documentation
See the [documentation page](https://kentonishi.github.io/torch-pitch-shift/) for detailed documentation!

## Contributing
Please feel free to submit issues or pull requests!
