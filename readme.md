# Torch Pitch Shift

Pitch-shift audio clips quickly with PyTorch (CUDA supported)! Additional utilities for searching efficient transformations are included.

[View on PyPI](https://pypi.org/project/torch-pitch-shift/) / [View Documentation](https://github.com/KentoNishi/torch_pitch_shift/wiki)

[![Publish to PyPI](https://github.com/KentoNishi/torch_pitch_shift/actions/workflows/publish.yaml/badge.svg)](https://github.com/KentoNishi/torch_pitch_shift/actions/workflows/publish.yaml)
[![PyPI version](https://img.shields.io/pypi/v/torch-pitch-shift.svg?style=flat)](https://pypi.org/project/torch-pitch-shift/)
[![Number of downloads from PyPI per month](https://img.shields.io/pypi/dm/torch-pitch-shift.svg?style=flat)](https://pypi.org/project/torch-pitch-shift/)
![Python version support](https://img.shields.io/pypi/pyversions/torch-pitch-shift)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/ambv/black)

## About

This package includes two main features:
* Pitch-shift audio clips quickly using PyTorch (with CUDA support)
* Calculate efficient pitch-shift targets (useful for augmentation, where speed is more important than precise pitch-shifts)

## Installation
```bash
pip install torch_pitch_shift
```

## Usage

### Example

It's super simple:
```python
# import libraries
import torch
from torch_pitch_shift import *

# specify the sample rate
SAMPLE_RATE = 16000
# create a random stereo audio clip (1s long)
audio = torch.rand(
    2,
    SAMPLE_RATE,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
# create the pitch shifter
pitch_shift = PitchShifter()

# for fast shift targets between -1 and +1 octaves
for ratio in get_fast_shifts(SAMPLE_RATE):
    # shift the audio clip
    shifted = pitch_shift(audio, ratio, SAMPLE_RATE)
    print(f"Pitch shift ({ratio}):", shifted)
```

Check out [example.py](https://github.com/KentoNishi/torch_pitch_shift/blob/master/example.py) to see `torch_pitch_shift` a more detailed example!

## Documentation
See the [GitHub Wiki Page](https://github.com/KentoNishi/torch_pitch_shift/wiki/3.-Documentation) for detailed documentation!

## Contributing
Please feel free to submit issues or pull requests!
