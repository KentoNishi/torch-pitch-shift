# Torch Pitch Shift

Pitch-shift audio clips quickly with PyTorch (CUDA Supported)!

[View on PyPI](https://pypi.org/project/torch-pitch-shift/)

[![Publish to PyPI](https://github.com/KentoNishi/torch_pitch_shift/actions/workflows/publish.yaml/badge.svg)](https://github.com/KentoNishi/torch_pitch_shift/actions/workflows/publish.yaml)

## About

This library can pitch-shift audio clips quickly to using PyTorch. For any given sample rate, the library calculates pitch-shift ratios that can be run extremely fast.

## Installation
```bash
pip install torch_pitch_shift
```

## Usage

### Example:
```python
import torch
import numpy as np
from scipy.io import wavfile
from torch_pitch_shift import *

SAMPLE_RATE, sample = wavfile.read("./wavs/test.wav")
dtype = sample.dtype
sample = torch.tensor(np.swapaxes(sample, 0, 1), dtype=torch.float32)

pitch_shift = PitchShifter()

up = pitch_shift(sample, 12, SAMPLE_RATE)
wavfile.write(
    "./wavs/test_+1.wav",
    SAMPLE_RATE,
    np.swapaxes(up.numpy(), 0, 1).astype(dtype),
)

down = pitch_shift(sample, -12, SAMPLE_RATE)
wavfile.write(
    "./wavs/test_-1.wav",
    SAMPLE_RATE,
    np.swapaxes(down.numpy(), 0, 1).astype(dtype),
)
```

## Documentation
Documentation is built into the class and function docstrings. If anyone wants to properly document the package, please feel free to contribute!
