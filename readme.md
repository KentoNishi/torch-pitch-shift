# Torch Pitch Shift

Pitch-shift audio clips quickly with PyTorch (CUDA Supported)!

[View on PyPI](https://pypi.org/project/torch-pitch-shift/)

[![Publish to PyPI](https://github.com/KentoNishi/torch_pitch_shift/actions/workflows/publish.yaml/badge.svg)](https://github.com/KentoNishi/torch_pitch_shift/actions/workflows/publish.yaml)

## Installation
```bash
pip install torch_pitch_shift
```

## Usage
```python
# import the libs
import torch
from torch_pitch_shift import *

# create a random sample
SAMPLE_RATE = 16000
NUM_SECONDS = 2
sample = torch.rand(2, SAMPLE_RATE * NUM_SECONDS)

# construct the pitch shifter (limit to between -1 and +1 octaves)
pitch_shift = PitchShifter(SAMPLE_RATE, lambda x: (x <= 2 and x >= 0.5))

for ratio in pitch_shift.fast_ratios:
    print(f"Ratio {ratio}:", pitch_shift(sample, ratio).shape)
```

## Documentation
Documentation is built into the class and function docstrings. If anyone wants to properly document the package, please feel free to contribute!
