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
# import the libs
import torch
from torch_pitch_shift import *

# create a random sample
SAMPLE_RATE = 16000
NUM_SECONDS = 2
sample = torch.rand(2, SAMPLE_RATE * NUM_SECONDS)

# construct the pitch shifter (limit to between -1 and +1 octaves)
pitch_shift = PitchShifter(SAMPLE_RATE, lambda x: (x <= 2 and x >= 0.5))

for ratio in pitch_shift.fast_shifts:
    shifted = pitch_shift(sample, ratio)
    print(f"Ratio {ratio}:", shifted.shape)
```

### Output:
```
Ratio 1/2: torch.Size([2, 32000])
Ratio 1: torch.Size([2, 32000])
Ratio 2: torch.Size([2, 32000])
Ratio 5/4: torch.Size([2, 32000])
Ratio 5/8: torch.Size([2, 32000])
Ratio 25/16: torch.Size([2, 32000])
Ratio 25/32: torch.Size([2, 32000])
Ratio 4/5: torch.Size([2, 32000])
Ratio 125/64: torch.Size([2, 32000])
Ratio 125/128: torch.Size([2, 32000])
Ratio 64/125: torch.Size([2, 32000])
Ratio 128/125: torch.Size([2, 32000])
Ratio 8/5: torch.Size([2, 32000])
Ratio 32/25: torch.Size([2, 32000])
Ratio 16/25: torch.Size([2, 32000])
```

## Documentation
Documentation is built into the class and function docstrings. If anyone wants to properly document the package, please feel free to contribute!
