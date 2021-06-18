# Torch Pitch Shift

Pitch-shift audio clips quickly with PyTorch (CUDA Supported)!

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
NUM_CHANNELS = 2
SAMPLE_RATE = 44100
NUM_SECONDS = 2
sample = torch.rand(NUM_CHANNELS, SAMPLE_RATE * NUM_SECONDS, dtype=torch.float32)
# you can also use CUDA tensors!

# construct the pitch shifter
pitch_shift = PitchShifter(SAMPLE_RATE)

# pitch shift the sample
SHIFT_SEMITONES = 5
print(pitch_shift(sample, SHIFT_SEMITONES))
```