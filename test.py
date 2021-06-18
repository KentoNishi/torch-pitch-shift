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
