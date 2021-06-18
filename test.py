# import the libs
import torch
from torch_pitch_shift import *

# create a random sample
SAMPLE_RATE = 44100
NUM_SECONDS = 2
sample = torch.rand(2, SAMPLE_RATE * NUM_SECONDS, dtype=torch.float32)

# construct the pitch shifter
pitch_shift = PitchShifter(SAMPLE_RATE)

for i in range(-12, 12 + 1):
    print(pitch_shift(sample, i))
