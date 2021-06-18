import torch
from gpu_pitch_shift import *

a = torch.rand(2, 44100 * 2, dtype=torch.float32)
pitch_shift = PitchShifter(41000)

for i in range(-12, 12 + 1):
    print(pitch_shift(a, i))
