# import the libs
import torch
import numpy as np
from scipy.io import wavfile
from torch_pitch_shift import *

# create a random sample
SAMPLE_RATE = 16000
SAMPLE_RATE, sample = wavfile.read("./wavs/test.wav")
dtype = sample.dtype
sample = torch.tensor(np.swapaxes(sample, 0, 1), dtype=torch.float32)

# construct the pitch shifter (limit to between -1 and +1 octaves)
pitch_shift = PitchShifter(SAMPLE_RATE, lambda x: (x <= 2 and x >= 0.5))

up = pitch_shift(sample, 2)
wavfile.write(
    "./wavs/test_+1.wav",
    SAMPLE_RATE,
    np.swapaxes(up.numpy(), 0, 1).astype(dtype),
)

down = pitch_shift(sample, 0.5)
wavfile.write(
    "./wavs/test_-1.wav",
    SAMPLE_RATE,
    np.swapaxes(down.numpy(), 0, 1).astype(dtype),
)
