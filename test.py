import torch
import numpy as np
from scipy.io import wavfile
from torch_pitch_shift import *

SAMPLE_RATE, sample = wavfile.read("./wavs/test.wav")
dtype = sample.dtype
sample = torch.tensor(np.swapaxes(sample, 0, 1), dtype=torch.float32).cuda()

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
