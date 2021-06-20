import torch
import numpy as np
from scipy.io import wavfile
from torch_pitch_shift import *

# read an audio file
SAMPLE_RATE, sample = wavfile.read("./wavs/test.wav")

# convert to tensor of shape (channels, samples)
dtype = sample.dtype
sample = torch.tensor(
    np.swapaxes(sample, 0, 1),  # (samples, channels) --> (channels, samples)
    dtype=torch.float32,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# initialize the pitch shifter
pitch_shift = PitchShifter()

# pitch up by 12 semitones
up = pitch_shift(sample, 12, SAMPLE_RATE)
wavfile.write(
    "./wavs/shifted_octave_+1.wav",
    SAMPLE_RATE,
    np.swapaxes(up.cpu().numpy(), 0, 1).astype(dtype),
)

# pitch down by 12 semitones
down = pitch_shift(sample, -12, SAMPLE_RATE)
wavfile.write(
    "./wavs/shifted_octave_-1.wav",
    SAMPLE_RATE,
    np.swapaxes(down.cpu().numpy(), 0, 1).astype(dtype),
)

# get shift ratios that are fast (between +1 and -1 octaves)
for ratio in get_fast_shifts(SAMPLE_RATE):
    print("Shifting", ratio)
    wavfile.write(
        f"./wavs/shifted_ratio_{ratio.numerator}-{ratio.denominator}.wav",
        SAMPLE_RATE,
        np.swapaxes(pitch_shift(sample, ratio, SAMPLE_RATE).cpu().numpy(), 0, 1).astype(
            dtype
        ),
    )
