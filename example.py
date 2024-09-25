import torch
import numpy as np
from scipy.io import wavfile
from torch_pitch_shift import *

# read an audio file
SAMPLE_RATE, sample = wavfile.read("./wavs/test.wav")

# convert to tensor of shape (batch_size, channels, samples)
dtype = sample.dtype
sample = torch.tensor(
    np.expand_dims(np.swapaxes(sample, 0, 1),0),  # (samples, channels) --> (channels, samples)
    dtype=torch.float32,
    device="cuda" if torch.cuda.is_available() else "cpu",
)


def test_pitch_shift_12_up():
    # pitch up by 12 semitones
    up = pitch_shift(sample, 12, SAMPLE_RATE)
    assert up.shape == sample.shape
    wavfile.write(
        "./wavs/shifted_octave_+1.wav",
        SAMPLE_RATE,
        np.swapaxes(up.cpu()[0].numpy(), 0, 1).astype(dtype),
    )


def test_pitch_shift_12_down():
    # pitch down by 12 semitones
    down = pitch_shift(sample, -12, SAMPLE_RATE)
    assert down.shape == sample.shape
    wavfile.write(
        "./wavs/shifted_octave_-1.wav",
        SAMPLE_RATE,
        np.swapaxes(down.cpu()[0].numpy(), 0, 1).astype(dtype),
    )


def test_pitch_shift_to_fast_ratios():
    # get shift ratios that are fast (between +1 and -1 octaves)
    for ratio in get_fast_shifts(SAMPLE_RATE):
        print("Shifting", ratio)
        shifted = pitch_shift(sample, ratio, SAMPLE_RATE)
        assert shifted.shape == sample.shape
        wavfile.write(
            f"./wavs/shifted_ratio_{ratio.numerator}-{ratio.denominator}.wav",
            SAMPLE_RATE,
            np.swapaxes(shifted.cpu()[0].numpy(), 0, 1).astype(dtype),
        )
