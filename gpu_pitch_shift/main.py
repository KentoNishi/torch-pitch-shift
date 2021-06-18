# Based on the librosa implementation of pitch shifting.
# https://github.com/librosa/librosa/blob/1468db6d95b276f426935966102f05b86fcc499c/librosa/effects.py#L260

import torch
from cusignal.filtering.resample import resample


def pitch_shift(
    input: torch.Tensor, sample_rate: int, n_steps: int, bins_per_octave=12, **kwargs
):
    """
    Shift the pitch of a waveform by ``n_steps`` steps.
    A step is equal to a semitone if ``bins_per_octave`` is set to 12.

    Parameters
    ----------
    input: torch.Tensor [shape=(batch_size, channels, samples)]
        Input samples of shape (batch_size, channels, samples)
    sample_rate: int
        sample rate of audio clips
    n_steps: int
        how many (fractional) steps to shift the input
    bins_per_octave: int
        how many steps per octave
    kwargs: additional keyword arguments.

    Returns
    -------
    output: torch.Tensor [shape=(batch_size, channels, samples)]
        The pitch-shifted batch of audio clips
    """
