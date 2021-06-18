# Based on the librosa implementation of pitch shifting.
# https://github.com/librosa/librosa/blob/1468db6d95b276f426935966102f05b86fcc499c/librosa/effects.py#L260

from .filtering import lilfilter
from numpy import isin
import torch
from torch.nn.functional import interpolate
from typing import Tuple


class PitchShifter:
    def __init__(
        self,
        sample_rate: int,
        bounds: Tuple[int, int] = (-12, 12),
        bins_per_octave: int = 12,
        dtype: torch.dtype = torch.float32,
        approximation_constant: int = 100,
    ):
        """
        PitchShifter constructor.
        Shift the pitch of a waveform by any number of steps contained within ``bounds``.
        A step is equal to a semitone if ``bins_per_octave`` is set to 12.

        Parameters
        ----------
        sample_rate: int
            sample rate of input audio clips
        bounds: Tuple[int, int] [optional]
            how many (fractional) steps to shift the input
        bins_per_octave: int [optional]
            how many steps per octave
        approximation_constant: int [optional]
            the larger the constant, the faster (but also less accurate) the pitchshift.
            default value is 100 (prioritizing speed).
        """
        if bins_per_octave < 1 or not isinstance(bins_per_octave, int):
            raise ValueError("bins_per_octave must be a positive integer")
        self._sample_rate = sample_rate
        self._bins_per_octave = bins_per_octave
        self._resamplers = []
        self._bounds = bounds
        for i in range(bounds[0], bounds[1] + 1):
            rate = 2.0 ** (-float(i) / bins_per_octave)
            s1, s2 = int(sample_rate / rate), int(sample_rate)
            self._resamplers.append(
                lilfilter.Resampler(
                    int(s1 / approximation_constant),
                    int(s2 / approximation_constant),
                    dtype=dtype,
                )
            )

    def __call__(self, input: torch.Tensor, n_steps: int):
        """

        Parameters
        ----------
        input: torch.Tensor [shape=(channels, samples)]
            Input samples of shape (channels, samples)
        n_steps: int
            Number of steps to shift the sample.
            A step is equal to a semitone if ``bins_per_octave`` is set to 12.

        Returns
        -------
        output: torch.Tensor [shape=(channels, samples)]
            The pitch-shifted batch of audio clips
        """
        rate = 2.0 ** (-float(n_steps) / self._bins_per_octave)
        stretched = interpolate(input[None, ...], scale_factor=rate)
        resampler = self._resamplers[n_steps - self._bounds[0]]
        output = resampler.resample(stretched[0])
        output = output[0][: input.shape[1]]
        return output
