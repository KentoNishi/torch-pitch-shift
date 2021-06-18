# Based on the librosa implementation of pitch shifting.
# https://github.com/librosa/librosa/blob/1468db6d95b276f426935966102f05b86fcc499c/librosa/effects.py#L260

import torchaudio.transforms as T
from numpy import isin
import torch
from torch.nn.functional import interpolate
from typing import Tuple, Callable
from primePy import primes
from functools import reduce
from fractions import Fraction
from itertools import combinations


class PitchShifter:
    def __init__(self, sample_rate: int, condition: Callable):
        """
        PitchShifter constructor.
        Shift the pitch of a waveform

        Parameters
        ----------
        sample_rate: int
            sample rate of input audio clips
        condition: Callable
            a function to determine if a ratio is valid.
            Example: ``lambda x: (x <= 2 and x >= 0.5)``
        """
        self._sample_rate = sample_rate
        self._resamplers = []
        self.fast_ratios = set()
        self._bins_per_octave = 12
        factors = primes.factors(sample_rate)
        products = []
        for i in range(1, len(factors) + 1):
            products.extend(
                [reduce(lambda x, y: x * y, x) for x in combinations(factors, i)]
            )
        for i in products:
            for j in products:
                f = Fraction(i, j)
                if condition(f):
                    self.fast_ratios.add(f)

    def __call__(self, input: torch.Tensor, shift: Fraction):
        """

        Parameters
        ----------
        input: torch.Tensor [shape=(channels, samples)]
            Input samples of shape (channels, samples)
        shift: Fraction
            For ratios: you can retrieve ratios that can be calculated quickly by accessing ``.fast_ratios``.

        Returns
        -------
        output: torch.Tensor [shape=(channels, samples)]
            The pitch-shifted batch of audio clips
        """
        shift = (shift.numerator, shift.denominator)
        resampler = T.Resample(
            self._sample_rate, int(self._sample_rate * shift[0] / shift[1])
        )
        output = interpolate(
            input[None, ...],
            size=int(input.shape[1] * shift[1] / shift[0]),
        )
        output = resampler(output)
        output = output[0]
        return output
