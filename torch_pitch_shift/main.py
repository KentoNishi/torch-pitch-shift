import torchaudio.transforms as T
import torch
from torch.nn.functional import interpolate
from typing import Callable, Union
from primePy import primes
from functools import reduce
from fractions import Fraction
from itertools import chain, repeat, count, islice
from collections import Counter

# https://stackoverflow.com/a/46623112/9325832
def combinations_without_repetition(r, iterable=None, values=None, counts=None):
    if iterable:
        values, counts = zip(*Counter(iterable).items())

    f = lambda i, c: chain.from_iterable(map(repeat, i, c))
    n = len(counts)
    indices = list(islice(f(count(), counts), r))
    if len(indices) < r:
        return
    while True:
        yield tuple(values[i] for i in indices)
        for i, j in zip(reversed(range(r)), f(reversed(range(n)), reversed(counts))):
            if indices[i] != j:
                break
        else:
            return
        j = indices[i] + 1
        for i, j in zip(range(i, r), f(count(j), counts[j:])):
            indices[i] = j


class PitchShifter:
    def __init__(
        self,
        sample_rate: int,
        condition: Callable,
        generate_fast_shifts: bool = True,
        n_fft: int = 256,
    ):
        """
        PitchShifter constructor.
        Shift the pitch of a waveform by a given ratio.

        Parameters
        ----------
        sample_rate: int
            Sample rate of input audio clips
        condition: Callable
            A function to determine if a ratio is valid.
            Example: ``lambda x: (x <= 2 and x >= 0.5)``
        generate_fast_shifts: bool [optional]
            Whether or not to generate ``fast_shifts``. Default `True`.
        n_fft: int [optional]
            Size of FFT. Default 256. Smaller is faster.
        """
        self._n_fft = n_fft
        self._sample_rate = sample_rate
        self._resamplers = []
        self.fast_shifts = set()
        factors = primes.factors(sample_rate)
        products = []
        for i in range(1, len(factors) + 1):
            products.extend(
                [
                    reduce(lambda x, y: x * y, x)
                    for x in combinations_without_repetition(i, iterable=factors)
                ]
            )
        for i in products:
            for j in products:
                f = Fraction(i, j)
                if condition(f):
                    self.fast_shifts.add(f)

    def __call__(
        self, input: torch.Tensor, shift: Union[Fraction, float]
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input: torch.Tensor [shape=(channels, samples)]
            Input audio clip of shape (channels, samples)
        shift: Union[Fraction, float]
            You can retrieve ratios that can be calculated quickly by accessing ``.fast_shifts``.

        Returns
        -------
        output: torch.Tensor [shape=(channels, samples)]
            The pitch-shifted audio clip
        """
        # max_magnitude = torch.max(torch.abs(input))
        frac = Fraction(shift)
        shift = (frac.numerator, frac.denominator)
        resampler = T.Resample(
            self._sample_rate, int(self._sample_rate * shift[1] / shift[0])
        )
        output = torch.stft(input, self._n_fft)[None, ...]
        stretcher = T.TimeStretch(fixed_rate=float(1 / frac), n_freq=output.shape[2])
        output = stretcher(output)
        output = torch.istft(output[0], self._n_fft)
        output = resampler(output)
        # new_max_magnitude = torch.max(torch.abs(output))
        # output *= max_magnitude / new_max_magnitude
        return output
