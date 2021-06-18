import torchaudio.transforms as T
import torch
from torch.nn.functional import interpolate
from typing import Callable
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
    def __init__(self, sample_rate: int, condition: Callable):
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
        """
        self._sample_rate = sample_rate
        self._resamplers = []
        self.fast_shifts = set()
        self._bins_per_octave = 12
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

    def __call__(self, input: torch.Tensor, shift: Fraction) -> torch.Tensor:
        """
        Parameters
        ----------
        input: torch.Tensor [shape=(channels, samples)]
            Input audio clip of shape (channels, samples)
        shift: Fraction
            You can retrieve ratios that can be calculated quickly by accessing ``.fast_shifts``.

        Returns
        -------
        output: torch.Tensor [shape=(channels, samples)]
            The pitch-shifted audio clip
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
