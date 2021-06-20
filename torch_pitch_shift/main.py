from collections import Counter
from fractions import Fraction
from functools import reduce
from itertools import chain, count, islice, repeat
from typing import Union, Callable
from torch.nn.functional import pad
import torch
import torchaudio.transforms as T
from primePy import primes
import warnings

warnings.simplefilter("ignore")

# https://stackoverflow.com/a/46623112/9325832
def _combinations_without_repetition(r, iterable=None, values=None, counts=None):
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


def get_fast_shifts(
    sample_rate: int, condition: Callable = lambda x: x >= 0.5 and x <= 2 and x != 1
):
    """
    Search for pitch-shift targets that can be computed quickly for a given sample rate.

    Parameters
    ----------
    sample_rate: int
        The sample rate of an audio clip.
    condition: Callable [optional]
        A function to validate fast shift ratios.
        Default is `lambda x: x >= 0.5 and x <= 2 and x != 1` (between -1 and +1 octaves).
    """
    fast_shifts = set()
    factors = primes.factors(sample_rate)
    products = []
    for i in range(1, len(factors) + 1):
        products.extend(
            [
                reduce(lambda x, y: x * y, x)
                for x in _combinations_without_repetition(i, iterable=factors)
            ]
        )
    for i in products:
        for j in products:
            f = Fraction(i, j)
            if condition(f):
                fast_shifts.add(f)
    return list(fast_shifts)


class PitchShifter:
    def __init__(self, n_fft: int = 256, bins_per_octave: int = 12):
        """
        PitchShifter constructor.

        Parameters
        ----------
        n_fft: int [optional]
            Size of FFT. Default 256. Smaller is faster.
        bins_per_octave: int [optional]
            Number of bins per octave. Default is 12.
        """
        self._n_fft = n_fft
        self._bins_per_octave = bins_per_octave

    def __call__(
        self, input: torch.Tensor, shift: Union[float, Fraction], sample_rate: int
    ) -> torch.Tensor:
        """
        Shift the pitch of a waveform by a given amount.

        Parameters
        ----------
        input: torch.Tensor [shape=(channels, samples)]
            Input audio clip of shape (channels, samples)
        shift: float OR Fraction
            `float`: Amount to pitch-shift in # of bins. (1 bin == 1 semitone if `bins_per_octave` == 12)
            `Fraction`: A `fractions.Fraction` object indicating the shift ratio. Usually an element in `get_fast_shifts()`.
        sample_rate: int
            The sample rate of the input audio clip.

        sample_rate: int

        Returns
        -------
        output: torch.Tensor [shape=(channels, samples)]
            The pitch-shifted audio clip
        """
        if not isinstance(shift, Fraction):
            shift = 2.0 ** (float(shift) / self._bins_per_octave)
        resampler = T.Resample(sample_rate, int(sample_rate / shift)).to(input.device)
        output = input
        output = resampler(output)
        output = torch.stft(output, self._n_fft)[None, ...]
        stretcher = T.TimeStretch(
            fixed_rate=float(1 / shift), n_freq=output.shape[2]
        ).to(input.device)
        output = stretcher(output)
        output = torch.istft(output[0], self._n_fft)
        if output.shape[1] >= input.shape[1]:
            return output[:, : (input.shape[1])]
        return pad(output, pad=(0, 0, 0, input.shape[1] - output.shape[1]))
