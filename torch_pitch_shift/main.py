from numpy import number
import torchaudio.transforms as T
import torch


class PitchShifter:
    def __init__(self, n_fft: int = 256, bins_per_octave: int = 12):
        """
        PitchShifter constructor.
        Shift the pitch of a waveform by a given ratio.

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
        self, input: torch.Tensor, shift: number, sample_rate: int
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input: torch.Tensor [shape=(channels, samples)]
            Input audio clip of shape (channels, samples)
        shift: number
            Amount to pitch-shift in # of bins.
            1 bin is 1 semitone if ``bins_per_octave`` is 12.
        sample_rate: int

        Returns
        -------
        output: torch.Tensor [shape=(channels, samples)]
            The pitch-shifted audio clip
        """
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
        return output
