from numpy import number
import torchaudio.transforms as T
import torch


class PitchShifter:
    def __init__(
        self,
        n_fft: int = 256,
    ):
        """
        PitchShifter constructor.
        Shift the pitch of a waveform by a given ratio.

        Parameters
        ----------
        n_fft: int [optional]
            Size of FFT. Default 256. Smaller is faster.
        """
        self._n_fft = n_fft

    def __call__(
        self, input: torch.Tensor, shift: number, sample_rate: int
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input: torch.Tensor [shape=(channels, samples)]
            Input audio clip of shape (channels, samples)
        shift: float
            Amount to pitch-shift in semitones.
        sample_rate: int

        Returns
        -------
        output: torch.Tensor [shape=(channels, samples)]
            The pitch-shifted audio clip
        """
        shift = 2.0 ** (float(shift) / 12)
        resampler = T.Resample(sample_rate, int(sample_rate / shift))
        output = input
        output = resampler(output)
        output = torch.stft(output, self._n_fft)[None, ...]
        stretcher = T.TimeStretch(fixed_rate=float(1 / shift), n_freq=output.shape[2])
        output = stretcher(output)
        output = torch.istft(output[0], self._n_fft)
        return output
