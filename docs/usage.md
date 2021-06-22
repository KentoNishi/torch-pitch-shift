# Usage

## Installation
You can install `torch-pitch-shift` from [PyPI](https://pypi.org/project/torch-pitch-shift/).

```bash
pip install torch-pitch-shift
```

To upgrade an existing installation of `torch-pitch-shift`, use the following command:

```bash
pip install --upgrade --no-cache-dir torch-pitch-shift
```

## Importing

First, import `torch-pitch-shift`.

```python
# import all functions
from torch_pitch_shift import *

# ... or import them manually
from torch_pitch_shift import get_fast_shifts, pitch_shift
```

## What's included
`torch-pitch-shift` includes the following:

<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>Name</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Function</td>
      <td><code>get_fast_shifts</code></td>
      <td>Utility function for calculating pitch-shifts that can be executed quickly.</td>
    </tr>
    <tr>
      <td>Function</td>
      <td><code>pitch_shift</code></td>
      <td>Shift the pitch of a batch of waveforms by a given amount.</td>
    </tr>
  </tbody>
</table>

## Methods

### `pitch_shift`
Shift the pitch of a batch of waveforms by a given amount.

#### Arguments

<table>
  <thead>
    <tr>
      <th>Argument</th>
      <th>Required</th>
      <th>Default Value</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>input</code></td>
      <td>Yes</td>
      <td></td>
<td><pre><code>torch.Tensor [
    shape=(
        batch_size,
        channels,
        samples
    )
]</code></pre></td>
      <td>Input audio clips of shape (batch_size, channels, samples)</td>
    </tr>
    <tr>
      <td><code>shift</code></td>
      <td>Yes</td>
      <td></td>
      <td><code>float</code> or <code>Fraction</code></td>
      <td>Inputs of type <code>float</code> indicate the amount to pitch-shift in # of bins (where 1 bin == 1 semitone if <code>bins_per_octave</code> == 12). Inputs of type <code>Fraction</code> indicate a pitch-shift ratio (usually an element in <code>get_fast_shifts()</code>).</td>
    </tr>
    <tr>
      <td><code>sample_rate</code></td>
      <td>Yes</td>
      <td></td>
      <td><code>int</code></td>
      <td>The sample rate of the input audio clips.</td>
    </tr>
    <tr>
      <td><code>n_fft</code></td>
      <td>No</td>
      <td><code>256</code></td>
      <td><code>int</code></td>
      <td>Size of FFT. Default 256. Smaller is faster.</td>
    </tr>
    <tr>
      <td><code>bins_per_octave</code></td>
      <td>No</td>
      <td><code>12</code></td>
      <td><code>int</code></td>
      <td>Number of bins per octave. Default is 12.</td>
    </tr>
  </tbody>
</table>

#### Return value

<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
<td><pre><code>torch.Tensor [
    shape=(
        batch_size,
        channels,
        samples
    )
]</code></pre></td>
      <td>The pitch-shifted batch of audio clips</td>
    </tr>
  </tbody>
</table>

### `get_fast_shifts`
Search for pitch-shift targets that can be computed quickly for a given sample rate.

<table>
  <thead>
    <tr>
      <th>Argument</th>
      <th>Required</th>
      <th>Default Value</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>sample_rate</code></td>
      <td>Yes</td>
      <td></td>
      <td><code>int</code></td>
      <td>The sample rate of an audio clip.</td>
    </tr>
    <tr>
      <td><code>condition</code></td>
      <td>No</td>
      <td>
<pre><code>lambda x: (
    x &gt;= 0.5 and x &lt;= 2 and x != 1
)</code></pre>
      </td>
      <td><code>Callable</code></td>
      <td>A function to validate fast shift ratios. Default value limits computed targets to values between <code>-1</code> and <code>+1</code> octaves.</td>
    </tr>
  </tbody>
</table>
#### Return value

<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>List[Fraction]</code></td>
      <td>A list of fast pitch-shift target ratios that satisfy the given conditions.</td>
    </tr>
  </tbody>
</table>

## Example

See [example.py](https://github.com/KentoNishi/torch-pitch-shift/blob/master/example.py) to see an example of `torch-pitch-shift` in action!