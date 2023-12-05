from typing import *
from functools import partial
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from torchaudio.transforms import Spectrogram
from scipy.signal.windows import dpss

DEFAULT_1020_CHANNELS = [
    "f7",
    "f3",
    "fp1",
    "f4",
    "f8",
    "c3",
    "cz",
    "c4",
    "a1",
    "a2",
    "t3",
    "p3",
    "p4",
    "t4",
    "o1",
    "o2",
    "fz",
    "fp2",
    "t5",
    "pz",
    "t6",
]


def gen_x(
    sample_rate: float,
    sample_secs: float,
    channels: int = 21,
    frequencies: Optional[Iterable[float]] = None,
    start_time: float = 0,
):
    """Generate random or sinusoidal data for testing purposes."""
    times = np.arange(0, sample_secs * 1000, 1 / sample_rate * 1000)
    if frequencies:
        x_channel = 0
        for frequency in frequencies:
            x_channel += np.sin(2 * np.pi * frequency * times / 1000)
        x = np.tile(x_channel, (channels, 1)).T
    else:
        x = np.random.randn(len(times), channels)

    if start_time:
        times += start_time

    return pd.DataFrame(x, columns=DEFAULT_1020_CHANNELS, index=times)


# data params
sample_rate = 256
sample_secs = 60
start_time = datetime.fromisoformat("2020-03-04").timestamp() * 1e3
batch_size = 3

# spectrogram params
n_fft = sample_rate
win_length = sample_rate
hop_length = 8
normalized = True

# dpss prams
alpha = 3
def dpss_tensor(win_length, **kwargs):
    foo = dpss(win_length, **kwargs)
    return torch.Tensor(foo)


if __name__ == "__main__":
    samples = []
    for i in range(batch_size):
        x = (
            gen_x(
                sample_rate=sample_rate,
                sample_secs=sample_secs,
                start_time=start_time + i * sample_secs * 1e3,
                frequencies=[5, 12, 20],
            )
            .to_numpy()
            .T
        )
        samples.append(x)

    xs = np.stack([x for x in samples], axis=0)

    spectrogram_0 = Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        normalized=normalized,
        window_fn=torch.hann_window,
    )
    spectrogram_1 = Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        normalized=normalized,
        window_fn=lambda n, **kwargs: torch.Tensor(dpss(n, **kwargs)),
        wkwargs=dict(NW=alpha),
    )

    xs_hat_0 = spectrogram_0(torch.Tensor(xs))
    xs_hat_1 = spectrogram_1(torch.Tensor(xs))
