# neural-emd

**N-EMD: Neural Empirical Mode Decomposition.** A differentiable replacement for the
iterative sifting in Empirical Mode Decomposition (EMD), written in PyTorch.

Classical EMD splits a signal into intrinsic mode functions (IMFs) by repeated
envelope fitting. It works without assumptions about the signal, but it is not
differentiable, the number of IMFs is whatever the sifting produces, and it is prone
to mode mixing. N-EMD learns the decomposition instead:

1. take the signal's spectrum (rFFT),
2. a small 1-D CNN reads magnitude and phase and predicts *K* filter responses,
3. a softmax across the *K* filters turns them into a partition of unity,
4. each filter is applied to the spectrum and inverted back to the time domain.

That construction gives three guarantees by design, not by training:

- **Exact reconstruction.** The IMFs always sum back to the input.
- **Band-limited IMFs.** Every filter is non-negative, so each IMF is a bandpass of the input.
- **No energy inflation.** The IMFs' total energy can never exceed the signal's.

Training only has to make the filters sharp, ordered by frequency and orthogonal.
Because the whole pipeline is differentiable, N-EMD can sit inside a larger model and
be trained end to end with a downstream loss.

## Install

```bash
pip install neural-emd                 # the library: torch, numpy, scipy
pip install "neural-emd[baselines]"    # adds classical EMD / EEMD / VMD for comparison
```

The import name is `nemd`. Python 3.11 or newer.

## Quick start

[`examples/quickstart.py`](examples/quickstart.py) fits N-EMD to a single noisy signal
made of three AM-FM tones (50, 15 and 3 Hz) and compares it with classical EMD:

```python
import numpy as np
import torch

from nemd.classical import ClassicalEMD
from nemd.train import overfit_single_signal
from nemd.utils import generate_synthetic_signal, mode_mixing_index, reconstruction_error

t, x, truth = generate_synthetic_signal(
    n_samples=1000,
    components=[
        {"f0": 50.0, "f_mod": 2.0, "a_mod": 0.5},
        {"f0": 15.0, "f_mod": 0.5, "a_mod": 0.3},
        {"f0": 3.0, "f_mod": 0.1, "a_mod": 0.2},
    ],
    noise_std=0.05,
    seed=42,
)

out = overfit_single_signal(torch.tensor(x, dtype=torch.float32),
                            num_imfs=3, num_steps=1500, fs=1000.0, verbose=False)
imfs = out["imfs"].detach().numpy()   # (3, 1000)
```

Output on a laptop CPU (about 30 seconds):

```
N-EMD IMF dominant frequencies (Hz): [58.0, 16.0, 3.0]
N-EMD reconstruction error: 1.96e-07
N-EMD mode-mixing index:    0.241
EMD   mode-mixing index:    0.236  (6 IMFs)
EMD   first 3 IMF dominant frequencies (Hz): [58.0, 42.0, 15.0]
```

N-EMD returns exactly the three components asked for, one per tone, and reconstructs
the signal to float precision. Classical EMD splits the same signal into six IMFs,
including a spurious 42 Hz mode.

## Training on a dataset

For signals from one family, train a model once and reuse it rather than fitting
each signal:

```python
from nemd.train import TrainConfig, train

result = train(TrainConfig(num_imfs=3, signal_length=1024, sample_rate=1000.0, num_epochs=100))
model = result["model"]                 # nemd.model.NEMD
imfs, residual, meta = model(batch)     # batch: (B, T) tensor -> imfs: (B, K, T)
```

`train` generates synthetic training data by default. Pass `dataset_fn` to train on
your own signals. Best and final weights are saved to `TrainConfig.save_dir`.

## What's in the package

| Module | Contents |
|---|---|
| `nemd.model` | `NEMD`, the filter-bank model; `NEMDSifting`, the earlier iterative variant kept for ablations |
| `nemd.losses` | Filter sharpness, frequency ordering, orthogonality, balance and the combined `NEMDLoss` |
| `nemd.train` | `TrainConfig`, `train`, `overfit_single_signal` |
| `nemd.utils` | Synthetic AM-FM, chirp and nonstationary signal generators; reconstruction error, orthogonality index, mode-mixing index, IF tracking error |
| `nemd.classical` | Thin wrappers for EMD, EEMD (PyEMD) and VMD (vmdpy), with the `[baselines]` extra |

## Status

Research code, alpha. The API may change before 1.0. A paper describing the method
and its evaluation is in preparation. Until then, please cite this repository if you
use it.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT. Copyright (c) 2026 Aous Abdo.
