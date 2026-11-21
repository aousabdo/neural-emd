"""Decompose a three-tone AM-FM signal with N-EMD and compare it to classical EMD."""
import numpy as np
import torch

from nemd.classical import ClassicalEMD
from nemd.train import overfit_single_signal
from nemd.utils import generate_synthetic_signal, mode_mixing_index, reconstruction_error

# 1 s at 1 kHz: carriers at 50, 15 and 3 Hz with slow amplitude modulation, plus noise
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

# Fit N-EMD to this one signal (no pretrained weights needed)
out = overfit_single_signal(torch.tensor(x, dtype=torch.float32),
                            num_imfs=3, num_steps=1500, fs=1000.0, verbose=False)
imfs = out["imfs"].detach().numpy()                      # (3, 1000)
freqs = np.fft.rfftfreq(len(x), d=1 / 1000.0)
peak = lambda c: freqs[np.argmax(np.abs(np.fft.rfft(c)))]
print("N-EMD IMF dominant frequencies (Hz):", sorted((round(float(peak(c)), 1) for c in imfs), reverse=True))
print(f"N-EMD reconstruction error: {reconstruction_error(x, imfs):.2e}")
print(f"N-EMD mode-mixing index:    {mode_mixing_index(truth, imfs):.3f}")

emd_imfs = ClassicalEMD().decompose(x)                   # needs the [baselines] extra
print(f"EMD   mode-mixing index:    {mode_mixing_index(truth, emd_imfs):.3f}  ({len(emd_imfs)} IMFs)")
print("EMD   first 3 IMF dominant frequencies (Hz):", [round(float(peak(c)), 1) for c in emd_imfs[:3]])
