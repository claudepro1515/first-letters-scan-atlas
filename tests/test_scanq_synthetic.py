"""Synthetic check of scanq's measurement: sheets of known period, contrast and orientation.

Builds 128^3 blocks of planar sinusoidal 'sheets' (random normal, period 10-20 voxels, relative
amplitude 0.10-0.40, Gaussian noise), runs the same structure-tensor + profile code as scanq and
reports how well period, modulation and orientation are recovered.
"""
import os
import sys

import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from scanq import eig_at, profile_metrics, structure_normals  # noqa: E402


def block(normal, period, rel_amp, noise, mean=120.0, size=128, seed=0):
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(*[np.arange(size)] * 3, indexing="ij")
    phase = 2 * np.pi * (normal[0] * z + normal[1] * y + normal[2] * x) / period
    b = mean * (1 + rel_amp * np.cos(phase)) + rng.normal(0, noise, (size,) * 3)
    return np.clip(b, 1, 255).astype(np.uint8)


def main():
    rng = np.random.default_rng(1)
    rows = []
    for t in range(12):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        period = rng.uniform(10, 20)
        amp = rng.uniform(0.10, 0.40)
        blk = block(n, period, amp, noise=8.0, seed=t)
        T = structure_normals(blk)
        pts = rng.integers(42, 86, size=(40, 3))
        nn, coh = eig_at(T, pts)
        ts = np.arange(-40, 41, 1.0)
        P = pts[:, None, :] + ts[None, :, None] * nn[:, None, :]
        prof = map_coordinates(blk.astype(np.float32), [P[..., 0].ravel(), P[..., 1].ravel(), P[..., 2].ravel()],
                               order=1, mode="nearest").reshape(len(pts), len(ts))
        sp, mod, snr, mean = profile_metrics(prof)
        ang = np.degrees(np.arccos(np.clip(np.abs(nn @ n), 0, 1)))
        rows.append((period, np.median(sp), amp, np.median(mod), np.median(ang), np.median(coh)))
    R = np.array(rows)
    print("true_period est_period  true_mod est_mod  normal_err_deg coherence")
    for r in R:
        print("  %6.2f   %6.2f      %5.3f   %5.3f     %5.2f        %4.2f" % tuple(r))
    ratio = R[:, 3] / R[:, 2]
    print("period error median %.2f vox, modulation ratio est/true median %.3f (range %.3f-%.3f), "
          "normal error median %.2f deg (max %.2f)"
          % (np.median(np.abs(R[:, 1] - R[:, 0])), np.median(ratio), ratio.min(), ratio.max(),
             np.median(R[:, 4]), R[:, 4].max()))
    assert np.max(np.abs(R[:, 1] - R[:, 0])) < 0.1, "sheet period off by more than 0.1 voxel"
    assert 0.94 < ratio.min() and ratio.max() < 1.03, "modulation bias outside -6 % .. +3 %"
    assert R[:, 4].max() < 2.0, "sheet normal off by more than 2 degrees"
    print("OK")


if __name__ == "__main__":
    main()
