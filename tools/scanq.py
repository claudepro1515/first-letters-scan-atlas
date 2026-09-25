"""Scan-quality atlas: how clearly the papyrus sheets show up in the CT, region by region.

For a scroll volume (full resolution, straight from the public bucket) we sample level-0
chunks spread over height (z) and depth (distance from the scroll axis). Inside each chunk a
3D structure tensor gives the local sheet normal; along the normal we take an 81-voxel CT
profile and measure the periodic sheet signal:
  * spacing_vox  - dominant period (sheet-to-sheet distance; small = compressed papyrus)
  * modulation   - amplitude of that period relative to the mean intensity (sheet contrast)
  * snr          - power at that period over the median power of the other periods
  * coherence    - how planar/consistent the local sheet orientation is
Everything runs on CPU with numpy/scipy; one chunk = 2 MB download.
"""
from __future__ import annotations

import argparse
import json
import re
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcz import BUCKET, OmeZarr, get_json, list_keys, list_prefixes  # noqa: E402


def pick_volume(scroll, prefer=("9.362um", "9.366um", "8.640um", "7.910um")):
    vols = [p.rstrip("/").split("/")[-1] for p in list_prefixes(f"{scroll}/volumes/")]
    for pref in prefer:
        for v in vols:
            if pref in v:
                return v
    return vols[0] if vols else None


def structure_normals(block, sigma_g=1.0, sigma_t=4.0):
    b = gaussian_filter(block.astype(np.float32), sigma_g)
    gz, gy, gx = np.gradient(b)
    T = {}
    for k, (a, c) in {"zz": (gz, gz), "zy": (gz, gy), "zx": (gz, gx), "yy": (gy, gy), "yx": (gy, gx), "xx": (gx, gx)}.items():
        T[k] = gaussian_filter(a * c, sigma_t)
    return T


def eig_at(T, pts):
    """Principal eigenvector (normal, zyx) and coherence at integer points."""
    zi, yi, xi = pts[:, 0], pts[:, 1], pts[:, 2]
    M = np.zeros((len(pts), 3, 3), np.float64)
    M[:, 0, 0] = T["zz"][zi, yi, xi]; M[:, 1, 1] = T["yy"][zi, yi, xi]; M[:, 2, 2] = T["xx"][zi, yi, xi]
    M[:, 0, 1] = M[:, 1, 0] = T["zy"][zi, yi, xi]
    M[:, 0, 2] = M[:, 2, 0] = T["zx"][zi, yi, xi]
    M[:, 1, 2] = M[:, 2, 1] = T["yx"][zi, yi, xi]
    w, v = np.linalg.eigh(M)
    n = v[:, :, 2]
    coh = (w[:, 2] - w[:, 1]) / np.maximum(w[:, 2] + w[:, 1], 1e-9)
    return n, coh


def profile_metrics(prof, pmin=5.0, pmax=40.0, pad=1024):
    """Dominant sheet period (sub-bin, zero-padded FFT + parabolic peak), its relative amplitude
    (modulation), spectral SNR and mean intensity, per profile."""
    N = prof.shape[1]
    mean = prof.mean(1)
    win = np.hanning(N)
    x = (prof - mean[:, None]) * win[None, :]
    F = np.fft.rfft(x, n=pad, axis=1)
    P = np.abs(F) ** 2
    freqs = np.fft.rfftfreq(pad)
    band = np.nonzero((freqs >= 1 / pmax) & (freqs <= 1 / pmin))[0]
    Pb = P[:, band]
    k = np.argmax(Pb, 1)
    kk = np.clip(k, 1, len(band) - 2)
    ym, y0, yp = Pb[np.arange(len(k)), kk - 1], Pb[np.arange(len(k)), kk], Pb[np.arange(len(k)), kk + 1]
    den = ym - 2 * y0 + yp
    with np.errstate(divide="ignore", invalid="ignore"):  # flat spectra: frac falls back to 0
        frac = np.where(np.abs(den) > 1e-12, 0.5 * (ym - yp) / den, 0.0).clip(-0.5, 0.5)
    fk = freqs[band][kk] + frac * (freqs[1] - freqs[0])
    amp = 2 * np.sqrt(Pb[np.arange(len(k)), k]) / win.sum()
    # SNR against the rest of the band, excluding the peak's neighbourhood
    snr = Pb[np.arange(len(k)), k] / np.maximum(np.median(Pb, 1), 1e-9)
    # periodicity: share of the profile's variance explained by a sinusoid at the sheet period
    # (least squares on the un-windowed, mean-removed profile); unchanged by intensity offset/gain
    t = np.arange(N)[None, :]
    c = np.cos(2 * np.pi * fk[:, None] * t)
    s_ = np.sin(2 * np.pi * fk[:, None] * t)
    xr = prof - mean[:, None]
    a11 = (c * c).sum(1); a22 = (s_ * s_).sum(1); a12 = (c * s_).sum(1)
    b1 = (xr * c).sum(1); b2 = (xr * s_).sum(1)
    det = a11 * a22 - a12 * a12
    ca = (b1 * a22 - b2 * a12) / np.maximum(det, 1e-9)
    sa = (b2 * a11 - b1 * a12) / np.maximum(det, 1e-9)
    fit = ca[:, None] * c + sa[:, None] * s_
    r2 = 1 - ((xr - fit) ** 2).sum(1) / np.maximum((xr ** 2).sum(1), 1e-9)
    profile_metrics.last_r2 = r2
    profile_metrics.last_amp = amp
    return 1 / fk, amp / np.maximum(mean, 1e-6), snr, mean


def sample_chunk(arr, idx, n_pts=40, half=40, rng=None, min_mean=20):
    blk = arr._load_chunk(idx)
    if blk is None:
        return None
    blk = np.asarray(blk)
    if (blk > 0).mean() < 0.95 or blk.mean() < min_mean:
        return None
    T = structure_normals(blk)
    c = blk.shape[0]
    lo, hi = half + 2, c - half - 2
    if hi <= lo:
        lo, hi = c // 2 - 4, c // 2 + 4
    rng = rng or np.random.default_rng(0)
    pts = rng.integers(lo, hi, size=(n_pts, 3))
    n, coh = eig_at(T, pts)
    ts = np.arange(-half, half + 1, 1.0)
    P = pts[:, None, :] + ts[None, :, None] * n[:, None, :]
    prof = map_coordinates(blk.astype(np.float32), [P[..., 0].ravel(), P[..., 1].ravel(), P[..., 2].ravel()], order=1,
                           mode="nearest").reshape(n_pts, len(ts))
    spacing, mod, snr, mean = profile_metrics(prof)
    r2 = profile_metrics.last_r2
    amp = profile_metrics.last_amp
    base = np.array(idx) * np.array(arr.chunks)
    p2, p50, p98 = np.percentile(blk[::2, ::2, ::2], [2, 50, 98])
    return dict(z=base[0] + pts[:, 0], y=base[1] + pts[:, 1], x=base[2] + pts[:, 2], coherence=coh,
                spacing=spacing, modulation=mod, snr=snr, mean=mean, periodicity=r2, amplitude=amp,
                chunk_p2=np.full(n_pts, p2), chunk_p50=np.full(n_pts, p50), chunk_p98=np.full(n_pts, p98))


def scroll_mask_and_axis(g):
    """Coarse interior mask from the smallest pyramid level; axis (centroid per z) for radius."""
    levels = sorted(g.scales, key=lambda p: g.scales[p])
    lv = levels[-1]
    a = g.level(lv)
    sc = g.scales[lv]
    m = a.read(0, a.shape[0], 0, a.shape[1], 0, a.shape[2]) > 0
    return m, sc


def run(scroll, n_z=10, n_r=3, per_bin=4, workers=8, volume=None, seed=0, cache=None):
    vol = volume or pick_volume(scroll)
    g = OmeZarr(f"{BUCKET}/{scroll}/volumes/{vol}", cache_dir=cache)
    a0 = g.level("0")
    m, sc = scroll_mask_and_axis(g)
    cs = np.array(a0.chunks)
    Z, Y, X = a0.shape
    rng = np.random.default_rng(seed)
    # candidate chunks: fully inside the coarse mask
    nz, ny, nx = (np.array(a0.shape) + cs - 1) // cs
    cand = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                z0, y0, x0 = np.array([iz, iy, ix]) * cs / sc
                z1, y1, x1 = (np.array([iz, iy, ix]) + 1) * cs / sc
                sub = m[int(z0):int(np.ceil(z1)), int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))]
                if sub.size and sub.all():
                    cand.append((iz, iy, ix))
    cand = np.array(cand)
    if len(cand) == 0:
        return None
    # radius relative to per-z-slab centroid of the mask, normalised by the slab's max radius
    rel = np.zeros(len(cand))
    for iz in np.unique(cand[:, 0]):
        sel = cand[:, 0] == iz
        cy, cx = cand[sel, 1].mean(), cand[sel, 2].mean()
        r = np.hypot(cand[sel, 1] - cy, cand[sel, 2] - cx)
        rel[sel] = r / max(r.max(), 1e-6)
    zb = np.minimum((cand[:, 0] / nz * n_z).astype(int), n_z - 1)
    rb = np.minimum((rel * n_r).astype(int), n_r - 1)
    chosen = []
    for i in range(n_z):
        for j in range(n_r):
            ids = np.nonzero((zb == i) & (rb == j))[0]
            if len(ids):
                chosen += list(rng.choice(ids, size=min(per_bin, len(ids)), replace=False))
    t = time.time()

    def job(k):
        iz, iy, ix = cand[k]
        r = sample_chunk(a0, (int(iz), int(iy), int(ix)), rng=np.random.default_rng(int(k)))
        if r is not None:
            r["zbin"] = int(zb[k]); r["rbin"] = int(rb[k])
        return r

    with ThreadPoolExecutor(workers) as ex:
        res = [r for r in ex.map(job, chosen) if r is not None]
    out = {"scroll": scroll, "volume": vol, "n_chunks": len(res), "seconds": round(time.time() - t, 1)}
    if not res:
        return out
    cat = {k: np.concatenate([r[k] for r in res]) for k in ("z", "y", "x", "coherence", "spacing", "modulation", "snr", "mean",
                                                             "periodicity", "amplitude", "chunk_p2", "chunk_p50", "chunk_p98")}
    cat["zbin"] = np.concatenate([np.full(len(r["z"]), r["zbin"]) for r in res])
    cat["rbin"] = np.concatenate([np.full(len(r["z"]), r["rbin"]) for r in res])
    good = cat["coherence"] > 0.3
    out.update({
        "n_profiles": int(good.sum()),
        "modulation_median": float(np.median(cat["modulation"][good])),
        "modulation_p75": float(np.percentile(cat["modulation"][good], 75)),
        "snr_median": float(np.median(cat["snr"][good])),
        "spacing_median_vox": float(np.median(cat["spacing"][good])),
        "coherence_median": float(np.median(cat["coherence"])),
        "fog_fraction": float(np.mean(cat["modulation"][good] < 0.15)),
        "periodicity_median": float(np.median(cat["periodicity"][good])),
        "incoherent_fraction": float(np.mean(~good)),
        "by_rbin_modulation": [float(np.median(cat["modulation"][good & (cat["rbin"] == j)])) if (good & (cat["rbin"] == j)).any() else None for j in range(n_r)],
        "by_zbin_modulation": [float(np.median(cat["modulation"][good & (cat["zbin"] == i)])) if (good & (cat["zbin"] == i)).any() else None for i in range(n_z)],
    })
    return out, cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scrolls", nargs="+")
    ap.add_argument("--per-bin", type=int, default=4)
    ap.add_argument("--nz", type=int, default=10)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for s in a.scrolls:
        r = run(s, n_z=a.nz, per_bin=a.per_bin, cache=a.cache)
        if r is None:
            print(json.dumps({"scroll": s, "error": "no interior chunks"}))
            continue
        if isinstance(r, tuple):
            out, cat = r
            np.savez_compressed(f"{a.outdir}/scanq_{s}.npz", **cat)
        else:
            out = r
        print(json.dumps(out), flush=True)
        json.dump(out, open(f"{a.outdir}/scanq_{s}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
