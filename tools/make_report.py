"""Summary tables for the atlas: per-scroll medians with chunk-bootstrap intervals, by protocol, plus
per-height-band medians of sheet modulation and a chunk-level permutation test of the clearest band.

Usage: python make_report.py <scanq_dir> <out_dir>
Writes atlas_summary.json/.csv, atlas_height_bands.json and atlas_height_band_tests.json.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atlas_figs import ELIGIBLE, LETTERS_REPORTED, load  # noqa: E402


def protocol(vol):
    """Scan protocol label from a volume name like 20250821151803-9.362um-1.2m-113keV-masked.zarr."""
    import re
    um = re.search(r"-([\d.]+)um", vol)
    dist = re.search(r"-([\d.]+)m-", vol)
    kev = re.search(r"-(\d+)keV", vol)
    u = float(um.group(1)) if um else None
    ulab = f"{round(u, 2):g} µm" if u else "?"
    if u and abs(u - 9.36) < 0.02:
        ulab = "9.36 µm"
    return ", ".join(x for x in [ulab, f"{dist.group(1)} m" if dist else None, f"{kev.group(1)} keV" if kev else None] if x)


def boot_median(values, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.nonzero(groups == g)[0] for g in ug}
    meds = []
    for _ in range(n):
        pick = rng.choice(ug, size=len(ug), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        meds.append(np.median(values[sel]))
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def summarize(scanq_dir, cs=128):
    rows = load(scanq_dir)
    out = []
    for s, (j, c) in rows.items():
        good = c["coherence"] > 0.3
        chunk = (c["z"] // cs) * 1_000_000 + (c["y"] // cs) * 1000 + (c["x"] // cs)
        m = c["modulation"][good]
        lo, hi = boot_median(m, chunk[good])
        per = c["periodicity"][good] if "periodicity" in c else None
        plo, phi = boot_median(per, chunk[good]) if per is not None else (None, None)
        um = float(j["volume"].split("-")[1].replace("um", ""))
        out.append(dict(
            scroll=s,
            group="ink found" if s in LETTERS_REPORTED else ("eligible" if s in ELIGIBLE else "other"),
            protocol=protocol(j["volume"]),
            volume=j["volume"],
            chunks=int(len(np.unique(chunk))),
            profiles=int(good.sum()),
            modulation=round(float(np.median(m)), 3),
            modulation_ci95=[round(lo, 3), round(hi, 3)],
            coherence=round(float(np.median(c["coherence"])), 2),
            spacing_um=round(float(np.median(c["spacing"][good])) * um, 0),
            fog_fraction=round(float(np.mean(m < 0.15)), 3),
            periodicity=round(float(np.median(per)), 3) if per is not None else None,
            periodicity_ci95=[round(plo, 3), round(phi, 3)] if per is not None else None,
            incoherent_fraction=round(float(np.mean(~good)), 3),
        ))
    out.sort(key=lambda r: -r["modulation"])
    return out, rows


def height_bands(rows, nz=12, min_n=15, key="periodicity"):
    res = {}
    for s, (j, c) in rows.items():
        good = c["coherence"] > 0.3
        bands = []
        for k in range(nz):
            sel = good & (c["zbin"] == k)
            bands.append(round(float(np.median(c[key][sel])), 3) if sel.sum() >= min_n else None)
        res[s] = bands
    return res


def band_permutation(rows, nz=12, key="periodicity", n=2000, min_profiles=50, seed=0, cs=128):
    """For each scroll: is its clearest height band clearer than chance? Band labels are shuffled
    between chunks (not profiles); p = share of shuffles whose best band beats the observed one."""
    rng = np.random.default_rng(seed)
    res = {}
    for s, (j, c) in rows.items():
        good = c["coherence"] > 0.3
        chunk = (c["z"] // cs) * 1_000_000 + (c["y"] // cs) * 1000 + (c["x"] // cs)
        ch = chunk[good]
        v = c[key][good]
        zb = c["zbin"][good]
        uch, inv = np.unique(ch, return_inverse=True)
        chunk_band = np.zeros(len(uch), int)
        chunk_band[inv] = zb
        def best(bands_of_chunk):
            b = bands_of_chunk[inv]
            vals = []
            for k in range(nz):
                sel = b == k
                if sel.sum() >= min_profiles:
                    vals.append((np.median(v[sel]), k))
            return max(vals) if vals else (np.nan, None)
        obs, kobs = best(chunk_band)
        null = np.array([best(rng.permutation(chunk_band))[0] for _ in range(n)])
        p = float((np.sum(null >= obs) + 1) / (n + 1))
        res[s] = dict(best_band=kobs, best_value=round(float(obs), 3), overall=round(float(np.median(v)), 3), p=round(p, 4))
    return res


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    scanq_dir, outd = sys.argv[1], sys.argv[2]
    os.makedirs(outd, exist_ok=True)
    summ, rows = summarize(scanq_dir)
    if not summ:
        sys.exit(f"no scanq_*.npz files found in {scanq_dir} or {scanq_dir}/profiles")
    json.dump(summ, open(os.path.join(outd, "atlas_summary.json"), "w"), indent=1)
    with open(os.path.join(outd, "atlas_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scroll", "group", "protocol", "chunks", "profiles", "periodicity", "periodicity_ci95_low",
                    "periodicity_ci95_high", "modulation", "ci95_low", "ci95_high", "coherence", "spacing_um",
                    "fog_fraction", "incoherent_fraction", "volume"])
        for r in summ:
            w.writerow([r["scroll"], r["group"], r["protocol"], r["chunks"], r["profiles"], r["periodicity"],
                        (r["periodicity_ci95"] or [None, None])[0], (r["periodicity_ci95"] or [None, None])[1],
                        r["modulation"], r["modulation_ci95"][0], r["modulation_ci95"][1], r["coherence"],
                        r["spacing_um"], r["fog_fraction"], r["incoherent_fraction"], r["volume"]])
    hb = height_bands(rows, key="modulation")
    json.dump(hb, open(os.path.join(outd, "atlas_height_bands.json"), "w"), indent=1)
    perm = band_permutation(rows, key="modulation")
    json.dump(perm, open(os.path.join(outd, "atlas_height_band_tests.json"), "w"), indent=1)
    alpha = 0.05 / max(len(perm), 1)
    sig = {s: r for s, r in perm.items() if r["p"] < alpha}
    print(f"height bands clearer than chance after Bonferroni (p < {alpha:.4f}): {sorted(sig)}")
    for r in summ:
        print(f"{r['scroll']:12s} {r['group']:16s} {r['protocol']:22s} per {r['periodicity'] if r['periodicity'] is not None else float('nan'):.3f} mod {r['modulation']:.3f} "
              f"[{r['modulation_ci95'][0]:.3f},{r['modulation_ci95'][1]:.3f}] coh {r['coherence']:.2f} "
              f"sp {r['spacing_um']:.0f}um fog {r['fog_fraction']:.2f} chunks {r['chunks']}")
