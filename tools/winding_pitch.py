"""Winding pitch audit for spiral-fit meshes: how far apart are consecutive fitted windings, compared
with how far apart the papyrus sheets are in the scan?

In a spiral fit, winding i+1 is one full turn outside winding i, so on a scroll whose sheets are
spaced s apart the two surfaces should be about s apart. For each consecutive pair (wNNN, wNNN+1)
this tool measures, for up to 20,000 vertices of wNNN, the distance to the nearest point of wNNN+1
(densified 8 times, so points are 2.5 voxels apart on a 20-voxel grid; the discretization can only
make a gap look larger, never smaller). Pass the scan's sheet spacing (the atlas measures it:
data/atlas/atlas_summary.json, spacing_um) to get the ratio.

Usage:
  python winding_pitch.py <dir_with_wNNN_meshes> --start 20 40 60 --sheet-um 141 [--voxel-um 9.362]
Prints one JSON line per pair and a summary line.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import zoom
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_roughness import load_tifxyz  # noqa: E402


def densify(P, valid, f=8):
    Q = P.copy()
    Q[~valid] = np.nan
    Z = np.stack([zoom(Q[..., k], (f, f), order=1, mode="nearest") for k in range(3)], -1)
    return Z[np.isfinite(Z).all(-1)]


def pair_gap(d_a, d_b, n=20000, seed=0):
    Pa, va, _ = load_tifxyz(d_a)
    Pb, vb, _ = load_tifxyz(d_b)
    A = Pa[va]
    B = densify(Pb, vb)
    if len(A) == 0 or len(B) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(A), min(n, len(A)), replace=False)
    d, _ = cKDTree(B).query(A[idx])
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("meshes", help="folder with one tifxyz folder per winding, named wNNN")
    ap.add_argument("--start", type=int, nargs="+", required=True, help="first winding index of each pair")
    ap.add_argument("--sheet-um", type=float, default=None, help="the scan's sheet spacing in micrometres")
    ap.add_argument("--voxel-um", type=float, default=9.362)
    a = ap.parse_args()
    rows = []
    for i in a.start:
        da, db = (os.path.join(a.meshes, f"w{k:03d}") for k in (i, i + 1))
        if not (os.path.isdir(da) and os.path.isdir(db)):
            print(json.dumps({"pair": [i, i + 1], "error": "missing mesh"}))
            continue
        d = pair_gap(da, db)
        if d is None:
            continue
        p10, p50, p90 = np.percentile(d, [10, 50, 90])
        row = dict(pair=[i, i + 1], vertices=int(len(d)), gap_vox_p10=round(float(p10), 1),
                   gap_vox_p50=round(float(p50), 1), gap_vox_p90=round(float(p90), 1),
                   gap_um_p50=round(float(p50) * a.voxel_um, 0),
                   share_below_half_sheet=None)
        if a.sheet_um:
            half = a.sheet_um / a.voxel_um / 2
            row["share_below_half_sheet"] = round(float(np.mean(d < half)), 2)
            row["gap_over_sheet_spacing"] = round(float(p50) * a.voxel_um / a.sheet_um, 2)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if rows and a.sheet_um:
        r = np.array([x["gap_over_sheet_spacing"] for x in rows])
        print(json.dumps({"summary": True, "pairs": len(rows), "sheet_um": a.sheet_um,
                          "gap_over_sheet_spacing_median": round(float(np.median(r)), 2),
                          "range": [round(float(r.min()), 2), round(float(r.max()), 2)]}))


if __name__ == "__main__":
    main()
