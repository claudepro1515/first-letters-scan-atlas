"""Scan several z-slices of a scroll and estimate its spiral_outward_sense (CW / ACW).

Usage:
  python sense_scan.py PHerc0826 [--umbilicus FILE_OR_URL] [--nz 10] [--out result.json]

EXPERIMENTAL, NOT RELIABLE: on PHerc0826 (published sense CW) it voted 5 CW / 3 ACW over 8 slices
(python sense_scan.py PHerc0826 --nz 8 --jitter 40 --out sense_PHerc0826.json). Kept so that the
negative result in the README can be checked; do not use it to set spiral_outward_sense.

Data are streamed from the public bucket (Lasagna cos / nx / ny predictions, pyramid level 2).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import sense_lsq as S  # noqa: E402
from vcz import BUCKET, OmeZarr, get_json, http_get, list_prefixes, list_keys  # noqa: E402


def _direct_lsq(A, b, **kw):
    AtA = (A.T @ A).tocsc() + 1e-8 * identity(A.shape[1], format="csc")
    return (spsolve(AtA, A.T @ b), 0, 0)


S.lsqr = _direct_lsq


def find_inputs(scroll: str) -> dict:
    out = {"scroll": scroll}
    vols = [p.rstrip("/").split("/")[-1] for p in list_prefixes(f"{scroll}/volumes/")]
    out["volumes"] = vols
    las = list_prefixes(f"{scroll}/representations/predictions/lasagna/")
    out["lasagna_dirs"] = [p.rstrip("/") for p in las]
    umb = [k for k, _ in list_keys(f"{scroll}/representations/umbilicus/") if k.endswith(".json")]
    out["umbilicus_keys"] = umb
    return out


def lasagna_urls(lasagna_dir: str) -> dict:
    keys = [k for k, _ in list_keys(lasagna_dir + "/") if k.endswith(".lasagna.json")]
    meta = get_json(f"{BUCKET}/{keys[0]}") if keys else None
    names = {}
    for p in list_prefixes(lasagna_dir + "/"):
        n = p.rstrip("/").split("/")[-1]
        for ch in ("cos", "nx", "ny", "grad_mag"):
            if n.endswith(f"_{ch}.ome.zarr"):
                names[ch] = f"{BUCKET}/{p.rstrip('/')}"
    return {"meta": meta, "stores": names}


def load_umbilicus(src: str):
    if src.startswith("http"):
        d = json.loads(http_get(src))
    elif src.startswith("s3key:"):
        d = get_json(f"{BUCKET}/{src[6:]}")
    else:
        d = json.load(open(src))
    pts = d["control_points"] if isinstance(d, dict) and "control_points" in d else d
    pts = sorted(pts, key=lambda p: p["z"])
    z = np.array([p["z"] for p in pts], float)
    x = np.array([p["x"] for p in pts], float)
    y = np.array([p["y"] for p in pts], float)
    return z, x, y


def read_slice(store_url: str, level: str, z: int, cache=None, bbox=None):
    g = OmeZarr(store_url, cache_dir=cache)
    a = g.level(level)
    sc = g.scales.get(level, 4.0)
    zl = int(round(z / sc))
    if bbox is None:
        y0, y1, x0, x1 = 0, a.shape[1], 0, a.shape[2]
    else:
        y0, y1, x0, x1 = bbox
    return a.read(zl, zl + 1, y0, y1, x0, x1)[0], sc


def scroll_bbox(cos_url: str, z: int, level_small="4", level="2", cache=None, pad=8):
    g = OmeZarr(cos_url, cache_dir=cache)
    small, sc_small = read_slice(cos_url, level_small, z, cache)
    sc = g.scales.get(level, 4.0)
    yy, xx = np.nonzero(small > 0)
    if len(yy) == 0:
        return None
    f = sc_small / sc
    a = g.level(level)
    y0 = max(0, int((yy.min() - pad) * f)); y1 = min(a.shape[1], int((yy.max() + pad + 1) * f))
    x0 = max(0, int((xx.min() - pad) * f)); x1 = min(a.shape[2], int((xx.max() + pad + 1) * f))
    return (y0, y1, x0, x1)


def crest_spacing(cos_u8, cx, cy, r0=60, r1=None):
    P, th, rs = S.polar_resample(cos_u8.astype(np.float32), cx, cy, 360, r0, r1 or 0.45 * max(cos_u8.shape), 1.0)
    sp = []
    for k in range(0, P.shape[0], 3):
        v = P[k]
        pk = np.nonzero((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] > 200))[0]
        if len(pk) > 3:
            sp += list(np.diff(pk))
    return float(np.median(sp)) if sp else 6.0


def sense_one_slice(stores, z, cx0, cy0, cache=None, level="2", n_theta=720, dr=2.0, r_min_px=None,
                    center_jitter=(), mirror=True):
    bbox = scroll_bbox(stores["cos"], z, cache=cache)
    if bbox is None:
        return None
    y0, y1, x0, x1 = bbox
    with ThreadPoolExecutor(3) as ex:
        futs = {ch: ex.submit(read_slice, stores[ch], level, z, cache, bbox) for ch in ("cos", "nx", "ny")}
        arrs = {ch: f.result()[0] for ch, f in futs.items()}
        sc = futs["cos"].result()[1]
    cos, nx, ny = arrs["cos"], arrs["nx"], arrs["ny"]
    cx, cy = cx0 / sc - x0, cy0 / sc - y0
    spacing = crest_spacing(cos, cx, cy)
    r_min = r_min_px if r_min_px is not None else max(40.0, 6 * spacing)
    res = {"z": int(z), "level": level, "scale": sc, "bbox": [int(v) for v in bbox], "spacing_px": spacing,
           "r_min_px": r_min}
    t = time.time()
    s, _, info = S.estimate_sense_slice(cos, nx, ny, cx, cy, r_min=r_min, n_theta=n_theta, dr=dr, spacing_px=spacing)
    res.update(s=s, residual=info["residual"], frac_valid=info["frac_valid"], solve_s=time.time() - t)
    if mirror:
        W = cos.shape[1]
        sm, _, _ = S.estimate_sense_slice(cos[:, ::-1], (256 - nx.astype(np.int32)).clip(0, 255).astype(np.uint8)[:, ::-1],
                                          ny[:, ::-1], W - 1 - cx, cy, r_min=r_min, n_theta=n_theta, dr=dr,
                                          spacing_px=spacing)
        res["s_mirror"] = sm
    if center_jitter:
        js = []
        for dx, dy in center_jitter:
            sj, _, _ = S.estimate_sense_slice(cos, nx, ny, cx + dx / sc, cy + dy / sc, r_min=r_min, n_theta=n_theta,
                                              dr=dr, spacing_px=spacing)
            js.append([dx, dy, sj])
        res["jitter"] = js
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scroll")
    ap.add_argument("--umbilicus", default=None, help="json file / URL / s3key:<key>; default = official in bucket")
    ap.add_argument("--nz", type=int, default=10)
    ap.add_argument("--zfrac", default="0.15,0.85")
    ap.add_argument("--jitter", type=float, default=0.0, help="also test centre offsets of +-J level-0 voxels")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    inp = find_inputs(a.scroll)
    if not inp["lasagna_dirs"]:
        print(json.dumps({"scroll": a.scroll, "error": "no lasagna predictions in bucket"}))
        return
    stores = lasagna_urls(inp["lasagna_dirs"][0])["stores"]
    usrc = a.umbilicus or (("s3key:" + inp["umbilicus_keys"][0]) if inp["umbilicus_keys"] else None)
    if usrc is None:
        print(json.dumps({"scroll": a.scroll, "error": "no umbilicus available"}))
        return
    uz, ux, uy = load_umbilicus(usrc)
    f0, f1 = map(float, a.zfrac.split(","))
    zs = np.linspace(uz.min() + f0 * (uz.max() - uz.min()), uz.min() + f1 * (uz.max() - uz.min()), a.nz).round().astype(int)
    jit = []
    if a.jitter:
        J = a.jitter
        jit = [(J, 0), (-J, 0), (0, J), (0, -J)]
    rows = []
    for z in zs:
        cx, cy = float(np.interp(z, uz, ux)), float(np.interp(z, uz, uy))
        t = time.time()
        r = sense_one_slice(stores, int(z), cx, cy, cache=a.cache, center_jitter=jit)
        if r is None:
            continue
        r["seconds"] = round(time.time() - t, 1)
        rows.append(r)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "bbox"}), flush=True)
    s = np.array([r["s"] for r in rows])
    sm = np.array([r.get("s_mirror", np.nan) for r in rows])
    n_cw = int((s < 0).sum()); n_acw = int((s > 0).sum())
    sense = "CW" if n_cw > n_acw else ("ACW" if n_acw > n_cw else "UNDECIDED")
    summary = {
        "scroll": a.scroll, "umbilicus": usrc, "lasagna": inp["lasagna_dirs"][0], "n_slices": len(rows),
        "votes_CW": n_cw, "votes_ACW": n_acw, "median_s": float(np.median(s)) if len(s) else None,
        "mirror_flips": int(np.sum(np.sign(sm) == -np.sign(s))), "sense": sense,
        "agreement": (max(n_cw, n_acw) / len(rows)) if rows else None,
    }
    print("SUMMARY " + json.dumps(summary))
    if a.out:
        json.dump({"summary": summary, "slices": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
