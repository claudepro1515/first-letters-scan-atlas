"""Does local sheet clarity go with stronger raw ink contrast at 9 um? (PHerc0139 labelled segments)

For every 128x128 tile inside the supervision mask with enough ink and background pixels:
  clarity  = contrast of the tile's mean-per-layer profile in the published 9.362 um surface volume
             ((max - min) / (max + min) over the 28 layers)
  ink_d    = strongest standardized difference between ink and background pixels over the layers
             (mean_ink - mean_bg) / std_bg
Spearman correlation across tiles and segments.

Usage: python ink_vs_clarity.py 20250831000000-w040_2025083102 20260108000000-w041_2026010816 ...
       python ink_vs_clarity.py --plot ink_vs_clarity.json ink_vs_clarity.png
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shard_v3 import read_sharded_2d  # noqa: E402
from vcz import BUCKET, ZArray, list_prefixes  # noqa: E402

VOL = "9.362um-1.2m-113keV-volume-20250728140407"


def analyse(seg, tile=128, min_px=300):
    base = f"{BUCKET}/PHerc0139/segments/{seg}"
    lab_dir = f"{base}/ink-labels/9.362um-volume-20250728140407"
    dates = [p.rstrip('/').split('/')[-1] for p in list_prefixes(f"PHerc0139/segments/{seg}/ink-labels/9.362um-volume-20250728140407/")]
    d = sorted(dates)[-1]
    lab = read_sharded_2d(f"{lab_dir}/{d}/inklabels.zarr/0") > 0
    sup = read_sharded_2d(f"{lab_dir}/{d}/supervision.zarr/0") > 0
    sv = ZArray(f"{base}/surface-volumes/{VOL}.zarr/0", workers=32)
    L, H, W = sv.shape
    rows = []
    for y in range(0, H - tile + 1, tile):
        for x in range(0, W - tile + 1, tile):
            s = sup[y:y + tile, x:x + tile]
            if s.mean() < 0.5:
                continue
            ink = lab[y:y + tile, x:x + tile] & s
            bg = (~lab[y:y + tile, x:x + tile]) & s
            if ink.sum() < min_px or bg.sum() < min_px:
                continue
            v = sv.read(0, L, y, y + tile, x, x + tile).astype(np.float32)
            if (v == 0).mean() > 0.05:
                continue
            prof = v.reshape(L, -1).mean(1)
            clarity = (prof.max() - prof.min()) / (prof.max() + prof.min() + 1e-6)
            dl = []
            for l in range(L):
                a = v[l][ink]; b = v[l][bg]
                dl.append((a.mean() - b.mean()) / (b.std() + 1e-6))
            dl = np.array(dl)
            rows.append(dict(seg=seg, y=y, x=x, clarity=float(clarity), ink_d=float(dl[np.argmax(np.abs(dl))]),
                             ink_d_abs=float(np.abs(dl).max()), best_layer=int(np.argmax(np.abs(dl))),
                             ink_frac=float(ink.sum() / s.sum())))
    return rows


SEG_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def plot(rows, out):
    """Scatter of tile clarity against raw ink contrast, one colour per segment (fixed order)."""
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    segs = list(dict.fromkeys(t["seg"] for t in rows))
    rho, p = spearmanr([t["clarity"] for t in rows], [t["ink_d_abs"] for t in rows])
    fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for i, sg in enumerate(segs):
        rs = [t for t in rows if t["seg"] == sg]
        m = re.search(r"-(w\d+)_", sg)
        ax.scatter([t["clarity"] for t in rs], [t["ink_d_abs"] for t in rs], s=16, color=SEG_COLORS[i % 8],
                   edgecolor="#fcfcfb", linewidth=0.5, label=m.group(1) if m else sg, zorder=3)
    ax.set_xlabel("sheet clarity of the tile (layer-profile contrast, 28 layers)", fontsize=8, color="#52514e")
    ax.set_ylabel("raw ink contrast |d| (ink vs background, best layer)", fontsize=8, color="#52514e")
    ax.set_title(f"PHerc0139 at 9.36 µm, {len(rows)} labelled tiles: Spearman ρ = {rho:.2f} (p = {p:.2f})",
                 fontsize=9, color="#0b0b0b")
    ax.grid(color="#e4e3de", lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#e4e3de")
    ax.tick_params(colors="#52514e", labelsize=7.5)
    ax.legend(title="segment", fontsize=7, title_fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, facecolor="#fcfcfb")
    plt.close(fig)


def report(allrows):
    rho, p = spearmanr([t["clarity"] for t in allrows], [t["ink_d_abs"] for t in allrows])
    print(f"ALL: {len(allrows)} tiles, spearman = {rho:.2f} (p={p:.3g})")
    c = np.array([t["clarity"] for t in allrows]); dd = np.array([t["ink_d_abs"] for t in allrows])
    q = np.quantile(c, [0.33, 0.66])
    for name, sel in [("low clarity", c <= q[0]), ("mid", (c > q[0]) & (c <= q[1])), ("high clarity", c > q[1])]:
        print(f"  {name}: median |ink d| = {np.median(dd[sel]):.2f} (n={sel.sum()})")


if __name__ == "__main__":
    # python ink_vs_clarity.py <segment> [<segment> ...]      -> ink_vs_clarity.json (+ stats)
    # python ink_vs_clarity.py --plot <json> <png>           -> figure and stats from a saved run
    if len(sys.argv) >= 4 and sys.argv[1] == "--plot":
        allrows = json.load(open(sys.argv[2]))
        report(allrows)
        plot(allrows, sys.argv[3])
        sys.exit(0)
    segs = sys.argv[1:]
    if not segs:
        sys.exit(__doc__)
    allrows = []
    for sg in segs:
        r = analyse(sg)
        allrows += r
        if len(r) >= 5:
            rho, p = spearmanr([t["clarity"] for t in r], [t["ink_d_abs"] for t in r])
            print(f"{sg}: {len(r)} tiles, spearman(clarity, |ink d|) = {rho:.2f} (p={p:.3g})", flush=True)
        else:
            print(f"{sg}: {len(r)} tiles", flush=True)
    json.dump(allrows, open("ink_vs_clarity.json", "w"), indent=1)
    if len(allrows) >= 5:
        report(allrows)
        plot(allrows, "ink_vs_clarity.png")
