"""Scroll-level check of the atlas: do scans of scrolls where ink has been found show clearer sheets
than the First Letters eligible scans (no letters yet), within the same scan protocol?

Per scan, from the scanq samples (coherent profiles only, coherence > 0.3):
  modulation            median sheet amplitude / mean CT intensity of the profile (the atlas measure)
  normalized_amplitude  median sheet amplitude / (papyrus level - air level of the scan), where the
                        papyrus level is the 95th percentile of the sampled chunks' 98th percentiles and
                        the air level the 5th percentile of their 2nd percentiles (chunks that touch the
                        zeroed, masked part of the volume are left out of the air level). Unlike
                        modulation, it does not change when a scan's intensities are shifted by a constant.
  periodicity           median share of the profile's variance explained by the sheet period
Groups come from atlas_figs.INK_FOUND and atlas_figs.ELIGIBLE. For each measure: exact two-sided
Mann-Whitney test, ink found vs eligible, over all scans and within each scan protocol, and
AUC = probability that a random ink-found scan scores higher than a random eligible scan.

Usage: python scroll_level_test.py <scanq_dir> <out_data_dir> [<out_figure.png>]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atlas_figs import BLUE, ELIGIBLE, GRAY, GRID, INK, INK2, INK_FOUND, ORANGE, SURF, load  # noqa: E402
from make_report import protocol  # noqa: E402

MEASURES = [("modulation", "sheet modulation\n(amplitude / mean intensity)"),
            ("normalized_amplitude", "normalized amplitude\n(amplitude / (papyrus − air level))"),
            ("periodicity", "periodicity\n(variance share of the sheet period)")]
FRAGMENTS = {"PHerc0500P2"}  # detached fragment (atlasOverlay.json "type": "fragment")


def measures(rows):
    out = []
    for s, (j, c) in sorted(rows.items()):
        good = c["coherence"] > 0.3
        # chunks whose 2nd percentile is 0 touch the zeroed (masked) part of the volume: they say
        # nothing about the scan's air level, so they are left out of it
        p2 = c["chunk_p2"][np.isfinite(c["chunk_p2"]) & (c["chunk_p2"] > 0)]
        air = np.percentile(p2, 5) if len(p2) else 0.0
        pap = np.percentile(c["chunk_p98"][np.isfinite(c["chunk_p98"])], 95)
        out.append(dict(
            scroll=s,
            group="ink found" if s in INK_FOUND else ("eligible" if s in ELIGIBLE else "other"),
            protocol=protocol(j["volume"]),
            modulation=float(np.median(c["modulation"][good])),
            normalized_amplitude=float(np.median(c["amplitude"][good]) / (pap - air)),
            periodicity=float(np.median(c["periodicity"][good])),
            air_level=float(air), papyrus_level=float(pap),
        ))
    return out


def compare(ms, key, keep=lambda r: True):
    a = [r[key] for r in ms if r["group"] == "ink found" and keep(r)]
    b = [r[key] for r in ms if r["group"] == "eligible" and keep(r)]
    if not a or not b:
        return None
    u, p = mannwhitneyu(a, b, alternative="two-sided", method="exact")
    return dict(n_ink=len(a), n_eligible=len(b), auc=round(float(u) / (len(a) * len(b)), 3), p=round(float(p), 4))


def tests(ms):
    subsets = [("all scans", lambda r: True),
               ("9.36 µm protocol", lambda r: r["protocol"].startswith("9.36")),
               ("8.64 µm protocol", lambda r: r["protocol"].startswith("8.64")),
               ("9.36 µm protocol, without the detached fragment PHerc0500P2",
                lambda r: r["protocol"].startswith("9.36") and r["scroll"] not in FRAGMENTS)]
    res = []
    for name, keep in subsets:
        for key, _ in MEASURES:
            t = compare(ms, key, keep)
            if t:
                res.append(dict(subset=name, measure=key, **t))
    return res


def figure(ms, res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [("9.36", "ink found"), ("9.36", "eligible"), ("8.64", "ink found"), ("8.64", "eligible")]
    xpos = [0, 1, 2.6, 3.6]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=150)
    fig.patch.set_facecolor(SURF)
    for ax, (key, lab) in zip(axes, MEASURES[:2]):
        ax.set_facecolor(SURF)
        for (pr, g), x in zip(cols, xpos):
            pts = sorted([r for r in ms if r["protocol"].startswith(pr) and r["group"] == g], key=lambda r: r[key])
            col = ORANGE if g == "ink found" else BLUE
            vals = np.array([r[key] for r in pts])
            # deterministic side-by-side offsets for near-equal values
            offs = np.zeros(len(vals))
            span = (max(r[key] for r in ms) - min(r[key] for r in ms)) or 1
            for i in range(1, len(vals)):
                if vals[i] - vals[i - 1] < 0.012 * span:
                    offs[i] = 0.09 if offs[i - 1] <= 0 else -0.09
            ax.scatter(x + offs, vals, s=34, color=col, edgecolor=SURF, linewidth=1.0, zorder=3)
            if len(vals):
                ax.plot([x - 0.2, x + 0.2], [np.median(vals)] * 2, color=INK, lw=1.4, zorder=2)
            # name every ink-found scan and the two highest eligible ones; labels pushed apart vertically
            cut = np.sort(vals)[-2] if len(vals) >= 2 else -np.inf
            labs = [(v, r["scroll"].replace("PHerc", "")) for r, v in zip(pts, vals)
                    if g == "ink found" or v >= cut]
            gap = 0.045 * span
            ys = [v for v, _ in labs]
            for i in range(1, len(ys)):
                ys[i] = max(ys[i], ys[i - 1] + gap)
            for (v, name), yl in zip(labs, ys):
                ax.text(x + 0.3, yl, name, fontsize=6.5, color=INK2, va="center")
                if abs(yl - v) > 0.2 * gap:
                    ax.plot([x + 0.12, x + 0.28], [v, yl], color=GRID, lw=0.7, zorder=1)
        for pr, x0 in (("9.36", 0.5), ("8.64", 3.1)):
            t = next((t for t in res if t["measure"] == key and t["subset"] == f"{pr} µm protocol"), None)
            if t:
                ax.text(x0, 1.0, f"AUC {t['auc']:.2f}, p = {t['p']:.3f}", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{g}\n({sum(1 for r in ms if r['protocol'].startswith(pr) and r['group'] == g)})"
                            for pr, g in cols], fontsize=8, color=INK2)
        for pr, x0 in (("9.36 µm scans", 0.5), ("8.64 µm scans", 3.1)):
            ax.text(x0, -0.2, pr, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, color=INK)
        ax.set_xlim(-0.6, 4.45)
        ax.set_ylabel(lab, fontsize=8.5, color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
    fig.suptitle("Scroll level: scans of scrolls with ink found vs First Letters eligible scans, same protocol",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.2, wspace=0.28)
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    scanq_dir, outd = sys.argv[1], sys.argv[2]
    rows = load(scanq_dir)
    if not rows:
        sys.exit(f"no scanq_*.npz files found in {scanq_dir} or {scanq_dir}/profiles")
    ms = measures(rows)
    res = tests(ms)
    os.makedirs(outd, exist_ok=True)
    json.dump([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in ms],
              open(os.path.join(outd, "scroll_level_measures.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(res, open(os.path.join(outd, "scroll_level_tests.json"), "w"), indent=1, ensure_ascii=False)
    for t in res:
        print(f"{t['subset']:62s} {t['measure']:21s} ink {t['n_ink']} vs eligible {t['n_eligible']:2d}: "
              f"AUC {t['auc']:.3f}  p {t['p']:.4f}")
    if len(sys.argv) > 3:
        figure(ms, res, sys.argv[3])
