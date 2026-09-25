"""Figures and tables for the scan-quality atlas (reads scanq_<scroll>.json/.npz)."""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ELIGIBLE = ["PHerc0125", "PHerc0175A", "PHerc0175B", "PHerc0191", "PHerc0211", "PHerc0257", "PHerc0268",
            "PHerc0306B", "PHerc0343", "PHerc0358", "PHerc0483A", "PHerc0483B", "PHerc0490A", "PHerc0490B",
            "PHerc0800", "PHerc0813", "PHerc0826", "PHerc0846A", "PHerc0846B", "PHerc1203", "PHerc1218",
            "PHerc1545"]  # PHerc1447 removed 2026-09-24: letters found (villa #1887)
# Scrolls where ink or letters have been found, at any scan resolution. One rule for all of them: the
# "textFound" field of villa scrollprize.org/src/data/atlasOverlay.json at commit 75c79ac (2026-09-24)
#   "Title found":  PHerc0139, PHerc0172
#   "Ink detected": PHerc0009B, PHerc0343P, PHerc0500P2, PHerc0814, PHerc0841
# plus PHerc1447, removed from First Letters eligibility that day because letters were found (villa #1887).
# Letters in PHerc0814 and PHerc0841 were reported in ~2 um scans ("Multiple scrolls now show Greek
# letters", Vesuvius Challenge, 9 Oct 2025); the atlas measures the ~9 um scans of the same scrolls.
INK_FOUND = {"PHerc0009B", "PHerc0139", "PHerc0172", "PHerc0343P", "PHerc0500P2", "PHerc0814", "PHerc0841",
             "PHerc1447"}
LETTERS_REPORTED = INK_FOUND
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8a86"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"


def load(d):
    """Load scanq outputs; the .npz files may sit next to the .json files or in d/profiles/."""
    rows = {}
    npz_dirs = [d, os.path.join(d, "profiles")]
    for f in sorted(os.listdir(d)):
        if f.startswith("scanq_") and f.endswith(".json"):
            s = f[6:-5]
            npz = next((os.path.join(nd, f"scanq_{s}.npz") for nd in npz_dirs
                        if os.path.exists(os.path.join(nd, f"scanq_{s}.npz"))), None)
            if npz is None:
                continue
            z = np.load(npz)
            j = json.load(open(os.path.join(d, f)))
            rows[s] = (j, {k: z[k] for k in z.files})
    return rows


def group(s):
    if s in INK_FOUND:
        return "ink or letters already found (any resolution)", ORANGE
    if s in ELIGIBLE:
        return "First Letters eligible (no letters yet)", BLUE
    return "other scans", GRAY


def _protocol(vol):
    """Short scan-protocol label from a volume name (e.g. '9.36 µm, 1.2 m, 113 keV')."""
    import re
    um = re.search(r"-([\d.]+)um", vol)
    dist = re.search(r"-([\d.]+)m-", vol)
    kev = re.search(r"-(\d+)keV", vol)
    u = float(um.group(1)) if um else None
    ulab = "9.36 µm" if u and abs(u - 9.36) < 0.02 else (f"{round(u, 2):g} µm" if u else "?")
    return ", ".join(x for x in [ulab, f"{dist.group(1)} m" if dist else None, f"{kev.group(1)} keV" if kev else None] if x)


def fig_ranking(rows, out, key="modulation", xlabel=None,
                title="How clearly papyrus sheets show up in each CT scan, by scan protocol"):
    """One row per scan, grouped by scan protocol (compare within a group), sorted by the median."""
    items = []
    for s, (j, c) in rows.items():
        good = c["coherence"] > 0.3
        m = c[key][good]
        vol = j.get("volume", "")
        items.append((np.median(m), np.percentile(m, 25), np.percentile(m, 75), s, vol, _protocol(vol)))
    blocks = {}
    for it in items:
        blocks.setdefault(it[5], []).append(it)
    def _um(p):
        try:
            return float(p.split(" ")[0])
        except ValueError:
            return 0.0
    order = sorted(blocks, key=lambda p: (-len(blocks[p]), -_um(p)))
    n_rows = len(items) + 2 * len(order)
    fig, ax = plt.subplots(figsize=(8.6, 0.28 * n_rows + 1.4), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    y = n_rows - 1
    placed = []
    xmax = max(it[2] for it in items) + 0.05
    for p in order:
        ax.text(-0.004, y, f"{p} ({len(blocks[p])} scan{'s' if len(blocks[p]) > 1 else ''})", va="center",
                ha="right", fontsize=8.5, color=INK, fontweight="bold")
        ax.plot([0, xmax], [y - 0.55, y - 0.55], color=GRID, lw=0.8)
        y -= 1
        for med, q1, q3, s, vol, _ in sorted(blocks[p], reverse=True):
            lab, col = group(s)
            ax.plot([q1, q3], [y, y], color=col, lw=2, solid_capstyle="round", alpha=0.55)
            ax.scatter([med], [y], s=38, color=col, zorder=3, edgecolor=SURF, linewidth=1.2)
            ax.text(q3 + 0.004, y, f"{med:.3f}", va="center", fontsize=7.5, color=INK2)
            ax.text(-0.004, y, s, va="center", ha="right", fontsize=8, color=INK)
            placed.append((med, q1, q3, s, vol))
            y -= 1
        y -= 1
    ax.set_yticks([])
    ax.set_xlim(0, xmax)
    ax.set_ylim(y + 0.5, n_rows)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlabel(xlabel or f"{key} — dot: median, bar: middle 50 % of profiles", fontsize=8, color=INK2)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l) for l, c in
               [group("PHerc0139"), group("PHerc0125"), group("PHerc1451")]]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    fig.subplots_adjust(left=0.3, right=0.97, top=0.95, bottom=0.07)
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)
    return sorted(placed)


def fig_height(rows, out, nz=12, key="modulation", vmin=0.1, vmax=0.4, label="median sheet modulation"):
    names = sorted(rows, key=lambda s: np.median(rows[s][1][key][rows[s][1]["coherence"] > 0.3]))
    M = np.full((len(names), nz), np.nan)
    for i, s in enumerate(names):
        c = rows[s][1]
        good = c["coherence"] > 0.3
        zb = c["zbin"]
        for k in range(nz):
            sel = good & (zb == k)
            if sel.sum() >= 15:
                M[i, k] = np.median(c[key][sel])
    fig, ax = plt.subplots(figsize=(7.2, 0.26 * len(names) + 1.5), dpi=150)
    fig.patch.set_facecolor(SURF)
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.5, color=INK)
    ax.set_xticks(range(nz))
    ax.set_xticklabels([f"{int(100 * k / nz)}–{int(100 * (k + 1) / nz)}" for k in range(nz)], fontsize=6.5, rotation=45, color=INK2)
    ax.set_xlabel("height band (% of scan height, bottom → top)", fontsize=8, color=INK2)
    for i in range(len(names)):
        for k in range(nz):
            if np.isfinite(M[i, k]):
                ax.text(k, i, f"{M[i, k]:.2f}"[1:], ha="center", va="center", fontsize=5.5,
                        color="#ffffff" if M[i, k] > vmin + 0.6 * (vmax - vmin) else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.set_label(label, fontsize=8, color=INK2)
    ax.set_title("Where along the scroll height the sheets are clearest", fontsize=11, color=INK, loc="left")
    fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.12)
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)
    return names, M


def chunk_medians(c, cs=128, key="modulation"):
    good = c["coherence"] > 0.3
    ck = np.stack([c["z"] // cs, c["y"] // cs, c["x"] // cs], 1)
    out = {}
    for k in np.unique(ck[good], axis=0):
        sel = good & np.all(ck == k, 1)
        if sel.sum() >= 12:
            out[tuple(int(v) for v in k)] = float(np.median(c[key][sel]))
    return out


def fig_examples(rows, out, picks, cache_dir, bucket, key="modulation"):
    """picks: list of (scroll, 'high'|'low'); draws the central xy slice of that scroll's
    highest / lowest-modulation sampled chunk (full resolution, 128 x 128 voxels)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vcz import OmeZarr
    n = len(picks)
    cols = n // 2
    fig, axes = plt.subplots(2, cols, figsize=(2.3 * cols, 5.2), dpi=150)
    fig.patch.set_facecolor(SURF)
    for ax, (s, which) in zip(axes.ravel(), picks):
        j, c = rows[s]
        med = chunk_medians(c, key=key)
        ks = sorted(med, key=med.get)
        # representative rather than extreme: the chunk at the 90th / 10th percentile
        k = ks[int(round(0.9 * (len(ks) - 1)))] if which == "high" else ks[int(round(0.1 * (len(ks) - 1)))]
        g = OmeZarr(f"{bucket}/{s}/volumes/{j['volume']}", cache_dir=cache_dir)
        blk = np.asarray(g.level("0")._load_chunk(k))
        ax.imshow(blk[blk.shape[0] // 2], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        um = float(j["volume"].split("-")[1].replace("um", ""))
        ax.set_title(f"{s}\n{key} {med[k]:.2f}", fontsize=8, color=INK)
        L = 500 / um  # 0.5 mm scale bar
        ax.plot([6, 6 + L], [120, 120], color="#ffffff", lw=2.5)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    fig.text(0.01, 0.995, f"Clear: chunk at the scroll's 90th percentile of {key}", fontsize=9, color=INK, va="top")
    fig.text(0.01, 0.525, "Foggy: chunk at the scroll's 10th percentile", fontsize=9, color=INK, va="top")
    fig.text(0.99, 0.01, "128 × 128 voxel CT slices at full resolution · white bar = 0.5 mm", fontsize=7, color=INK2, ha="right")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.05, wspace=0.08, hspace=0.42)
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Atlas figures from scanq outputs")
    ap.add_argument("scanq_dir", help="folder with scanq_<scroll>.json and scanq_<scroll>.npz")
    ap.add_argument("out_dir")
    ap.add_argument("--key", default="modulation", help="modulation (default) or periodicity")
    ap.add_argument("--examples", nargs="*", default=[],
                    help="scroll:high|low pairs, an even number, first half drawn on the top row "
                         "(e.g. PHerc0139:high PHerc0826:high PHerc0268:low PHerc0490B:low)")
    ap.add_argument("--cache", default=None, help="chunk cache used by scanq (for --examples)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rows = load(a.scanq_dir)
    if not rows:
        sys.exit(f"no scanq_*.npz files found in {a.scanq_dir} or {a.scanq_dir}/profiles")
    sfx = "" if a.key == "modulation" else f"_{a.key}"
    items = fig_ranking(rows, os.path.join(a.out_dir, f"atlas_ranking{sfx}.png"), key=a.key,
                        xlabel=("sheet modulation (sheet amplitude / mean intensity) — dot: median, "
                                "bar: middle 50 % of profiles") if a.key == "modulation" else None)
    fig_height(rows, os.path.join(a.out_dir, f"atlas_height{sfx}.png"), key=a.key,
               vmin=0.0 if a.key == "periodicity" else 0.1, vmax=0.8 if a.key == "periodicity" else 0.4,
               label=f"median {a.key}")
    if a.examples:
        if len(a.examples) % 2:
            sys.exit("--examples needs an even number of scroll:high|low pairs")
        picks = [tuple(p.split(":")) for p in a.examples]
        fig_examples(rows, os.path.join(a.out_dir, f"atlas_examples{sfx}.png"), picks, a.cache,
                     "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com", key=a.key)
    for med, q1, q3, s, vol in items[::-1]:
        print(f"{s:12s} {med:.3f} [{q1:.3f},{q3:.3f}] {_protocol(vol):24s} {group(s)[0]}")
