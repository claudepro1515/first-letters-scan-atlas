"""First Letters readiness matrix for the 22 eligible scrolls + generated spiral-scroll.json files.

Inputs: catalog_report.json (catalog_check.py), scanq_<scroll>.json (scanq.py), and the public
community umbilicus repo (github.com/AlexeyDrobkovStrikesBack/herculaneum-umbilici, MIT).
Writes: readiness.csv, readiness.json and spiral-scroll/<scroll>.json for every scroll that has
the spiral-fitting inputs.

spiral-scroll.json fields filled from the data itself:
  voxel_size_um          <- volume name (e.g. 20250821151803-9.362um-...)
  normal_zarr_group      <- the group the Lasagna job itself uses for nx/ny (<scroll>.lasagna.json)
  lasagna_scale          <- that group's real downsample factor from the store's .zattrs
  paths.tracks_dbm       <- the track store published under dl.ash2txt.org/datasets/spiral_datasets
  spiral_outward_sense   <- only when published (PHerc0826: CW, villa spiral-fitting README);
                            otherwise left as "CW|ACW" with a note, never guessed
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

ELIGIBLE = ["PHerc0125", "PHerc0175A", "PHerc0175B", "PHerc0191", "PHerc0211", "PHerc0257", "PHerc0268",
            "PHerc0306B", "PHerc0343", "PHerc0358", "PHerc0483A", "PHerc0483B", "PHerc0490A", "PHerc0490B",
            "PHerc0800", "PHerc0813", "PHerc0826", "PHerc0846A", "PHerc0846B", "PHerc1203", "PHerc1218",
            "PHerc1545"]  # PHerc1447 removed 2026-09-24: letters found (villa #1887)
KNOWN_SENSE = {"PHerc0826": ("CW", "villa spiral-fitting/README.md example")}
COMMUNITY_UMB = "https://github.com/AlexeyDrobkovStrikesBack/herculaneum-umbilici"
# public end-to-end or survey runs we could find (Sep 24 2026); letters found: none so far
PUBLIC_RUNS = {
    "PHerc0826": "Miller & Müller (Aug 2026 award, null); ShribyrLabs/vesuvius-reports (null)",
    "PHerc0211": "armando-gaona/pherc0211-first-letters-free-compute (null); bnleft/first-light-pherc0211 (null)",
    "PHerc0125": "villa PR #1837 recipe: 1,500-slice fit (no ink readout found)",
    "PHerc1447": "TAUIL-Abd-Elilah/pherc1447-ink-survey",
}


def main(report="catalog_report.json", scanq_dir="scanq", umb_repo=None, out="."):
    rep = {r["scroll"]: r for r in json.load(open(report))}
    comm = set()
    if umb_repo and os.path.isdir(umb_repo):
        comm = {os.path.basename(p).split("_")[0] for p in glob.glob(os.path.join(umb_repo, "PHerc*_umbilicus.json"))}
    rows = []
    os.makedirs(os.path.join(out, "spiral-scroll"), exist_ok=True)
    for s in ELIGIBLE:
        r = rep.get(s, {})
        vols = r.get("volumes", {})
        vname = next((v for v in vols if "um-1.2m" in v), next(iter(vols), ""))
        vox = vols.get(vname, {}).get("voxel_um")
        shape = vols.get(vname, {}).get("shape")
        las = (r.get("lasagna") or [{}])[0]
        grp = las.get("spiral_normal_zarr_group")
        scale = las.get("spiral_lasagna_scale")
        sd = r.get("spiral_dataset") or {}
        tracks = None
        for vid, files in (sd.get("tracks") or {}).items():
            dbm = [f for f in files if f.endswith(".dbm")]
            if dbm:
                tracks = f"tracks/{dbm[0]}"
        umb = "official" if r.get("umbilicus") else ("community (manual)" if s in comm else "none")
        q = {}
        qp = os.path.join(scanq_dir, f"scanq_{s}.json")
        if os.path.exists(qp):
            q = json.load(open(qp))
        ready = bool(tracks and grp and umb != "none")
        row = dict(
            scroll=s, volume=vname, voxel_um=vox,
            height_mm=round(shape[0] * vox / 1000, 1) if shape and vox else None,
            surfaces="yes" if r.get("surfaces") else "no",
            lasagna="yes" if grp else "no",
            normal_zarr_group=grp, lasagna_scale=scale,
            tracks="yes" if tracks else "no",
            umbilicus=umb,
            segments=None,
            sheet_modulation=round(q["modulation_median"], 3) if q.get("modulation_median") else None,
            coherence=round(q["coherence_median"], 2) if q.get("coherence_median") else None,
            fog_fraction=round(q["fog_fraction"], 2) if q.get("fog_fraction") is not None else None,
            spiral_inputs_complete="yes" if ready else "no",
            sense=KNOWN_SENSE.get(s, ("unknown", ""))[0],
            public_runs=PUBLIC_RUNS.get(s, ""),
        )
        rows.append(row)
        if tracks and grp:
            spec = {
                "schema_version": 1,
                "name": s,
                "voxel_size_um": vox,
                "spiral_outward_sense": KNOWN_SENSE.get(s, ("CW|ACW",))[0],
                "normal_zarr_group": str(grp),
                "lasagna_scale": int(scale) if scale and float(scale).is_integer() else scale,
                "paths": {"tracks_dbm": tracks},
                "_generated_by": "first-letters-scan-atlas tools/readiness.py",
                "_notes": {
                    "normal_zarr_group": f"group used by the Lasagna job for nx/ny; its .zattrs scale is {scale}",
                    "umbilicus": umb if umb != "community (manual)" else f"community manual umbilicus: {COMMUNITY_UMB}",
                    "spiral_outward_sense": (f"published: {KNOWN_SENSE[s][1]}" if s in KNOWN_SENSE else
                                             "not published for this scroll: choose CW or ACW after checking in VC3D"),
                },
            }
            json.dump(spec, open(os.path.join(out, "spiral-scroll", f"{s}.json"), "w"), indent=2)
    with open(os.path.join(out, "readiness.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(os.path.join(out, "readiness.json"), "w"), indent=1)
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="First Letters readiness matrix + generated spiral-scroll.json files")
    ap.add_argument("--report", default="catalog_report.json", help="output of catalog_check.py")
    ap.add_argument("--scanq", default="out", help="folder with scanq_<scroll>.json files (optional)")
    ap.add_argument("--umbilici", default=None, help="local clone of the community umbilici repo (optional)")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    if a.umbilici and not glob.glob(os.path.join(a.umbilici, "PHerc*_umbilicus.json")):
        print(f"warning: no PHerc*_umbilicus.json files in {a.umbilici}; community umbilici will be missing", file=sys.stderr)
    if not a.umbilici:
        print("note: --umbilici not given; only official umbilici are counted", file=sys.stderr)
    rows = main(a.report, a.scanq, a.umbilici, a.out)
    for r in rows:
        print(r["scroll"], r["voxel_um"], "las", r["lasagna"], r["normal_zarr_group"], r["lasagna_scale"], "tracks", r["tracks"],
              "umb", r["umbilicus"], "mod", r["sheet_modulation"], "ready", r["spiral_inputs_complete"])
