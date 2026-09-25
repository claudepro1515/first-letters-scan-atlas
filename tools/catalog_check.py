"""Readiness + consistency check of the public Vesuvius Challenge open-data bucket.

For every scroll prefix: volumes, surface predictions, Lasagna predictions, umbilicus, spiral
dataset (dl.ash2txt.org). Reports what each First Letters / spiral-fitting input looks like and
flags inconsistencies (shape mismatches between a volume and its predictions, OME-Zarr scale
metadata that disagrees with the real level shapes, umbilicus points outside the volume, ...).
Only metadata (.zattrs/.zarray/json) is read: a full run is a few thousand small requests.
"""
from __future__ import annotations

import json
import math
import re
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcz import BUCKET, get_json, http_get, list_keys, list_prefixes  # noqa: E402

ELIGIBLE = ["PHerc0125", "PHerc0175A", "PHerc0175B", "PHerc0191", "PHerc0211", "PHerc0257", "PHerc0268",
            "PHerc0306B", "PHerc0343", "PHerc0358", "PHerc0483A", "PHerc0483B", "PHerc0490A", "PHerc0490B",
            "PHerc0800", "PHerc0813", "PHerc0826", "PHerc0846A", "PHerc0846B", "PHerc1203", "PHerc1218",
            "PHerc1545"]  # PHerc1447 removed 2026-09-24: letters found (villa #1887)
SPIRAL = "https://dl.ash2txt.org/datasets/spiral_datasets"


def zarr_levels(url):
    """Return {path: (shape, chunks, dtype, compressor, scale_from_zattrs)} for an OME-Zarr group."""
    za = get_json(url + "/.zattrs") or {}
    out = {}
    dsets = []
    for ms in za.get("multiscales", []):
        dsets += ms.get("datasets", [])
    for ds in dsets:
        p = str(ds["path"])
        sc = None
        for t in ds.get("coordinateTransformations", []):
            if t.get("type") == "scale":
                sc = t["scale"]
        za_ = get_json(f"{url}/{p}/.zarray")
        if za_ is None:
            out[p] = dict(missing=True, scale=sc)
            continue
        comp = za_.get("compressor") or {}
        out[p] = dict(shape=za_["shape"], chunks=za_["chunks"], dtype=za_["dtype"],
                      compressor=(comp.get("id"), comp.get("cname")), scale=sc,
                      sep=za_.get("dimension_separator", "."))
    return za, out


def check_pyramid(levels, base_shape, label, issues, base_scale=1.0):
    for p, d in levels.items():
        if d.get("missing"):
            issues.append(f"{label}: level {p} listed in .zattrs but .zarray missing")
            continue
        sc = d["scale"]
        if sc is None:
            issues.append(f"{label}: level {p} has no scale transform")
            continue
        exp = [math.ceil(b / (s / base_scale)) for b, s in zip(base_shape, sc)]
        exp_floor = [b // int(round(s / base_scale)) for b, s in zip(base_shape, sc)]
        if d["shape"] != exp and d["shape"] != exp_floor:
            issues.append(f"{label}: level {p} shape {d['shape']} != base {base_shape} / scale {sc} (expected {exp})")


def spiral_dataset(scroll):
    html = http_get(f"{SPIRAL}/{scroll}/")
    if html is None:
        return None
    subs = re.findall(r'href="([^"?/][^"?]*)/"', html.decode(errors="ignore"))
    res = {"folders": subs}
    for vid in subs:
        h2 = http_get(f"{SPIRAL}/{scroll}/{vid}/")
        if h2:
            res.setdefault("contents", {})[vid] = re.findall(r'href="([^"?][^"?]*)"', h2.decode(errors="ignore"))
        th = http_get(f"{SPIRAL}/{scroll}/{vid}/tracks/")
        if th:
            files = re.findall(r'href="([^"?][^"?]*)"', th.decode(errors="ignore"))
            res.setdefault("tracks", {})[vid] = files
            ex = [f for f in files if f.endswith(".extract.json")]
            if ex:
                res.setdefault("extract", {})[vid] = get_json(f"{SPIRAL}/{scroll}/{vid}/tracks/{ex[0]}")
    return res


def check_scroll(scroll):
    issues, info = [], {"scroll": scroll, "eligible": scroll in ELIGIBLE}
    vols = {}
    for p in list_prefixes(f"{scroll}/volumes/"):
        name = p.rstrip("/").split("/")[-1]
        m = re.match(r"(\d+)-([\d.]+)um-([\d.]+)m-(\d+)keV", name)
        za, lv = zarr_levels(f"{BUCKET}/{p.rstrip('/')}")
        l0 = lv.get("0", {})
        vols[name] = dict(id=m.group(1) if m else None, voxel_um=float(m.group(2)) if m else None,
                          shape=l0.get("shape"), chunks=l0.get("chunks"), levels=sorted(lv))
        if l0.get("shape"):
            check_pyramid(lv, l0["shape"], f"volume {name}", issues)
    info["volumes"] = vols
    byid = {v["id"]: (k, v) for k, v in vols.items()}
    preds = [p.rstrip("/").split("/")[-1] for p in list_prefixes(f"{scroll}/representations/predictions/")]
    info["prediction_types"] = preds
    # surfaces
    surf = []
    for p in list_prefixes(f"{scroll}/representations/predictions/surfaces/"):
        name = p.rstrip("/").split("/")[-1]
        if not name.endswith(".zarr"):
            continue
        za, lv = zarr_levels(f"{BUCKET}/{p.rstrip('/')}")
        l0 = lv.get("0", {})
        vid = name.split("-")[0]
        surf.append(dict(name=name, shape=l0.get("shape"), levels=sorted(lv)))
        if vid in byid and l0.get("shape") and byid[vid][1]["shape"] and l0["shape"] != byid[vid][1]["shape"]:
            issues.append(f"surface {name}: shape {l0['shape']} != volume {byid[vid][0]} {byid[vid][1]['shape']}")
        if vid not in byid:
            issues.append(f"surface {name}: no volume with id {vid} in bucket")
        if l0.get("shape"):
            check_pyramid(lv, l0["shape"], f"surface {name}", issues)
    info["surfaces"] = surf
    # lasagna
    las = []
    for p in list_prefixes(f"{scroll}/representations/predictions/lasagna/"):
        d = p.rstrip("/")
        name = d.split("/")[-1]
        vid = name.split("-")[0]
        keys = [k for k, _ in list_keys(d + "/") if k.endswith(".lasagna.json")]
        meta = get_json(f"{BUCKET}/{keys[0]}") if keys else None
        entry = dict(name=name, has_json=bool(meta))
        if meta:
            base = meta.get("base_shape_zyx")
            entry["base_shape_zyx"] = base
            entry["groups"] = {g: v.get("zarr") for g, v in (meta.get("groups") or {}).items()}
            if vid in byid and base and byid[vid][1]["shape"] and list(base) != list(byid[vid][1]["shape"]):
                issues.append(f"lasagna {name}: base_shape_zyx {base} != volume {byid[vid][1]['shape']}")
            stores = {}
            for g, v in (meta.get("groups") or {}).items():
                zpath = v.get("zarr", "")
                store, _, grp = zpath.rpartition("/")
                za, lv = zarr_levels(f"{BUCKET}/{d}/{store}")
                if not lv:
                    issues.append(f"lasagna {name}: store {store} missing or has no multiscales")
                    continue
                if grp not in lv:
                    issues.append(f"lasagna {name}: group '{g}' points to {zpath} but level '{grp}' not in store")
                if base:
                    check_pyramid(lv, base, f"lasagna {name}/{store}", issues)
                sc = lv.get(grp, {}).get("scale")
                stores[g] = dict(store=store, group=grp, scale=sc, levels=sorted(lv),
                                 shape=lv.get(grp, {}).get("shape"))
            entry["stores"] = stores
            # the spiral fitter's (normal_zarr_group, lasagna_scale) pair: nx/ny group and its true factor
            nx = stores.get("nx")
            if nx and nx.get("scale"):
                entry["spiral_normal_zarr_group"] = nx["group"]
                entry["spiral_lasagna_scale"] = nx["scale"][-1]
        las.append(entry)
    info["lasagna"] = las
    # umbilicus
    um = []
    for k, _ in list_keys(f"{scroll}/representations/umbilicus/"):
        if not k.endswith(".json"):
            continue
        u = get_json(f"{BUCKET}/{k}")
        pts = u.get("control_points", []) if isinstance(u, dict) else []
        md = u.get("metadata", {}) if isinstance(u, dict) else {}
        e = dict(key=k, n=len(pts))
        if pts:
            zs = [p["z"] for p in pts]
            e.update(zmin=min(zs), zmax=max(zs))
            vname = md.get("volume")
            if vname in vols and vols[vname]["shape"]:
                Z, Y, X = vols[vname]["shape"]
                bad = [p for p in pts if not (0 <= p["x"] < X and 0 <= p["y"] < Y and 0 <= p["z"] < Z)]
                if bad:
                    issues.append(f"umbilicus {k}: {len(bad)} control points outside volume {vname}")
                e["z_coverage"] = round((max(zs) - min(zs)) / Z, 3)
                if md.get("voxelsize_um") and vols[vname]["voxel_um"] and abs(md["voxelsize_um"] - vols[vname]["voxel_um"]) > 1e-3:
                    issues.append(f"umbilicus {k}: voxelsize_um {md['voxelsize_um']} != volume name {vols[vname]['voxel_um']}")
            elif vname:
                issues.append(f"umbilicus {k}: metadata.volume {vname} not found in bucket")
        um.append(e)
    info["umbilicus"] = um
    # spiral dataset
    sd = spiral_dataset(scroll)
    info["spiral_dataset"] = None
    if sd:
        info["spiral_dataset"] = {"folders": sd.get("folders"), "tracks": sd.get("tracks"),
                                  "contents": sd.get("contents")}
        for vid, ex in (sd.get("extract") or {}).items():
            au = (ex or {}).get("array_url", "")
            m = re.match(r"s3://vesuvius-challenge-open-data/(.*)/(\d+)$", au)
            if m:
                zarr_path = m.group(1)
                if get_json(f"{BUCKET}/{zarr_path}/{m.group(2)}/.zarray") is None:
                    issues.append(f"spiral dataset {vid}: extract.json array_url {au} does not exist")
            zb = (ex or {}).get("effective_z_bounds")
            vk = [k for k, v in vols.items() if v["id"] == vid]
            if zb and vk and vols[vk[0]]["shape"] and zb[1] > vols[vk[0]]["shape"][0]:
                issues.append(f"spiral dataset {vid}: z_bounds {zb} beyond volume depth {vols[vk[0]]['shape'][0]}")
    info["issues"] = issues
    return info


def main():
    scrolls = [p.rstrip("/") for p in list_prefixes("") if p.startswith("PHerc")]
    only = sys.argv[1:] or scrolls
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(check_scroll, only))
    json.dump(res, open("catalog_report.json", "w"), indent=1)
    for r in res:
        print(r["scroll"], "E" if r["eligible"] else " ", "vols", len(r["volumes"]), "surf", len(r["surfaces"]),
              "las", len(r["lasagna"]), "umb", len(r["umbilicus"]), "spiral", bool(r["spiral_dataset"]),
              "ISSUES:", len(r["issues"]))
        for i in r["issues"]:
            print("    -", i)


if __name__ == "__main__":
    main()
