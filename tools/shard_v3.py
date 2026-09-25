"""Tiny reader for 2D zarr v3 arrays stored with sharding_indexed + blosc (as used for ink labels)."""
import json
import numpy as np
from numcodecs import Blosc
from vcz import http_get


def read_sharded_2d(url):
    meta = json.loads(http_get(url + "/zarr.json"))
    H, W = meta["shape"]
    sh = meta["chunk_grid"]["configuration"]["chunk_shape"]
    sc = meta["codecs"][0]["configuration"]
    ih, iw = sc["chunk_shape"]
    out = np.zeros((H, W), np.uint8)
    nsy, nsx = (H + sh[0] - 1) // sh[0], (W + sh[1] - 1) // sh[1]
    blosc = Blosc()
    for sy in range(nsy):
        for sx in range(nsx):
            raw = http_get(f"{url}/c/{sy}/{sx}")
            if raw is None:
                continue
            ny, nx = sh[0] // ih, sh[1] // iw
            n = ny * nx
            idx = np.frombuffer(raw[-(n * 16 + 4):-4], dtype="<u8").reshape(n, 2)
            for k in range(n):
                off, nb = idx[k]
                if off == 2 ** 64 - 1 or nb == 2 ** 64 - 1:
                    continue
                buf = blosc.decode(raw[off:off + nb])
                blk = np.frombuffer(buf, np.uint8).reshape(ih, iw)
                y0 = sy * sh[0] + (k // nx) * ih
                x0 = sx * sh[1] + (k % nx) * iw
                if y0 >= H or x0 >= W:
                    continue
                out[y0:min(H, y0 + ih), x0:min(W, x0 + iw)] = blk[:min(ih, H - y0), :min(iw, W - x0)]
    return out
