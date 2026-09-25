"""Spiral outward-sense estimation by a global least-squares 'winding coordinate' fit.

In one z-slice we look for a multivalued scalar field w (the winding coordinate) whose level
sets follow the papyrus layers (grad w orthogonal to the layer tangent) and which grows by
about one unit per layer outward. Going once around the umbilicus in +theta the field must
jump by an unknown amount s:  w(theta + 2*pi, r) = w(theta, r) + s.  We solve for w AND s by
sparse least squares. For a spiral s is -1 or +1:

  theta = atan2(y - cy, x - cx) in array index coordinates (spiral-fitting's convention).
  s = -1  <=>  a layer ends one layer further OUT after a full +theta turn  <=>  'CW'
  s = +1  <=>  'ACW'

Local errors (folds, damage) are spread over the whole slice instead of breaking a trace.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, uniform_filter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr


def polar_resample(img, cx, cy, n_theta, r0, r1, dr=1.0, order=1, cval=0.0):
    th = np.arange(n_theta) * (2 * np.pi / n_theta)
    rs = np.arange(r0, r1, dr, dtype=np.float64)
    T, R = np.meshgrid(th, rs, indexing="ij")  # [theta, r]
    X = cx + R * np.cos(T)
    Y = cy + R * np.sin(T)
    return map_coordinates(img, [Y, X], order=order, cval=cval), th, rs


def layer_fields(cos_u8, nx_u8, ny_u8, sigma=2.0):
    """Cartesian fields: cosine layer signal, validity, double-angle normal components, coherence."""
    valid = cos_u8 > 0
    nx = (nx_u8.astype(np.float32) - 128.0) / 127.0
    ny = (ny_u8.astype(np.float32) - 128.0) / 127.0
    c2 = nx * nx - ny * ny
    s2 = 2 * nx * ny
    m = valid.astype(np.float32)
    c2s = gaussian_filter(c2 * m, sigma)
    s2s = gaussian_filter(s2 * m, sigma)
    ms = gaussian_filter(m, sigma) + 1e-6
    c2s /= ms
    s2s /= ms
    coh = np.sqrt(c2s ** 2 + s2s ** 2)  # 1 = all normals agree locally
    c = (cos_u8.astype(np.float32) - 128.0) / 127.0
    amp = np.sqrt(np.maximum(uniform_filter(c * c * m, 7) / (uniform_filter(m, 7) + 1e-6), 0))
    return valid, c2s, s2s, coh, amp


def estimate_sense_slice(cos_u8, nx_u8, ny_u8, cx, cy, r_min=60, r_max=None, n_theta=1440,
                         dr=1.0, spacing_px=None, scale_weight=0.2, iters=4000, verbose=False, b_nr_min=0.0):
    H, W = cos_u8.shape
    valid, c2s, s2s, coh, amp = layer_fields(cos_u8, nx_u8, ny_u8)
    if r_max is None:
        yy, xx = np.nonzero(valid)
        r_max = float(np.percentile(np.hypot(xx - cx, yy - cy), 99.5))
    # polar fields
    V, th, rs = polar_resample(valid.astype(np.float32), cx, cy, n_theta, r_min, r_max, dr, order=0)
    C2, _, _ = polar_resample(c2s, cx, cy, n_theta, r_min, r_max, dr)
    S2, _, _ = polar_resample(s2s, cx, cy, n_theta, r_min, r_max, dr)
    CO, _, _ = polar_resample(coh, cx, cy, n_theta, r_min, r_max, dr)
    AM, _, _ = polar_resample(amp, cx, cy, n_theta, r_min, r_max, dr)
    N, M = V.shape
    ang = 0.5 * np.arctan2(S2, C2)  # normal angle (mod pi) in Cartesian frame
    nxp, nyp = np.cos(ang), np.sin(ang)
    T = th[:, None]
    R = rs[None, :]
    # polar components of the normal
    n_r = nxp * np.cos(T) + nyp * np.sin(T)
    n_t = -nxp * np.sin(T) + nyp * np.cos(T)
    # orient outward (n_r >= 0) for the scale constraint only
    sgn = np.where(n_r >= 0, 1.0, -1.0)
    n_r_o, n_t_o = n_r * sgn, n_t * sgn
    # tangent = normal rotated by 90 deg (orientation irrelevant for the orthogonality constraint)
    t_r, t_t = -n_t, n_r
    if spacing_px is None:
        spacing_px = 5.0
    g = 1.0 / spacing_px  # layers per pixel
    dth = 2 * np.pi / N

    ok = V > 0.5
    w_node = np.where(ok, 1.0, 0.0)
    # cell-centred equations between (i, j), (i+1, j), (i, j+1), (i+1, j+1)
    i = np.arange(N)[:, None].repeat(M - 1, 1)
    j = np.arange(M - 1)[None, :].repeat(N, 0)
    ip = (i + 1) % N
    seam = (i == N - 1)
    cell_ok = ok[i, j] & ok[ip, j] & ok[i, j + 1] & ok[ip, j + 1]
    # coefficients at the cell centre (average of the 4 nodes)
    def cav(F):
        return 0.25 * (F[i, j] + F[ip, j] + F[i, j + 1] + F[ip, j + 1])
    rc = rs[j] + 0.5 * dr
    tt, tr = cav(t_t), cav(t_r)
    ntt, nrr = cav(n_t_o), cav(n_r_o)
    wA = (cav(CO) * np.clip(cav(AM), 0, 1)) * cell_ok
    wB = scale_weight * cell_ok * cav(CO) * (np.abs(cav(n_r)) >= b_nr_min)
    idx = lambda a, b: a * M + b
    n_unk = N * M + 1  # last unknown is s
    s_col = N * M

    rows, cols, vals, rhs = [], [], [], []
    eq = 0

    def add_eq(coef_th, coef_r, weight, target):
        nonlocal eq
        sel = weight > 1e-6
        k = np.count_nonzero(sel)
        ii, jj, iip, sm = i[sel], j[sel], ip[sel], seam[sel]
        a = (coef_th[sel] / (2 * dth)) * weight[sel]
        b = (coef_r[sel] / (2 * dr)) * weight[sel]
        e = eq + np.arange(k)
        # d/dtheta: (w[ip,j] - w[i,j] + w[ip,j+1] - w[i,j+1]) / (2 dth)  (+ 2s/(2dth) on the seam)
        # d/dr    : (w[i,j+1] - w[i,j] + w[ip,j+1] - w[ip,j]) / (2 dr)
        terms = [
            (idx(iip, jj), a - b),
            (idx(ii, jj), -a - b),
            (idx(iip, jj + 1), a + b),
            (idx(ii, jj + 1), -a + b),
        ]
        for col, v in terms:
            rows.append(e); cols.append(col); vals.append(v)
        # seam: w[N, .] = w[0, .] + s  -> contributes (2 s)/(2 dth) * coef_th * weight
        es = e[sm]
        rows.append(es); cols.append(np.full(len(es), s_col)); vals.append(2 * a[sm])
        rhs.append(target[sel] * weight[sel])
        eq += k

    # A: grad(w) . tangent = 0     (polar: (t_t / r) dw/dtheta + t_r dw/dr = 0)
    add_eq(tt / rc, tr, wA, np.zeros_like(wA))
    # B: grad(w) . n_out = g       (weak; sets the unit = one layer)
    add_eq(ntt / rc, nrr, wB, np.full_like(wB, g))

    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    A = coo_matrix((vals, (rows, cols)), shape=(eq, n_unk)).tocsr()
    b = np.concatenate(rhs)
    sol = lsqr(A, b, atol=1e-10, btol=1e-10, iter_lim=iters, show=False)
    x = sol[0]
    s = float(x[s_col])
    res = float(np.linalg.norm(A @ x - b))
    info = dict(s=s, residual=res, n_eq=eq, n_unk=n_unk, itn=int(sol[2]), r_min=r_min, r_max=r_max,
                frac_valid=float(ok.mean()))
    if verbose:
        print(info)
    return s, x[:-1].reshape(N, M), info


def sense_from_s(s):
    return "CW" if s < 0 else "ACW"
