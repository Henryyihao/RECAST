from __future__ import annotations
import numpy as np

LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def visible_triangle(A, R, t, L):

    T = A.shape[0]
    u = np.arange(T)
    V = A.astype(np.float64).copy()
    Rv = np.where(np.isfinite(R), R, np.inf)
    for a in range(L + 1):
        hide = (u + a > t) | (Rv > t)
        V[hide, a] = np.nan

    return V


def chain_ladder_nowcast(A, R, t, L, N, W=None, eps=1e-6, min_pairs=3):


    T = A.shape[0]
    V = visible_triangle(A, R, t, L)
    u_all = np.arange(T)
    W = W if W is not None else max(4 * N, 60)
    lo = max(0, t - L - W)
    f = np.ones(L)
    for a in range(L):
        ok = np.isfinite(V[:, a]) & np.isfinite(V[:, a + 1]) & (u_all >= lo) & (u_all <= t)
        if ok.sum() >= min_pairs:
            num = V[ok, a + 1].sum(); den = V[ok, a].sum()
            if den > eps and num > eps:
                f[a] = float(np.clip(num / den, 0.1, 10.0))

    cum = np.ones(L + 1)
    for a in range(L - 1, -1, -1):
        cum[a] = cum[a + 1] * f[a]

    resid = {a: [] for a in range(L)}
    settled = np.isfinite(V[:, L]) & (u_all >= lo)
    for us in np.flatnonzero(settled)[-W:]:
        for a in range(L):
            va = V[us, a]
            if np.isfinite(va) and va > eps and V[us, L] > eps:
                resid[a].append(np.log(V[us, L]) - np.log(va * cum[a]))

    pos = np.arange(t - N + 1, t + 1)
    q = np.full((N, len(LEVELS)), np.nan)
    point_all = np.full(T, np.nan)
    for uu in range(max(0, t - L + 1), t + 1):
        a = t - uu
        x = V[uu, a] if a <= L else np.nan
        if not np.isfinite(x):

            continue
        point_all[uu] = x * cum[a] if x > eps else x

    rel = [(uu, t - uu) for uu in range(max(0, t - L), t + 1) if np.isfinite(V[uu, t - uu])]
    x_last, a_last = (rel[-1][0], rel[-1][1]) if rel else (None, None)
    for i, uu in enumerate(pos):
        if uu < 0 or uu >= T:
            continue
        a = t - uu
        x = V[uu, a]
        if not np.isfinite(x):

            if x_last is None:
                continue
            xv = V[x_last, a_last]; a_eff = a_last
            base = xv * cum[a_eff] if xv > eps else xv
            r = np.asarray(resid[a_eff]) if a_eff < L else np.zeros(1)
            q[i] = base * np.exp(np.quantile(r, LEVELS)) if (len(r) >= 5 and xv > eps) else base
            point_all[uu] = base
            continue
        base = x * cum[a] if x > eps else x
        r = np.asarray(resid[a]) if a < L else np.zeros(1)
        if len(r) >= 5 and x > eps:
            qq = np.quantile(r, LEVELS)
            q[i] = base * np.exp(qq)
        else:
            q[i] = base
    return q, point_all, f


def repaired_context(x_asof, ages, point_all, u_ctx):

    x = x_asof.astype(np.float64).copy()
    for j, uu in enumerate(u_ctx):
        if 0 <= uu < len(point_all) and np.isfinite(point_all[uu]) and np.isfinite(x[j]):
            x[j] = point_all[uu]
    return x.astype(np.float32)
