from __future__ import annotations
import numpy as np


def age_grid(L: int, K: int = 16) -> list[int]:


    if L <= K:
        return list(range(L))
    if K < 16:
        ages = sorted(set(int(a) for a in np.round(np.geomspace(1, L, K) - 1).astype(int) if 0 <= a < L))
        return ages[:K]
    base = list(range(8))
    rest = np.unique(np.round(np.geomspace(8, L - 1, K - 8)).astype(int))
    ages = sorted(set(base) | set(int(a) for a in rest if a < L))
    return ages[:K]


def build_streams(A: np.ndarray, R: np.ndarray, t: int, L: int, C: int, N: int, H: int,
                  ages: list[int] | None = None, with_target: bool = True, edge: bool = True, n_views: int = 16):


    ages = age_grid(L, n_views) if ages is None else ages
    T = A.shape[0]
    P = C + H
    u = np.arange(t - C + 1, t + H + 1)
    valid = (u >= 0) & (u < T)
    uu = np.clip(u, 0, T - 1)
    Rv = np.where(np.isfinite(R), R, np.inf)[uu]
    S = 2 + len(ages)
    vals = np.full((S, P), np.nan, dtype=np.float32)
    age_raw = np.zeros((S, P), dtype=np.float32)

    ok = valid & (u <= t - L) & (Rv <= t)
    vals[0, ok] = A[uu[ok], L]
    age_raw[0, ok] = L

    a_t = np.minimum(t - u, L)
    ok = valid & (u <= t) & (Rv <= t)
    vals[1, ok] = A[uu[ok], a_t[ok]]
    age_raw[1, ok] = a_t[ok]
    for k, a in enumerate(ages):
        ok = valid & (u <= t - a) & (Rv <= u + a)
        vals[2 + k, ok] = A[uu[ok], a]
        age_raw[2 + k, ok] = a
    mask = np.isfinite(vals).astype(np.float32)
    vals = np.where(mask > 0, vals, np.nan).astype(np.float32)
    age_raw = age_raw * mask

    prof = revision_profile_in_context(A, R, t, L, C, ages)
    rev = np.zeros((S, P), dtype=np.float32)
    rev[1] = np.where(mask[1] > 0, prof[np.clip(a_t, 0, L)], 0.0)
    for k, a in enumerate(ages):
        rev[2 + k] = np.where(mask[2 + k] > 0, prof[a], 0.0)
    out = dict(vals=vals, mask=mask, age_raw=age_raw, rev=rev, ctx_len=C if edge else C - N, L=L, N=N,
               stream_age=np.array([L, 0] + list(ages), dtype=np.float32))
    if with_target:
        tu = np.arange(t - N + 1, t + H + 1)
        tv = (tu >= 0) & (tu < T)
        tgt = np.full(N + H, np.nan, dtype=np.float32)
        tgt[tv] = A[tu[tv], L]
        out["target"] = tgt

        oc = np.full(C, np.nan, dtype=np.float32)
        okc = valid[:C] & (u[:C] <= t)
        oc[okc] = A[uu[:C][okc], L]
        out["oracle_ctx"] = oc
    return out


def revision_profile_in_context(A, R, t, L, C, ages, min_n=5, eps=1e-9):


    T = A.shape[0]
    u = np.arange(max(0, t - C + 1), t - L + 1)
    Rv = np.where(np.isfinite(R), R, np.inf)
    u = u[(u < T) & (Rv[np.clip(u, 0, T - 1)] <= t)] if len(u) else u
    prof = np.full(L + 1, np.nan)
    if len(u) >= min_n:
        fin = A[u, L].astype(np.float64)
        denom = np.abs(fin) + eps
        cand = sorted(set(list(ages) + [L]))
        for a in cand:
            va = A[u, a].astype(np.float64)
            ok = np.isfinite(va) & np.isfinite(fin) & (u + a <= t)
            if ok.sum() >= min_n:
                d = np.median(np.abs(va[ok] - fin[ok]) / denom[ok])
                prof[a] = np.log1p(100 * min(d, 1.0)) / np.log1p(100)
        prof[L] = 0.0
    if np.isfinite(prof).sum() >= 1:
        idx = np.arange(L + 1); ok = np.isfinite(prof)
        prof = np.interp(idx, idx[ok], prof[ok])
    else:
        prof = np.full(L + 1, -1.0)
    return prof.astype(np.float32)


def as_of_series(A, R, t, L, C):

    T = A.shape[0]
    u = np.arange(t - C + 1, t + 1)
    valid = (u >= 0) & (u < T)
    uu = np.clip(u, 0, T - 1)
    Rv = np.where(np.isfinite(R), R, np.inf)[uu]
    x = np.full(C, np.nan, dtype=np.float32)
    ok = valid & (Rv <= t)
    x[ok] = A[uu[ok], np.minimum(t - u, L)[ok]]
    return x


def settled_series(A, R, t, L, C):

    T = A.shape[0]
    u = np.arange(t - C + 1, t + 1)
    valid = (u >= 0) & (u < T)
    uu = np.clip(u, 0, T - 1)
    Rv = np.where(np.isfinite(R), R, np.inf)[uu]
    x = np.full(C, np.nan, dtype=np.float32)
    ok = valid & (Rv <= t)
    x[ok] = A[uu[ok], L]
    return x
