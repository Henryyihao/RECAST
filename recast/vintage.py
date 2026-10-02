from __future__ import annotations
import numpy as np
import pandas as pd


def _finalize(A: np.ndarray, R: np.ndarray, first_vals: np.ndarray, L: int):

    T = A.shape[0]
    u = np.arange(T)
    late = np.isfinite(R) & (R > u + L)
    A[late, L] = first_vals[late]
    return A, R


def age_view_from_changes(ref_idx, rep_idx, values, T: int, L: int):

    A = np.full((T, L + 1), np.nan, dtype=np.float32)
    R = np.full(T, np.nan)
    first_vals = np.full(T, np.nan)
    ref_idx = np.asarray(ref_idx); rep_idx = np.asarray(rep_idx); values = np.asarray(values, dtype=np.float64)
    order = np.lexsort((rep_idx, ref_idx))
    ref_idx, rep_idx, values = ref_idx[order], rep_idx[order], values[order]
    keep = np.ones(len(ref_idx), dtype=bool)
    same = (ref_idx[1:] == ref_idx[:-1]) & (rep_idx[1:] == rep_idx[:-1])
    keep[:-1][same] = False
    ref_idx, rep_idx, values = ref_idx[keep], rep_idx[keep], values[keep]
    starts = np.searchsorted(ref_idx, np.arange(T), side="left")
    ends = np.searchsorted(ref_idx, np.arange(T), side="right")
    ages = np.arange(L + 1)
    for u in range(T):
        s, e = starts[u], ends[u]
        if e <= s:
            continue
        reps = rep_idx[s:e]; vals = values[s:e]
        pos = np.searchsorted(reps, u + ages, side="right") - 1
        ok = pos >= 0
        row = np.full(L + 1, np.nan); row[ok] = vals[pos[ok]]
        A[u] = row
        R[u] = reps[0]; first_vals[u] = vals[0]
    return _finalize(A, R, first_vals, L)


def age_view_from_snapshots(mat: np.ndarray, rep_grid_idx: np.ndarray, L: int):


    T, Rn = mat.shape
    ff = pd.DataFrame(mat).ffill(axis=1).to_numpy(dtype=np.float64)
    A = np.full((T, L + 1), np.nan, dtype=np.float32)
    R = np.full(T, np.nan); first_vals = np.full(T, np.nan)
    ages = np.arange(L + 1)
    fin = np.isfinite(mat)
    for u in range(T):
        if not fin[u].any():
            continue
        r0 = int(np.argmax(fin[u]))
        R[u] = rep_grid_idx[r0]; first_vals[u] = mat[u, r0]
        pos = np.searchsorted(rep_grid_idx, u + ages, side="right") - 1
        ok = pos >= 0
        row = np.full(L + 1, np.nan); row[ok] = ff[u, pos[ok]]
        A[u] = row
    return _finalize(A, R, first_vals, L)


def age_view_from_triangle(inc: np.ndarray, L: int):

    T, K = inc.shape
    x = np.nan_to_num(np.asarray(inc, dtype=np.float64), nan=0.0)
    cum = np.cumsum(x, axis=1)
    A = np.full((T, L + 1), np.nan, dtype=np.float32)
    m = min(K, L + 1)
    A[:, :m] = cum[:, :m]
    if L + 1 > K:
        A[:, K:] = cum[:, [K - 1]]
    R = np.arange(T, dtype=float)
    rowmissing = ~np.isfinite(inc).any(axis=1)
    A[rowmissing] = np.nan; R[rowmissing] = np.nan
    return A, R


def rolling_sum_age_view(A: np.ndarray, R: np.ndarray, w: int, L: int):

    T = A.shape[0]
    S = np.full((T, L + 1), np.nan, dtype=np.float32)
    for a in range(L + 1):
        acc = np.zeros(T); valid = np.ones(T, dtype=bool)
        for j in range(w):
            col = A[:, min(a + j, L)]
            shifted = np.full(T, np.nan)
            shifted[j:] = col[: T - j]
            acc += np.nan_to_num(shifted); valid &= np.isfinite(shifted)
        S[:, a] = np.where(valid, acc, np.nan)
    RS = np.full(T, np.nan)
    for u in range(w - 1, T):
        RS[u] = np.max(R[u - w + 1: u + 1])
    return S, RS


def asof_context(A: np.ndarray, R: np.ndarray, t: int, L: int) -> np.ndarray:

    u = np.arange(t + 1)
    a = np.minimum(t - u, L)
    x = A[u, a].astype(np.float64)
    x[~(R[u] <= t)] = np.nan
    return x


def revision_profile(A: np.ndarray, L: int, rel: bool = True, eps: float = 1e-9):

    fin = A[:, L]
    out = []
    for a in range(L + 1):
        d = fin - A[:, a]
        if rel:
            d = d / (np.abs(fin) + eps)
        d = d[np.isfinite(d)]
        out.append((np.median(np.abs(d)) if len(d) else np.nan, np.mean(np.abs(d)) if len(d) else np.nan, np.mean(d) if len(d) else np.nan))
    return np.asarray(out)
