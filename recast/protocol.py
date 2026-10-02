from __future__ import annotations
import numpy as np
from .vintage import asof_context

LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


DEFAULTS = {
    "D": dict(C=256, H=28, stride=7, season=7, burn=120),
    "W": dict(C=104, H=8, stride=1, season=52, burn=20),
    "M": dict(C=120, H=12, stride=1, season=12, burn=24),
    "Q": dict(C=60, H=4, stride=1, season=4, burn=8),
    "h": dict(C=336, H=48, stride=48, season=24, burn=24 * 14),
}


class Task:


    def __init__(self, sid: str, A: np.ndarray, R: np.ndarray, time: np.ndarray, L: int, C: int, H: int, stride: int, season: int,
                 min_hist: int | None = None, max_origins: int | None = None, min_versioned: int = 0, n_warm: int = 0):
        self.sid, self.A, self.R, self.time, self.L, self.C, self.H, self.season = sid, A, R, time, L, C, H, season
        T = A.shape[0]
        final = A[:, L]
        min_hist = min_hist or C
        u = np.arange(T)


        Rv = np.where(np.isfinite(R), R, np.inf)
        if np.isfinite(R).sum() < min_hist + H + 1:
            self.origins = np.array([], dtype=int); return

        fin_idx = np.flatnonzero(np.isfinite(final))
        t_end = int(fin_idx.max()) - H if len(fin_idx) else -1
        versioned_start = np.flatnonzero(np.isfinite(R) & (R <= u + L))
        v0 = int(versioned_start.min()) if len(versioned_start) else 0


        rel_sorted = np.sort(Rv[np.isfinite(Rv)])
        t_first = None
        for t in range(0, t_end + 1):
            if np.searchsorted(rel_sorted, t, side="right") >= min_hist and t - v0 >= min_versioned:
                t_first = t; break
        keep = []
        if t_first is not None:
            fin_tgt = np.isfinite(final).astype(float)
            cs = np.concatenate([[0.0], np.cumsum(fin_tgt)])
            for t in range(t_first, t_end + 1, stride):
                if (cs[min(t + H + 1, T)] - cs[t + 1]) / H < 0.5:
                    continue
                x = asof_context(A, R, t, L)[-C:]
                if np.isfinite(x).sum() < max(8, C // 4):
                    continue
                keep.append(t)
        origins = np.asarray(keep, dtype=int)
        if max_origins and len(origins) > max_origins:
            origins = origins[np.linspace(0, len(origins) - 1, max_origins).round().astype(int)]


        self.n_warm = 0
        if len(origins) and n_warm > 0:
            st = int(np.median(np.diff(origins))) if len(origins) > 1 else stride
            warm = []
            for k in range(1, n_warm + 1):
                tw = int(origins[0]) - k * st
                if tw < max(C // 2, 16) or tw - v0 < 0:
                    break
                if np.isfinite(final[tw + 1: tw + H + 1]).mean() < 0.5:
                    continue
                if np.isfinite(asof_context(A, R, tw, L)[-C:]).sum() < max(8, C // 4):
                    continue
                warm.append(tw)
            warm = sorted(warm)
            self.n_warm = len(warm)
            origins = np.concatenate([np.asarray(warm, dtype=int), origins])
        self.origins = origins

    def contexts(self, kind: str = "asof"):


        out = []
        for t in self.origins:
            xa = asof_context(self.A, self.R, t, self.L)
            if kind == "asof":
                x = xa
            else:
                x = self.A[: t + 1, self.L].astype(np.float64)
                x[~(self.R[: t + 1] <= t + self.L)] = np.nan
                if kind == "final":
                    x[~np.isfinite(xa)] = np.nan
            out.append(x[-self.C:].astype(np.float32))
        return out

    def ages(self):

        out = []
        for t in self.origins:
            u = np.arange(max(0, t + 1 - self.C), t + 1)
            out.append((t - u).astype(np.int32))
        return out

    def targets(self):
        final = self.A[:, self.L]
        return np.stack([final[t + 1: t + 1 + self.H] for t in self.origins])

    def eval_mask(self):
        m = np.ones(len(self.origins), dtype=bool); m[: self.n_warm] = False
        return m

    def scale(self):

        final = self.A[:, self.L]
        s = np.empty(len(self.origins))
        for i, t in enumerate(self.origins):
            y = final[max(0, t + 1 - 2 * self.C): t + 1]
            y = y[np.isfinite(y)]
            m = self.season if len(y) > 2 * self.season else 1
            d = np.abs(y[m:] - y[:-m]) if len(y) > m else np.array([np.nan])
            s[i] = np.nanmean(d) if np.isfinite(d).any() else np.nan
            if not np.isfinite(s[i]) or s[i] <= 0:
                s[i] = np.nanmean(np.abs(y)) + 1e-6 if len(y) else 1.0
        return s


def make_tasks(bundle: dict, C=None, H=None, stride=None, max_origins=None, series=None, **kw):
    d = DEFAULTS[bundle["freq"]].copy()
    if C: d["C"] = C
    if H: d["H"] = H
    if stride: d["stride"] = stride
    tasks = []
    for sid, s in bundle["series"].items():
        if series and sid not in series:
            continue
        kw.setdefault("min_versioned", d["burn"])
        kw.setdefault("n_warm", 15)
        t = Task(sid, s["A"], s["R"], s["time"], bundle["L"], d["C"], d["H"], d["stride"], d["season"], max_origins=max_origins, **kw)
        if len(t.origins) > 0:
            tasks.append(t)
    return tasks


def pinball(q: np.ndarray, y: np.ndarray, levels=LEVELS):

    y = y[..., None]
    diff = y - q
    loss = np.maximum(levels * diff, (levels - 1) * diff)
    return 2 * loss.mean(axis=-1)


def metrics(q: np.ndarray, y: np.ndarray, scale: np.ndarray, levels=LEVELS) -> dict:


    mask = np.isfinite(y) & np.isfinite(q).all(axis=-1)
    if mask.sum() == 0:
        return {}
    med = q[..., 4]
    ae = np.abs(med - y)
    mase_cells = ae / scale[:, None]
    crps_cells = pinball(q, y, levels)
    wql_num = crps_cells
    lo, hi = q[..., 0], q[..., 8]
    cov80 = ((y >= lo) & (y <= hi)).astype(float)
    width80 = (hi - lo) / scale[:, None]

    cov_levels = np.stack([(y <= q[..., k]).astype(float) for k in range(len(levels))], axis=-1)
    out = {
        "MASE": float(np.nanmean(mase_cells[mask])),
        "WQL": float(np.nansum(wql_num[mask]) / (np.nansum(np.abs(y[mask])) + 1e-12)),
        "CRPS_s": float(np.nanmean((crps_cells / scale[:, None])[mask])),
        "COV80": float(np.nanmean(cov80[mask])),
        "WIDTH80": float(np.nanmean(width80[mask])),
        "PCE": float(np.mean(np.abs(np.nanmean(cov_levels[mask], axis=0) - levels))),
        "n_cells": int(mask.sum()),
    }

    H = y.shape[1]
    out["MASE_h"] = [float(np.nanmean(mase_cells[:, h][mask[:, h]])) if mask[:, h].any() else np.nan for h in range(H)]
    out["CRPS_h"] = [float(np.nanmean((crps_cells / scale[:, None])[:, h][mask[:, h]])) if mask[:, h].any() else np.nan for h in range(H)]
    out["COV80_h"] = [float(np.nanmean(cov80[:, h][mask[:, h]])) if mask[:, h].any() else np.nan for h in range(H)]
    return out


def per_origin_loss(q, y, scale):

    mask = np.isfinite(y) & np.isfinite(q).all(axis=-1)
    c = pinball(q, y) / scale[:, None]
    c = np.where(mask, c, np.nan)
    return np.nanmean(c, axis=1)
