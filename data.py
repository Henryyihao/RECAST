"""On-the-fly training batches: homogeneous (freq, L) batches of age-view streams from synthetic revision processes."""
from __future__ import annotations
import numpy as np, torch
from torch.utils.data import IterableDataset, get_worker_info
from .synth import sample_triangle, sample_L, nowcast_window, model_nowcast_window, PROTO, FREQS, FREQ_P, RealPool, sample_base, sample_revision
from .views import build_streams, age_grid


def batch_size_for(freq, L, token_budget=14000, ages=None, patch=16, min_bs=8, n_views=16):
    S = 2 + len(age_grid(L, n_views) if ages is None else ages)
    pr = PROTO[freq]; C, H = pr["C"], pr["H"]; N = model_nowcast_window(freq, L)
    n_tok = (C - N) // patch + 1 + int(np.ceil((N + H) / patch)) if patch > 1 else C + H  # point-wise models: one token per position
    return int(max(min_bs, min(64, token_budget // (S * n_tok))))


class TriangleBatches(IterableDataset):
    def __init__(self, pool_path=None, seed=0, token_budget=14000, freq_p=None, mech_filter=None, ctx_drop=0.3,
                 fixed_freq=None, fixed_L=None, nmax=None, modifiers=True, ages=None, edge=True, p_real=0.4, no_rev=False, full_window=True,
                 patch=16, min_bs=8, n_views=16):
        self.pool_path, self.seed, self.token_budget = pool_path, seed, token_budget
        self.n_views = n_views
        self.patch, self.min_bs = patch, min_bs
        self.freq_p = np.asarray(freq_p) if freq_p is not None else FREQ_P
        self.mech_filter = mech_filter  # optional callable(rng) -> override of the mechanism draw (for prior ablations)
        self.ctx_drop = ctx_drop
        self.fixed_freq, self.fixed_L = fixed_freq, fixed_L
        self.nmax = nmax or {}
        self.modifiers = modifiers
        self.ages = ages
        self.edge = edge
        self.p_real = p_real
        self.no_rev = no_rev
        self.full_window = full_window

    def __iter__(self):
        wi = get_worker_info()
        wid = wi.id if wi else 0
        rng = np.random.default_rng(self.seed * 1000 + wid)
        pool = RealPool(self.pool_path) if self.pool_path else None
        while True:
            freq = self.fixed_freq or str(rng.choice(FREQS, p=self.freq_p))
            L = self.fixed_L or sample_L(rng, freq)
            pr = PROTO[freq]; C, H = pr["C"], pr["H"]; N = min(model_nowcast_window(freq, L) if self.full_window else nowcast_window(freq, L), self.nmax.get(freq, 10**9))
            B = batch_size_for(freq, L, self.token_budget, self.ages, self.patch, self.min_bs, self.n_views)
            items = []
            while len(items) < B:
                try:
                    A, R, info = sample_triangle(rng, freq=freq, L=L, pool=pool, mech_filter=self.mech_filter, modifiers=self.modifiers, p_real=self.p_real)
                except Exception:
                    continue
                T = A.shape[0]
                lo = max(L + 8, C // 4); hi = T - H
                if hi <= lo:
                    continue
                t = int(rng.integers(lo, hi + 1))
                st = build_streams(A, R, t, L, C, N, H, ages=self.ages, edge=self.edge, n_views=self.n_views)
                if self.ctx_drop > 0 and rng.random() < self.ctx_drop:  # random shorter context
                    cut = int(rng.integers(0, C - N - 48)) if C - N > 60 else 0
                    if self.edge and cut > 0:
                        cut = int(rng.integers(0, C - 48))
                    st["vals"][:, :cut] = np.nan; st["mask"][:, :cut] = 0; st["age_raw"][:, :cut] = 0; st["rev"][:, :cut] = 0
                if self.no_rev:
                    st["rev"][:] = 0.0
                if not np.isfinite(st["target"]).any() or np.isfinite(st["vals"][1]).sum() < 16:
                    continue
                if not np.isfinite(st["vals"]).any():
                    continue
                items.append(st)
            b = {k: torch.tensor(np.stack([it[k] for it in items])) for k in ["vals", "mask", "age_raw", "rev", "stream_age", "target", "oracle_ctx"]}
            b["L"] = torch.full((B,), L, dtype=torch.long); b["ctx_len"] = C if self.edge else C - N; b["N"] = N
            b["freq"] = freq
            yield b
