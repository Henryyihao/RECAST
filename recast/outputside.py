from __future__ import annotations
import numpy as np
from .protocol import LEVELS


class RLS:
    def __init__(self, p: int, lam: float = 0.99, ridge: float = 1.0):
        self.P = np.eye(p) / ridge
        self.beta = np.zeros(p)
        self.lam = lam

    def predict(self, phi):
        return float(phi @ self.beta)

    def update(self, phi, y):
        Pphi = self.P @ phi
        k = Pphi / (self.lam + phi @ Pphi)
        self.beta = self.beta + k * (y - phi @ self.beta)
        self.P = (self.P - np.outer(k, Pphi)) / self.lam


class OutputSideBaselines:


    def __init__(self, tasks, states, Qn, Y, S, L, H, index, m_edge: int = 8, half_life: float | None = None):
        self.tasks, self.states, self.Qn, self.Y, self.S, self.L, self.H, self.index, self.m = tasks, states, Qn, Y, S, L, H, index, m_edge
        self.hl = half_life

    def run(self):
        H, L, m = self.H, self.L, self.m
        out = {k: np.empty_like(self.Qn) for k in ("b2f", "ra_prov", "ra_settled")}
        row = 0
        for tk in self.tasks:
            n = len(tk.origins)
            xs = tk.contexts("asof")
            A, R = tk.A, tk.R
            c = self.states[tk.sid].c
            lg = lambda v: np.log(np.maximum(v, 0) + c)
            lam = 0.5 ** (3.0 / self.hl)
            rls = [RLS(m + 2, lam=lam, ridge=5.0) for _ in range(H)]
            pend_b2f = []
            ew_prov = np.zeros(H); ew_prov_n = np.zeros(H)
            ew_set = np.zeros(H); ew_set_n = np.zeros(H)
            pend_set = []
            pend_prov = []
            for i in range(n):
                t = int(tk.origins[i]); r = row + i
                s = self.S[r]; qn = self.Qn[r]; med = qn[:, 4]

                keep = []
                for due, h, phi, tgt in pend_b2f:
                    if due <= t: rls[h].update(phi, tgt)
                    else: keep.append((due, h, phi, tgt))
                pend_b2f = keep
                keep = []
                for due, h, res in pend_set:
                    if due <= t:
                        ew_set[h] = lam * ew_set[h] + res; ew_set_n[h] = lam * ew_set_n[h] + 1
                    else: keep.append((due, h, res))
                pend_set = keep
                keep = []
                for t0, h, med0, s0 in pend_prov:
                    u = t0 + h + 1
                    if u <= t and u < len(R) and np.isfinite(R[u]) and R[u] <= t:
                        yprov = A[u, min(t - u, L)]
                        if np.isfinite(yprov):
                            res = lg(yprov) - lg(med0)
                            ew_prov[h] = lam * ew_prov[h] + res; ew_prov_n[h] = lam * ew_prov_n[h] + 1

                    elif u > t:
                        keep.append((t0, h, med0, s0))
                    elif u < len(R) and (not np.isfinite(R[u]) or R[u] > t):
                        keep.append((t0, h, med0, s0))
                pend_prov = keep

                x = xs[i].astype(np.float64)
                fin = np.flatnonzero(np.isfinite(x))
                ref = x[fin[-min(len(fin), L + 1)]] if len(fin) else 0.0
                edge = x[fin[-m:]] if len(fin) >= m else np.pad(x[fin], (m - len(fin), 0), constant_values=ref)
                phi_base = np.concatenate([[1.0], lg(edge) - lg(ref)])
                for h in range(H):
                    phi = np.concatenate([phi_base, [lg(med[h]) - lg(ref)]])
                    fac = np.exp(np.clip(rls[h].predict(phi), -2, 2))
                    out["b2f"][r, h] = (qn[h] + c) * fac - c
                    fp = np.exp(np.clip(ew_prov[h] / ew_prov_n[h], -2, 2)) if ew_prov_n[h] > 0 else 1.0
                    fs = np.exp(np.clip(ew_set[h] / ew_set_n[h], -2, 2)) if ew_set_n[h] > 0 else 1.0
                    out["ra_prov"][r, h] = (qn[h] + c) * fp - c
                    out["ra_settled"][r, h] = (qn[h] + c) * fs - c
                    y = self.Y[r, h]
                    if np.isfinite(y):
                        pend_b2f.append((t + h + 1 + L, h, phi, lg(y) - lg(med[h])))
                        pend_set.append((t + h + 1 + L, h, lg(y) - lg(med[h])))
                    pend_prov.append((t, h, med[h], s))
            row += n
        for k in out:
            out[k] = np.sort(out[k], axis=-1).astype(np.float32)
        return out
