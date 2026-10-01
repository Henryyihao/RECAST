"""Synthetic revision-process prior.

A sample is a *final* series y[0..T) (the settled truth) together with an age view
A[u, a] = value of position u as reported `a` steps after u (a = 0..L), NaN = not yet
released, and a release index R[u] (grid index of the first report containing u).  The
conventions match fpd.vintage so that RR-Bench and synthetic data share one code path.

Base signals
------------
  kernel_synth   GP with random composite kernels (linear / RBF / periodic / RQ / noise)
  epi_renewal    renewal-equation epidemic waves with NegBin observation
  seasonal_arma  trend + Fourier seasonality + ARMA noise, level shifts, heteroscedasticity
  real_pool      random crops of real series (optional pool, never RR-Bench data)

Revision mechanisms (composable)
--------------------------------
  backfill       delayed cumulative reporting: Dirichlet-multinomial delay distribution,
                 drifting parameters, regime switches, report-weekday effects, removals
  noise          estimate revisions converging to the truth: AR(1)-in-age noise with
                 geometrically decaying scale and decaying systematic bias, sparse revision ages
  benchmark      piecewise-constant level errors corrected at benchmark dates
  adjusted       raw provisional (spikes / dropouts / stale repeats) replaced by cleaned final
  none           no revision (the model must learn *not* to correct)
modifiers: trailing-window smoothing of the provisional signal, ratio of two backfilled
counts, publication lag, sporadic late first releases.
"""
from __future__ import annotations
import numpy as np

FREQS = ["D", "W", "M", "Q", "h"]
FREQ_P = np.array([0.36, 0.30, 0.10, 0.04, 0.20])
SEASON = {"D": [7, 365], "W": [52], "M": [12], "Q": [4], "h": [24, 168]}
# per-frequency protocol constants (mirroring RR-Bench)
PROTO = {
    "D": dict(C=256, H=28, Nmax=28, Lrange=(20, 90)),
    "W": dict(C=104, H=8, Nmax=12, Lrange=(3, 16)),
    "M": dict(C=120, H=12, Nmax=6, Lrange=(2, 6)),
    "Q": dict(C=60, H=4, Nmax=4, Lrange=(2, 5)),
    "h": dict(C=336, H=48, Nmax=24, Lrange=(24, 96)),
}
L_CHOICES = {  # emphasise the settlement ages that occur in practice
    "D": [40, 60, 75, 80, 90, 30, 50, 70],
    "W": [4, 8, 10, 12, 16, 6, 5, 3],
    "M": [3, 2, 4, 6, 5],
    "Q": [3, 2, 4, 5],
    "h": [72, 48, 24, 96],
}


def sample_L(rng, freq):
    if rng.random() < 0.7:
        return int(rng.choice(L_CHOICES[freq]))
    lo, hi = PROTO[freq]["Lrange"]
    return int(rng.integers(lo, hi + 1))


def nowcast_window(freq, L):
    """Scored nowcast window of the protocol (last positions)."""
    return int(min(L, PROTO[freq]["Nmax"]))


def model_nowcast_window(freq, L, cap=96):
    """Nowcast window produced by the model: all immature positions (age < L), capped."""
    return int(min(L, cap))


# ============================================================================ base signals
def _kernel(rng, T, freq):
    t = np.arange(T, dtype=np.float64)
    kinds = rng.choice(["lin", "rbf", "per", "rq", "wn"], size=rng.integers(1, 4), p=[0.15, 0.3, 0.3, 0.15, 0.1])
    K = None
    for k in kinds:
        if k == "lin":
            c = rng.uniform(-1, 1)
            Kk = np.outer(t / T - c, t / T - c)
        elif k == "rbf":
            ls = T * 10 ** rng.uniform(-1.7, -0.2)
            Kk = np.exp(-0.5 * ((t[:, None] - t[None, :]) / ls) ** 2)
        elif k == "per":
            p = float(rng.choice(SEASON[freq] + [rng.uniform(3, T / 3)]))
            ls = 10 ** rng.uniform(-0.5, 0.7)
            Kk = np.exp(-2 * np.sin(np.pi * np.abs(t[:, None] - t[None, :]) / p) ** 2 / ls**2)
        elif k == "rq":
            ls = T * 10 ** rng.uniform(-1.5, -0.3)
            al = 10 ** rng.uniform(-1, 1)
            Kk = (1 + ((t[:, None] - t[None, :]) ** 2) / (2 * al * ls**2)) ** (-al)
        else:
            Kk = np.eye(T) * 10 ** rng.uniform(-3, -1)
        Kk = Kk * 10 ** rng.uniform(-0.5, 0.5)
        if K is None:
            K = Kk
        else:
            K = K + Kk if rng.random() < 0.7 else K * Kk
    K = K + 1e-6 * np.eye(T)
    Lc = np.linalg.cholesky(K)
    z = Lc @ rng.standard_normal(T)
    z = (z - z.mean()) / (z.std() + 1e-9)
    return z


def _epi(rng, T, freq):
    """Renewal process with time-varying reproduction number; returns positive intensity."""
    gi_mean = {"D": rng.uniform(3, 8), "W": rng.uniform(0.6, 1.5), "M": rng.uniform(0.3, 0.8), "Q": 0.3, "h": rng.uniform(24, 96)}[freq]
    k = 10
    gi = np.arange(1, k + 1, dtype=np.float64)
    w = np.exp(-0.5 * ((gi - gi_mean) / max(gi_mean * 0.5, 0.4)) ** 2) + 1e-6
    w /= w.sum()
    # log R_t: mean-reverting random walk with occasional wave-inducing jumps
    logR = np.zeros(T)
    sig = rng.uniform(0.01, 0.08) * (1 if freq in ("D", "h") else 2)
    phi = rng.uniform(0.95, 0.995)
    for i in range(1, T):
        logR[i] = phi * logR[i - 1] + sig * rng.standard_normal() + (rng.standard_normal() * 0.3 if rng.random() < 0.01 else 0)
    logR = np.clip(logR, -0.8, 0.8)
    base = 10 ** rng.uniform(0.5, 3.5)
    I = np.zeros(T)
    I[:k] = base * np.exp(rng.standard_normal(k) * 0.1)
    for i in range(k, T):
        lam = np.exp(logR[i]) * np.dot(w, I[i - k:i][::-1]) + base * 0.02
        I[i] = min(max(lam, 1e-3), base * 3000)
    # keep magnitudes in a realistic range
    if I.max() > 10 ** rng.uniform(4, 6.5):
        I = I / I.max() * 10 ** rng.uniform(3, 5.5)
    # seasonal forcing
    if rng.random() < 0.5:
        p = float(rng.choice(SEASON[freq]))
        I = I * (1 + rng.uniform(0.05, 0.5) * np.sin(2 * np.pi * np.arange(T) / p + rng.uniform(0, 2 * np.pi)))
    return I


def _arma(rng, T, freq):
    t = np.arange(T, dtype=np.float64)
    y = np.zeros(T)
    # trend
    if rng.random() < 0.7:
        drift = rng.normal(0, 0.02)
        y += np.cumsum(drift + rng.normal(0, rng.uniform(0, 0.05), T))
    # Fourier seasonality
    for p in SEASON[freq]:
        if p < T / 2 and rng.random() < 0.8:
            nh = rng.integers(1, 4)
            for h in range(1, nh + 1):
                amp = rng.uniform(0, 1) / h
                y += amp * np.sin(2 * np.pi * h * t / p + rng.uniform(0, 2 * np.pi))
    # ARMA(2,1)
    ar = np.array([rng.uniform(0.2, 1.3), rng.uniform(-0.5, 0.2)])
    if abs(ar.sum()) > 0.98:
        ar *= 0.95 / abs(ar.sum())
    e = rng.standard_normal(T) * rng.uniform(0.1, 1.0)
    ma = rng.uniform(-0.5, 0.5)
    n = np.zeros(T)
    for i in range(2, T):
        n[i] = ar[0] * n[i - 1] + ar[1] * n[i - 2] + e[i] + ma * e[i - 1]
    y += n
    # level shifts
    if rng.random() < 0.3:
        for _ in range(rng.integers(1, 3)):
            s = rng.integers(T // 5, T)
            y[s:] += rng.normal(0, 2)
    y = (y - y.mean()) / (y.std() + 1e-9)
    return y


class RealPool:
    """Pool of real series (list of 1-D arrays) with random crops."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.series = list(z["series"])
        self.freq = list(z["freq"]) if "freq" in z else [None] * len(self.series)
        self.name = list(z["name"]) if "name" in z else ["x"] * len(self.series)
        self.by_freq = {}  # freq -> {source -> [idx]}
        for i, (f, nm) in enumerate(zip(self.freq, self.name)):
            self.by_freq.setdefault(str(f), {}).setdefault(str(nm), []).append(i)

    def sample(self, rng, T, freq):
        groups = self.by_freq.get(freq)
        if not groups:
            groups = {k: v for d in self.by_freq.values() for k, v in d.items()}
        keys = list(groups.keys())
        for _ in range(20):
            idxs = groups[keys[rng.integers(len(keys))]]  # sources uniformly, then a series
            s = self.series[idxs[rng.integers(len(idxs))]].astype(np.float64)
            s = s[np.isfinite(s)]
            if len(s) >= T:
                st = rng.integers(0, len(s) - T + 1)
                x = s[st:st + T]
                if np.nanstd(x) > 0:
                    return x
        return None


def sample_base(rng, T, freq, pool: RealPool | None = None, p_real: float = 0.4):
    """Returns (y, kind) with y strictly positive count-like or real-valued (kind='pos' or 'real')."""
    r = rng.random()
    p_real = p_real if pool is not None else 0.0
    if r < p_real:
        x = pool.sample(rng, T, freq)
        if x is not None:
            # random affine and sign
            if np.nanmin(x) >= 0 and rng.random() < 0.85:
                scale = 10 ** rng.uniform(-1, 2)
                y = x * scale
                return np.maximum(y, 0) , "pos"
            x = (x - x.mean()) / (x.std() + 1e-9)
            return x * 10 ** rng.uniform(-1, 3) + rng.normal(0, 1) * 10 ** rng.uniform(0, 3), "real"
        r = rng.uniform(p_real, 1)
    r = (r - p_real) / (1 - p_real)
    if r < 0.35:
        z = _kernel(rng, T, freq)
    elif r < 0.7:
        return _epi(rng, T, freq), "pos"
    else:
        z = _arma(rng, T, freq)
    if rng.random() < 0.7:  # positive, count-like
        level = 10 ** rng.uniform(0.7, 4.5)
        y = level * np.exp(rng.uniform(0.2, 0.9) * z)
        return y, "pos"
    return z * 10 ** rng.uniform(-1, 3) + rng.normal(0, 1) * 10 ** rng.uniform(0, 3), "real"


# ============================================================================ delay distributions
def _delay_pmf(rng, D, family=None):
    """pmf over delays 0..D (length D+1)."""
    d = np.arange(D + 1, dtype=np.float64)
    family = family or rng.choice(["nb", "lognorm", "geom0", "mix"], p=[0.35, 0.25, 0.2, 0.2])
    if family == "nb":
        m = 10 ** rng.uniform(np.log10(0.3), np.log10(max(0.5, 0.5 * D)))
        r = 10 ** rng.uniform(-0.5, 1.0)
        p = r / (r + m)
        from scipy.special import gammaln
        logp = gammaln(d + r) - gammaln(r) - gammaln(d + 1) + r * np.log(p) + d * np.log1p(-p)
        pmf = np.exp(logp - logp.max())
    elif family == "lognorm":
        mu = rng.uniform(np.log(0.5), np.log(max(0.6, 0.5 * D)))
        s = rng.uniform(0.3, 1.2)
        pmf = np.exp(-0.5 * ((np.log(d + 0.5) - mu) / s) ** 2) / (d + 0.5)
    elif family == "geom0":
        p0 = rng.uniform(0.05, 0.9)
        q = rng.uniform(0.2, 0.9)
        pmf = np.concatenate([[p0], (1 - p0) * (1 - q) * q ** d[:-1]])
    else:
        a = _delay_pmf(rng, D, "nb"); b = _delay_pmf(rng, D, "lognorm")
        w = rng.uniform(0.2, 0.8)
        pmf = w * a + (1 - w) * b
    pmf = pmf / pmf.sum()
    # put a random immediate-report mass
    if rng.random() < 0.4:
        p0 = rng.uniform(0.0, 0.8)
        pmf = (1 - p0) * pmf; pmf[0] += p0
    return pmf


def _weekday_weights(rng):
    w = np.ones(7)
    w[5:7] *= rng.uniform(0.05, 0.8, size=2)  # weekend under-reporting
    if rng.random() < 0.5:
        w[0] *= rng.uniform(1.0, 2.5)  # Monday catch-up
    w *= rng.uniform(0.7, 1.3, size=7)
    return w


# ============================================================================ mechanisms
def mech_backfill(rng, y, L, freq, kind):
    """Cumulative delayed reporting. Returns age view A [T, L+1] and revision-only meta."""
    T = len(y)
    D = L
    pmf0 = _delay_pmf(rng, D)
    # time variation: log-mean-delay random walk + regime switch
    drift_sig = rng.uniform(0, 0.03) if rng.random() < 0.7 else 0.0
    shift = np.cumsum(rng.normal(0, drift_sig, T))
    if rng.random() < 0.3:
        ts = rng.integers(T // 4, T)
        shift[ts:] += np.log(10 ** rng.uniform(-0.4, 0.4))
    # completion "speed" modulation: reshape pmf by stretching delays: p_u(d) ∝ pmf0(d * exp(-shift_u))
    d = np.arange(D + 1, dtype=np.float64)
    dow = _weekday_weights(rng) if (freq == "D" and rng.random() < 0.6) else None
    hod = None
    if freq == "h" and rng.random() < 0.4:
        hod = np.ones(24); hod[0:6] *= rng.uniform(0.1, 0.7)
    kappa = 10 ** rng.uniform(0.7, 2.7)  # Dirichlet concentration (overdispersion)
    removal_q = rng.uniform(0.0, 0.08) if rng.random() < 0.3 else 0.0
    A = np.full((T, L + 1), np.nan)
    cdf0 = np.cumsum(pmf0)
    for u in range(T):
        # stretched pmf
        if drift_sig > 0 or shift[u] != 0:
            s = np.exp(-shift[u])
            # interpolate cdf at d*s
            c = np.interp(d * s, d, cdf0, left=0.0, right=1.0)
            c[-1] = 1.0
            p = np.diff(np.concatenate([[0.0], c]))
        else:
            p = pmf0.copy()
        if dow is not None:
            p = p * dow[(u + d.astype(int)) % 7]
        if hod is not None:
            p = p * hod[(u + d.astype(int)) % 24]
        p = np.maximum(p, 1e-12); p /= p.sum()
        if kind == "pos":
            n_tot = int(np.round(y[u]))
            if n_tot > 0:
                pp = rng.dirichlet(np.maximum(kappa * p, 1e-3))
                if n_tot < 2_000_000:
                    cnt = rng.multinomial(n_tot, pp).astype(np.float64)
                else:
                    cnt = n_tot * pp
            else:
                cnt = np.zeros(D + 1)
            if removal_q > 0:  # later removals of early reports
                rem = rng.binomial(cnt[:-1].astype(int), removal_q).astype(np.float64)
                shiftd = rng.integers(1, D + 1, size=len(rem))
                for j in range(len(rem)):
                    if rem[j] > 0:
                        cnt[min(j + shiftd[j], D)] -= rem[j]
            cum = np.cumsum(cnt)
            A[u, :] = cum
            A[u, L] = n_tot  # settled = total (removals count as final)
            if removal_q > 0:
                A[u, L] = cum[-1]
        else:  # continuous "completion" of a real-valued quantity
            F = np.cumsum(p)
            eps = rng.normal(0, 0.02, D + 1) * np.sqrt(np.clip(1 - F, 0, 1))
            A[u, :] = y[u] * (F + eps)
            A[u, L] = y[u]
    return A


def mech_noise(rng, y, L, kind):
    """Estimate revisions converging to the truth (macro-style)."""
    T = len(y)
    scale = np.nanstd(y) + 1e-9
    lvl = np.abs(y) + 1e-9
    rel = rng.random() < 0.6 and kind == "pos"
    sig0 = 10 ** rng.uniform(-2.3, -0.7)
    rho = rng.uniform(0.4, 0.95)
    phi = rng.uniform(0.3, 0.95)
    b0 = rng.normal(0, 0.03) if rng.random() < 0.6 else 0.0
    sparse = rng.random() < 0.5
    ages = np.arange(L + 1)
    if sparse and L > 2:
        rev_ages = np.sort(rng.choice(np.arange(1, L), size=rng.integers(1, min(L - 1, 4) + 1), replace=False))
    else:
        rev_ages = np.arange(1, L)
    A = np.empty((T, L + 1))
    for u in range(T):
        e = np.zeros(L + 1)
        e[0] = rng.standard_normal()
        for a in range(1, L + 1):
            e[a] = phi * e[a - 1] + np.sqrt(1 - phi**2) * rng.standard_normal()
        sig = sig0 * rho ** ages
        bias = b0 * rho ** ages
        z = sig * e + bias
        z[L] = 0.0
        if sparse:  # hold value between revision ages
            keep = np.zeros(L + 1, bool); keep[0] = True; keep[rev_ages] = True; keep[L] = True
            last = 0
            for a in range(L + 1):
                if keep[a]:
                    last = a
                z[a] = z[last]
        if rel:
            A[u] = y[u] * np.exp(z)
        else:
            A[u] = y[u] + z * scale
    A[:, L] = y
    return A


def mech_benchmark(rng, A, y, L):
    """Piecewise-constant level errors in report time, corrected at benchmark dates."""
    T = len(y)
    n_b = rng.integers(1, 4)
    bdates = np.sort(rng.integers(0, T + L, size=n_b))
    tau = 10 ** rng.uniform(-2.3, -1.2)
    deltas = rng.normal(0, tau, size=n_b + 1)
    rel = np.nanmin(y) > 0
    scale = np.nanstd(y) + 1e-9
    for u in range(T):
        rep = u + np.arange(L)
        k = np.searchsorted(bdates, rep, side="right")
        if rel:
            A[u, :L] = A[u, :L] * np.exp(deltas[k])
        else:
            A[u, :L] = A[u, :L] + deltas[k] * scale * 10
    return A


def mech_adjusted(rng, y, L):
    """Provisional raw data with anomalies replaced by cleaned final at a random age."""
    T = len(y)
    scale = np.nanstd(y) + 1e-9
    p = rng.uniform(0.01, 0.12)
    a_star = int(rng.integers(1, L + 1))
    prov = y.copy()
    for u in range(T):
        if rng.random() < p:
            r = rng.random()
            if r < 0.4:
                prov[u] = y[u] * rng.uniform(1.5, 5) if y[u] > 0 else y[u] + rng.uniform(2, 6) * scale
            elif r < 0.7:
                prov[u] = 0.0 if rng.random() < 0.5 else y[u] * rng.uniform(0, 0.4)
            else:
                prov[u] = prov[u - 1] if u > 0 else y[u]
    if rng.random() < 0.5:  # small measurement noise on the raw feed
        prov = prov + rng.normal(0, 10 ** rng.uniform(-3, -1.5), T) * scale
    A = np.empty((T, L + 1))
    A[:, :a_star] = prov[:, None]
    A[:, a_star:] = y[:, None]
    return A


def smooth_age_view(A, w):
    """Trailing w-step mean of the provisional signal as published (Delphi-style smoothing)."""
    T, L1 = A.shape
    L = L1 - 1
    S = np.full_like(A, np.nan)
    for a in range(L1):
        acc = np.zeros(T); ok = np.ones(T, bool)
        for j in range(w):
            col = A[:, min(a + j, L)]
            sh = np.full(T, np.nan); sh[j:] = col[:T - j]
            acc += np.nan_to_num(sh); ok &= np.isfinite(sh)
        S[:, a] = np.where(ok, acc / w, np.nan)
    return S


def apply_release(rng, A, lag, T):
    """Publication lag and sporadic late first releases -> R and masking of unreleased cells."""
    R = np.arange(T, dtype=np.float64) + lag
    if rng.random() < 0.2:
        late = rng.random(T) < rng.uniform(0.01, 0.1)
        R[late] += rng.integers(1, 4, size=late.sum())
    L = A.shape[1] - 1
    u = np.arange(T)
    for a in range(L):
        A[(u + a) < R, a] = np.nan
    return A, R


MECHS = ["backfill", "noise", "adjusted", "benchmark", "none"]
MECH_P = np.array([0.35, 0.25, 0.10, 0.10, 0.20])


def sample_revision(rng, y, kind, L, freq, mech_filter=None, modifiers=True):
    """Sample a full revision process for final series y. Returns (A, R, info).
    mech_filter: optional list of allowed mechanism names (prior-component ablations)."""
    T = len(y)
    p = MECH_P.copy()
    if mech_filter is not None:
        p = np.array([p[i] if m in mech_filter else 0.0 for i, m in enumerate(MECHS)])
        p = p / p.sum()
    mech = str(rng.choice(MECHS, p=p))
    info = {"mech": mech}
    if mech == "backfill":
        yy = y if kind == "pos" else np.abs(y)
        A = mech_backfill(rng, yy, L, freq, kind)
        if modifiers and freq == "D" and rng.random() < 0.4:
            A = smooth_age_view(A, 7); info["smooth"] = 7
        if modifiers and rng.random() < 0.15 and kind == "pos":  # ratio of two backfilled counts
            den = mech_backfill(rng, yy * 10 ** rng.uniform(0.3, 1.5) + 1, L, freq, "pos")
            A = 100 * A / np.maximum(den, 1)
            info["mech"] = "ratio"
        if modifiers and rng.random() < 0.3 and kind == "pos":
            A = mech_benchmark(rng, A, A[:, L], L); info["benchmark"] = True
    elif mech == "noise":
        A = mech_noise(rng, y, L, kind)
        if modifiers and rng.random() < 0.4:
            A = mech_benchmark(rng, A, y, L); info["benchmark"] = True
    elif mech == "adjusted":
        A = mech_adjusted(rng, y, L)
    elif mech == "benchmark":
        A = mech_noise(rng, y, L, kind)
        A = mech_benchmark(rng, A, y, L)
    else:
        A = np.repeat(y[:, None], L + 1, axis=1)
    lag = 0 if (not modifiers or rng.random() < 0.65) else int(rng.integers(1, 4))
    if modifiers and freq in ("M", "Q") and rng.random() < 0.5:
        lag = max(lag, 1)
    A, R = apply_release(rng, A, lag, T)
    info["lag"] = lag
    return A.astype(np.float32), R, info


def sample_triangle(rng, freq=None, T=None, pool=None, L=None, mech_filter=None, modifiers=True, p_real=0.4):
    freq = freq or str(rng.choice(FREQS, p=FREQ_P))
    pr = PROTO[freq]
    L = L if L is not None else sample_L(rng, freq)
    C = pr["C"]
    T = T or int(C + L + pr["H"] + rng.integers(0, C // 2))
    y, kind = sample_base(rng, T, freq, pool, p_real=p_real)
    A, R, info = sample_revision(rng, y, kind, L, freq, mech_filter=mech_filter, modifiers=modifiers)
    info.update(freq=freq, L=L, kind=kind)
    return A, R, info


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    import time
    t0 = time.time()
    for i in range(20):
        A, R, info = sample_triangle(rng)
        print(info, A.shape, np.nanmean(np.abs(A[:, 0] - A[:, -1]) / (np.abs(A[:, -1]) + 1e-9)))
    print("per sample", (time.time() - t0) / 20)
