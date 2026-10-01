"""Forecast anchoring helpers: the frozen backbone applied to the context repaired by the model's own nowcast."""
from __future__ import annotations
import math
import torch


def naive_anchor(teacher, vals, mask, L, ctx_len, H, uses_arcsinh, target_row=1):
    """Frozen backbone on the as-of stream alone -> normalised quantiles [B, Q, H] (teacher's own loc/scale)."""
    B = vals.shape[0]
    va = vals[:, target_row:target_row + 1]; ma = mask[:, target_row:target_row + 1]
    with torch.no_grad():
        ta = teacher(va, ma, torch.zeros_like(va), torch.zeros(B, 1, device=vals.device), L, ctx_len=ctx_len, N=0, target_row=0)
    q = (ta["quantiles"].float()[:, :, :H] - ta["loc"][:, None, :]) / ta["scale"][:, None, :]
    return torch.arcsinh(q) if uses_arcsinh else q.clamp(-50, 50), ta


def repaired_anchor(teacher, vals, mask, L, ctx_len, N, H, now_med_raw, loc_t, scale_t, uses_arcsinh, target_row=1):
    """Frozen backbone on the as-of context whose last N positions are replaced by the model's nowcast medians
    (raw scale) -> quantiles normalised in the student's space (loc_t, scale_t: [B, 1] or [B, H])."""
    B, S, P = vals.shape
    dev = vals.device
    rep = vals[:, target_row].clone()
    rm = mask[:, target_row].clone()
    seg = rep[:, ctx_len - N: ctx_len]
    fill = torch.isfinite(now_med_raw)
    seg = torch.where(fill, now_med_raw.to(seg.dtype), seg)
    rep[:, ctx_len - N: ctx_len] = seg
    rm[:, ctx_len - N: ctx_len] = torch.where(fill, torch.ones_like(rm[:, ctx_len - N: ctx_len]), rm[:, ctx_len - N: ctx_len])
    rep = rep[:, None]; rm = rm[:, None]
    with torch.no_grad():
        ta = teacher(rep, rm, torch.zeros_like(rep), torch.zeros(B, 1, device=dev), L, ctx_len=ctx_len, N=0, target_row=0)
    qa_raw = ta["quantiles"].float()[:, :, :H]
    lt = loc_t if loc_t.shape[-1] == 1 else loc_t[:, -H:]
    st = scale_t if scale_t.shape[-1] == 1 else scale_t[:, -H:]
    q = (qa_raw - lt[:, None, :]) / st[:, None, :]
    return torch.arcsinh(q) if uses_arcsinh else q.clamp(-50, 50)


ARCSINH_CLIP = math.asinh(50.0)  # +-50 scale units, the same guard as the non-arcsinh models


def combine_forecast(out, q_anchor, N, H, uses_arcsinh, qlevels):
    """Gated combination q_fc = q_anchor + gate * (q_pre - q_anchor); rewrites out['q_norm'] / out['quantiles']."""
    q_pre = out["q_pre"].float()[:, :, :H]
    if not uses_arcsinh:
        q_pre = q_pre.clamp(-50, 50)
    gf = out["gate_fc"].float()
    q_fc = q_anchor.float() + gf * (q_pre - q_anchor.float())
    if uses_arcsinh:
        q_fc = q_fc.clamp(-ARCSINH_CLIP, ARCSINH_CLIP)  # guard against degenerate normalisation
    out["q_norm"] = q_fc
    loc_t, scale_t = out["loc"], out["scale"]
    lt = loc_t if loc_t.shape[-1] == 1 else loc_t[:, -H:]
    st = scale_t if scale_t.shape[-1] == 1 else scale_t[:, -H:]
    raw = (torch.sinh(q_fc) if uses_arcsinh else q_fc) * st[:, None, :] + lt[:, None, :]
    out["quantiles"] = torch.cat([out["quantiles"][:, :, :N].float(), raw], dim=-1) if N > 0 else raw
    return out


def repaired_anchor_mc(teacher, vals, mask, L, ctx_len, N, H, now_q_raw, loc_t, scale_t, uses_arcsinh, qlevels, target_row=1, paths=(0.1, 0.5, 0.9)):
    """Anchor that propagates nowcast uncertainty: the frozen backbone is run on K contexts repaired with the
    tau-quantile nowcast paths; the K x Q output quantiles are pooled (as samples) into Q quantiles."""
    qs = []
    lv = qlevels.detach().cpu().numpy()
    for tau in paths:
        j = int(abs(lv - tau).argmin())
        qs.append(repaired_anchor(teacher, vals, mask, L, ctx_len, N, H, now_q_raw[:, j], loc_t, scale_t, uses_arcsinh, target_row))
    samp = torch.cat(qs, dim=1)  # [B, K*Q, H]
    pooled = torch.quantile(samp.float(), qlevels.to(samp.device).float(), dim=1).permute(1, 0, 2)  # [B, Q, H]
    return pooled


# ----------------------------------------------------------------------------------------------------------------
# Alternatives for propagating the nowcast uncertainty into the anchor (sensitivity analysis of the anchoring scheme)
PATHS_BY_K = {1: (0.5,), 2: (0.2, 0.8), 3: (0.1, 0.5, 0.9), 5: (0.1, 0.3, 0.5, 0.7, 0.9), 9: (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}


def repaired_anchor_paths(teacher, vals, mask, L, ctx_len, N, H, now_q_raw, loc_t, scale_t, uses_arcsinh, qlevels, target_row=1, K=3):
    """Deterministic quadrature: K comonotone quantile paths of the nowcast (K in {1,2,3,5,9}); the K x Q anchor
    quantiles are pooled into Q quantiles.  K=1 is the median-path anchor, K=3 the 0.1/0.5/0.9 scheme."""
    return repaired_anchor_mc(teacher, vals, mask, L, ctx_len, N, H, now_q_raw, loc_t, scale_t, uses_arcsinh, qlevels, target_row, paths=PATHS_BY_K[K])


def _sample_paths_copula(now_q_raw, qlevels, K, rho, gen):
    """Monte-Carlo nowcast paths from the marginal quantiles with a Gaussian copula across nowcast positions
    (AR(1) correlation rho; rho=1 comonotone, rho=0 independent).  now_q_raw: [B, Q, N] -> [K, B, N] raw paths."""
    B, Q, N = now_q_raw.shape
    dev = now_q_raw.device
    lv = qlevels.to(dev).float()
    z = torch.randn(K, B, N, device=dev, generator=gen)
    if rho > 0:
        z_ar = torch.empty_like(z); z_ar[:, :, 0] = z[:, :, 0]
        for j in range(1, N):
            z_ar[:, :, j] = rho * z_ar[:, :, j - 1] + (1 - rho ** 2) ** 0.5 * z[:, :, j]
        z = z_ar
    u = 0.5 * (1 + torch.erf(z / 2 ** 0.5))  # uniform marginals
    u = u.clamp(float(lv[0]), float(lv[-1]))  # the marginal is known through its 0.1..0.9 quantiles only
    q = now_q_raw.float().permute(0, 2, 1)  # [B, N, Q], sorted along Q
    q = torch.sort(q, dim=-1).values
    # linear interpolation of the quantile function
    idx = torch.searchsorted(lv, u.reshape(-1, N).contiguous().flatten()[:, None].contiguous(), right=True).clamp(1, Q - 1).reshape(K, B, N)
    lo, hi = lv[idx - 1], lv[idx]
    w = ((u - lo) / (hi - lo).clamp(min=1e-6)).clamp(0, 1)
    qb = q[None].expand(K, B, N, Q)
    q_lo = torch.gather(qb, 3, (idx - 1)[..., None]).squeeze(-1); q_hi = torch.gather(qb, 3, idx[..., None]).squeeze(-1)
    return q_lo + w * (q_hi - q_lo)


def repaired_anchor_mc_copula(teacher, vals, mask, L, ctx_len, N, H, now_q_raw, loc_t, scale_t, uses_arcsinh, qlevels, target_row=1, K=16, rho=0.8, seed=0):
    """Monte-Carlo integration over the nowcast distribution: K sampled paths (Gaussian copula with AR(1)
    correlation rho across positions), frozen backbone on each repaired context, K x Q output quantiles pooled."""
    gen = torch.Generator(device=now_q_raw.device); gen.manual_seed(seed)
    paths = _sample_paths_copula(now_q_raw, qlevels, K, rho, gen)  # [K, B, N]
    qs = []
    for k in range(K):
        qs.append(repaired_anchor(teacher, vals, mask, L, ctx_len, N, H, paths[k], loc_t, scale_t, uses_arcsinh, target_row))
    samp = torch.cat(qs, dim=1)
    return torch.quantile(samp.float(), qlevels.to(samp.device).float(), dim=1).permute(1, 0, 2)


def repaired_anchor_moment(teacher, vals, mask, L, ctx_len, N, H, now_q_raw, loc_t, scale_t, uses_arcsinh, qlevels, target_row=1, K=3):
    """Moment matching: the K x Q pooled anchor points are summarised per horizon by their mean and standard
    deviation and replaced by the quantiles of the matching Gaussian."""
    qs = []
    lv = qlevels.detach().cpu().numpy()
    for tau in PATHS_BY_K[K]:
        j = int(abs(lv - tau).argmin())
        qs.append(repaired_anchor(teacher, vals, mask, L, ctx_len, N, H, now_q_raw[:, j], loc_t, scale_t, uses_arcsinh, target_row))
    samp = torch.cat(qs, dim=1).float()  # [B, K*Q, H]
    mu, sd = samp.mean(1, keepdim=True), samp.std(1, keepdim=True)
    from scipy.stats import norm
    z = torch.tensor(norm.ppf(qlevels.detach().cpu().numpy()), device=samp.device, dtype=samp.dtype)  # standard normal quantiles
    return mu + sd * z[None, :, None]
