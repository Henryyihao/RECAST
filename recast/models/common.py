from __future__ import annotations
import math
import torch
import torch.nn as nn
from einops import rearrange

_EDGES = torch.tensor([0.5, 1.5, 2.5, 3.5, 5.5, 8.5, 12.5, 18.5, 27.5, 40.5, 60.5, 90.5])
N_BUCKETS = 2 * len(_EDGES) + 1
LOG_AGE_SCALE = math.log1p(100.0)


def bucketize_age_diff(d: torch.Tensor) -> torch.Tensor:
    mag = torch.bucketize(d.abs().contiguous(), _EDGES.to(d.device))
    return (len(_EDGES) + torch.sign(d).long() * mag).long()


def shared_loc_scale(x, B, S, ctx_len, ref_row=1, shared=True):

    ctx = x[:, :ctx_len]
    if shared and S > 1:
        ref = ctx.reshape(B, S, ctx_len)[:, ref_row]
        loc = torch.nan_to_num(torch.nanmean(ref, dim=-1, keepdim=True), nan=0.0)
        scale = torch.nan_to_num((ref - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
        scale = torch.where(scale <= 0, torch.ones_like(scale), scale)
        return loc.repeat_interleave(S, 0), scale.repeat_interleave(S, 0)
    loc = torch.nan_to_num(torch.nanmean(ctx, dim=-1, keepdim=True), nan=0.0)
    scale = torch.nan_to_num((ctx - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
    scale = torch.where(scale <= 0, torch.ones_like(scale), scale)
    return loc, scale


def patch_1d(x, p):

    length = x.shape[-1]
    if length % p != 0:
        pad = torch.full((*x.shape[:-1], p - length % p), float("nan"), dtype=x.dtype, device=x.device)
        x = torch.cat([pad, x], dim=-1)
    return x.unfold(-1, p, p)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + self.eps)).to(x.dtype) * self.weight


class AgeAxisAttention(nn.Module):


    def __init__(self, d_model, n_heads, d_kv, dropout=0.0, scale=None):
        super().__init__()
        self.n_heads, self.d_kv = n_heads, d_kv
        inner = n_heads * d_kv
        self.norm = RMSNorm(d_model)
        self.q = nn.Linear(d_model, inner, bias=False); self.k = nn.Linear(d_model, inner, bias=False)
        self.v = nn.Linear(d_model, inner, bias=False); self.o = nn.Linear(inner, d_model, bias=False)
        nn.init.zeros_(self.o.weight)
        self.drop = nn.Dropout(dropout)
        self.scale = scale if scale is not None else d_kv ** -0.5

    def forward(self, hs, B, S, key_mask, bias=None):

        rows, T, d = hs.shape
        x = hs.reshape(B, S, T, d).permute(0, 2, 1, 3).reshape(B * T, S, d)
        n = self.norm(x)
        sh = lambda z: rearrange(z, "b s (h k) -> b h s k", h=self.n_heads, k=self.d_kv)
        q, k, v = sh(self.q(n)), sh(self.k(n)), sh(self.v(n))
        m = key_mask.reshape(B * T, 1, 1, S).to(q.dtype)
        m = (1.0 - m) * torch.finfo(q.dtype).min
        if bias is not None:
            m = m + bias.to(q.dtype)
        out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=m, dropout_p=self.drop.p if self.training else 0.0, scale=self.scale)
        out = self.o(rearrange(out, "b h s k -> b s (h k)"))
        x = x + self.drop(out)
        return x.reshape(B, T, S, d).permute(0, 2, 1, 3).reshape(rows, T, d)


def token_ages(age_raw_patched, mask_patched, nominal):

    cnt = mask_patched.sum(-1)
    ta = age_raw_patched.sum(-1) / cnt.clamp(min=1)
    return torch.where(cnt > 0, ta, nominal[:, None])


def quantile_loss(q_norm, y_norm, ymask, levels):

    ql = levels.to(q_norm.dtype)[None, :, None]
    ymask = ymask * (y_norm.abs() < 30.0).float()
    diff = y_norm[:, None, :] - q_norm
    pl = 2 * torch.abs(diff * ((diff <= 0).float() - ql)) * ymask[:, None, :]
    return (pl.sum(-1) / ymask.sum(-1).clamp(min=1)[:, None]).sum(1).mean()


def fix_timemoe_rotary(model):

    import torch
    for mod in model.modules():
        if type(mod).__name__ == "TimeMoeRotaryEmbedding":
            dev = next(model.parameters()).device
            inv = 1.0 / (mod.base ** (torch.arange(0, mod.dim, 2, dtype=torch.int64).float().to(dev) / mod.dim))
            mod.inv_freq = inv
            mod._set_cos_sin_cache(seq_len=mod.max_position_embeddings, device=dev, dtype=torch.float32)
    return model


def carry_forward_anchor(xn_row, mask_row, start, N):


    B, P = xn_row.shape
    idx = torch.arange(P, device=xn_row.device)[None, :].expand(B, P)
    last_obs = torch.where(mask_row > 0, idx, torch.full_like(idx, -1))
    last_obs = torch.cummax(last_obs, dim=1).values
    sel = last_obs[:, start: start + N]
    ok = sel >= 0
    anchor = torch.gather(xn_row, 1, sel.clamp(min=0))
    return torch.where(ok, anchor, torch.zeros_like(anchor))


def gate_supervision_loss(gate, q_u, anchor, y_norm, ymask, levels):


    ql = levels.to(q_u.dtype)[None, :, None]
    ymask = ymask * (y_norm.abs() < 30.0).float()
    du = y_norm[:, None, :] - q_u; da = (y_norm - anchor)[:, None, :].expand_as(du)
    pl_u = 2 * torch.abs(du * ((du <= 0).float() - ql)); pl_a = 2 * torch.abs(da * ((da <= 0).float() - ql))
    label = (pl_u < pl_a).float()
    g = gate.float().clamp(1e-4, 1 - 1e-4)
    bce = -(label * torch.log(g) + (1 - label) * torch.log(1 - g)) * ymask[:, None, :]
    return bce.sum() / (ymask.sum() * gate.shape[1]).clamp(min=1.0)
