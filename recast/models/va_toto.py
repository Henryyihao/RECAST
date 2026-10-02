from __future__ import annotations
import math
import torch
import torch.nn as nn
from einops import rearrange
from toto2 import Toto2Model
from .common import (gate_supervision_loss, N_BUCKETS, LOG_AGE_SCALE, bucketize_age_diff, shared_loc_scale, AgeAxisAttention, quantile_loss, carry_forward_anchor)


class VAToto(nn.Module):
    uses_arcsinh = True
    def __init__(self, path: str, age_channel=True, age_bias=True, shared_norm=True, age_attention=True, query_row=0, dtype=torch.float32):
        super().__init__()
        self.m = Toto2Model.from_pretrained(path)
        self.m = self.m.to(dtype)
        cfg = self.m.config
        self.p = cfg.patch_size
        self.age_channel, self.age_bias, self.shared_norm, self.age_attention = age_channel, age_bias, shared_norm, age_attention
        self.query_row = query_row
        if age_channel:


            widened = False
            for name, mod in list(self.m.patch_proj.named_modules()):
                if isinstance(mod, nn.Linear) and mod.in_features == 2 * self.p:
                    fac = math.sqrt(2.0)
                    fac = math.sqrt(2.5)
                    W = torch.zeros(mod.out_features, 5 * self.p, dtype=mod.weight.dtype, device=mod.weight.device)
                    W[:, : 2 * self.p] = mod.weight.data * fac
                    mod.weight = nn.Parameter(W)
                    if mod.bias is not None:
                        mod.bias = nn.Parameter(mod.bias.data * fac)
                    mod.in_features = 5 * self.p
                    widened = True
            assert widened, "could not widen Toto patch projection"
        layers = self.m.transformer.layers
        self.time_layer_idx = [i for i in range(len(layers)) if not self.m.transformer._if_variate_layer(i)]
        self.n_heads, self.d_kv = cfg.num_heads, cfg.qk_dim
        if age_attention:
            self.age_layers = nn.ModuleDict({str(i): AgeAxisAttention(cfg.d_model, cfg.num_heads, cfg.qk_dim, 0.0) for i in self.time_layer_idx})
            for i in self.time_layer_idx:
                layers[i].register_forward_hook(self._make_hook(i))
        if age_bias:
            self.bias_emb = nn.Parameter(torch.zeros(len(layers), N_BUCKETS, cfg.num_heads))
        self._ctx = None
        self.edge = True
        self.now_weight = 1.0

        import copy
        self.now_head = copy.deepcopy(self.m.output_head)
        with torch.no_grad():
            for name, mod in self.now_head.named_modules():
                if isinstance(mod, nn.Linear) and ("linear2" in name or "skip_proj" in name):
                    nn.init.zeros_(mod.weight)
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)
        self.now_gate = nn.Parameter(torch.full((9,), -2.0))
        self.gate_head = nn.Linear(cfg.d_model, 9)
        nn.init.zeros_(self.gate_head.weight); nn.init.constant_(self.gate_head.bias, -2.0)
        self.evidence_gate = False
        self.forecast_anchor = False
        self.gate_fc_head = nn.Linear(cfg.d_model, 9)
        nn.init.zeros_(self.gate_fc_head.weight); nn.init.constant_(self.gate_fc_head.bias, -2.0)
        self.mult_head = False
        self.now_head2 = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.SiLU(), nn.Linear(cfg.d_model, 2 * 9 * self.p))
        nn.init.zeros_(self.now_head2[2].weight); nn.init.zeros_(self.now_head2[2].bias)
        self.register_buffer("qlevels", torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]), persistent=False)
        self.target_row_default = 1
        self.median_idx = 4

    def new_parameters(self):
        ps = list(self.age_layers.parameters()) if self.age_attention else []
        if self.age_bias:
            ps.append(self.bias_emb)
        return ps + list(self.now_head.parameters()) + [self.now_gate] + list(self.gate_head.parameters()) + list(self.gate_fc_head.parameters()) + list(self.now_head2.parameters())

    def input_embedding_parameters(self):
        return list(self.m.patch_proj.parameters())

    def output_head_parameters(self):
        return list(self.m.output_head.parameters())

    def _make_hook(self, i):
        def hook(module, args, output):
            if self._ctx is None:
                return output
            c = self._ctx
            bias = None
            if c["idx"] is not None:
                bias = self.bias_emb[i][c["idx"]].permute(0, 1, 4, 2, 3).reshape(c["B"] * c["T"], self.n_heads, c["S"], c["S"])
            return self.age_layers[str(i)](output, c["B"], c["S"], c["key_mask"], bias)
        return hook

    def forward(self, vals, mask, age_raw, stream_age, L, ctx_len: int, N: int = 0, target=None, target_row: int = 0, rev=None, q_anchor=None, **kw):
        out_extra_qn_u = None
        m = self.m
        if rev is None:
            rev = torch.zeros_like(vals)
        B, S, P = vals.shape
        rows = B * S
        dev = vals.device
        n_ctx = ctx_len if self.edge else ctx_len + N
        H = P - n_ctx
        p = self.p
        pad = (-n_ctx) % p

        n_fut = math.ceil(H / p) - 1 if self.edge else math.ceil(H / p)
        n_fut = max(n_fut, 0)
        x = vals.reshape(rows, P).float()
        mk = mask.reshape(rows, P).float(); ar = age_raw.reshape(rows, P).float(); rvv = rev.reshape(rows, P).float()
        def prep(z, fill=0.0):
            z = z[:, :n_ctx]
            z = torch.cat([torch.full((rows, pad), fill, device=dev), z, torch.full((rows, n_fut * p), fill, device=dev)], 1)
            return z
        xr = torch.nan_to_num(prep(x), nan=0.0); ms = prep(mk); ag = prep(ar); rg = prep(rvv)

        Xr = xr.reshape(B, S, -1); Mr = ms.reshape(B, S, -1) > 0
        if self.shared_norm:
            _, loc_r, scale_r = m.scaler(Xr[:, 1:2], Mr[:, 1:2])
            loc = loc_r.expand(B, S, -1).reshape(rows, -1); scale = scale_r.expand(B, S, -1).reshape(rows, -1)
        else:
            _, loc, scale = m.scaler(Xr, Mr)
            loc = loc.reshape(rows, -1); scale = scale.reshape(rows, -1)
        xs = torch.asinh((xr - loc) / scale)
        xs = torch.where(ms > 0, xs, 0.0)
        Lr = L.float().repeat_interleave(S)
        seq = xs.shape[1] // p
        feats = [rearrange(xs, "r (s q) -> r s q", q=p), rearrange(1 - ms, "r (s q) -> r s q", q=p)]
        if self.age_channel:
            ap_ = rearrange(ag, "r (s q) -> r s q", q=p)
            feats += [ap_ / Lr[:, None, None], torch.log1p(ap_) / LOG_AGE_SCALE, rearrange(rg, "r (s q) -> r s q", q=p)]
        emb = m.patch_proj(torch.cat(feats, -1).to(next(m.patch_proj.parameters()).dtype))
        obs = rearrange(ms, "r (s q) -> r s q", q=p).sum(-1)
        gid = torch.arange(B, device=dev).repeat_interleave(S)[:, None].expand(rows, seq).clone()
        gid[obs == 0] = -1
        gid_v = gid.reshape(B, S, seq)
        qr = self.query_row
        if n_fut > 0:
            gid_v[:, qr, seq - n_fut:] = torch.arange(B, device=dev)[:, None]
        gid_v[:, qr, seq - n_fut - 1] = torch.arange(B, device=dev)
        agp = rearrange(ag, "r (s q) -> r s q", q=p)
        ta = agp.sum(-1) / obs.clamp(min=1)
        nominal = stream_age.reshape(rows).float()
        ta = torch.where(obs > 0, ta, nominal[:, None])
        key_mask = torch.ones(B, seq, S, device=dev)
        idx = None
        if self.age_bias:
            tb = ta.reshape(B, S, seq).permute(0, 2, 1)
            idx = bucketize_age_diff(tb[:, :, :, None] - tb[:, :, None, :])
        self._ctx = dict(B=B, S=S, T=seq, key_mask=key_mask, idx=idx)
        state = m.transformer(emb.reshape(B, S, seq, -1), group_ids=gid_v)
        self._ctx = None
        hq = state[:, qr]
        i_last = seq - n_fut - 1
        if self.edge:
            qf = m.output_head(hq[:, i_last:], q=None)
            qf = rearrange(qf, "q b s p -> b q (s p)")[:, :, :H]
            gate_fc_out = None; q_pre = None
            if self.forecast_anchor:
                gf = torch.sigmoid(self.gate_fc_head(hq[:, i_last]))[:, :, None]
                gate_fc_out = gf.expand(B, 9, H); q_pre = qf.float()
                if q_anchor is not None:
                    qf = q_anchor.to(qf.dtype) + gf.to(qf.dtype) * (qf - q_anchor.to(qf.dtype))
            if N > 0:
                k = math.ceil(N / p)
                edge_tok = hq[:, i_last - k + 1: i_last + 1]
                end_ctx = pad + n_ctx
                anchor = carry_forward_anchor(xs.reshape(B, S, -1)[:, qr], ms.reshape(B, S, -1)[:, qr], end_ctx - N, N)
                if self.evidence_gate:
                    g = torch.sigmoid(self.gate_head(edge_tok))
                    g = g.permute(0, 2, 1).repeat_interleave(p, dim=-1)[:, :, -N:]
                    gate_out = g
                else:
                    g = torch.sigmoid(self.now_gate)[None, :, None].expand(B, 9, N)
                if self.mult_head:
                    Lq_ = loc.reshape(B, S, -1)[:, qr][:, end_ctx - N: end_ctx]; Sq_ = scale.reshape(B, S, -1)[:, qr][:, end_ctx - N: end_ctx]
                    h2 = self.now_head2(edge_tok).view(B, k, 2, 9, p).permute(0, 2, 3, 1, 4).reshape(B, 2, 9, k * p)[:, :, :, -N:]
                    mm = h2[:, 0].float().clamp(-3, 3); dd = h2[:, 1].float()
                    x_raw = torch.sinh(anchor.float()) * Sq_ + Lq_
                    q_raw = x_raw[:, None, :] * torch.exp(g.float() * mm) + Sq_[:, None, :] * g.float() * dd
                    qn = torch.arcsinh((q_raw - Lq_[:, None, :]) / Sq_[:, None, :])
                    q_raw_u = x_raw[:, None, :] * torch.exp(mm) + Sq_[:, None, :] * dd
                    qn_ungated = torch.arcsinh((q_raw_u - Lq_[:, None, :]) / Sq_[:, None, :])
                else:
                    qn = self.now_head(edge_tok, q=None)
                    qn = rearrange(qn, "q b s p -> b q (s p)")[:, :, -N:]
                    qn = anchor[:, None, :] + g.to(qn.dtype) * qn
                q_all = torch.cat([qn.float(), qf.float()], -1)
                if self.mult_head:
                    out_extra_qn_u = qn_ungated
            else:
                q_all = qf
        else:
            q = m.output_head(hq, q=None)
            q = rearrange(q[:, :, -(n_fut + 1):], "q b s p -> b q (s p)")
            start = p - N
            q_all = q[:, :, start: start + N + H]

        Lq = loc.reshape(B, S, -1)[:, qr]; Sq = scale.reshape(B, S, -1)[:, qr]
        n_pos = q_all.shape[-1]
        if self.edge:
            end_ctx = pad + n_ctx
            fut_loc = Lq[:, end_ctx - 1: end_ctx].expand(B, H); fut_scale = Sq[:, end_ctx - 1: end_ctx].expand(B, H)
            if N > 0:
                loc_t = torch.cat([Lq[:, end_ctx - N: end_ctx], fut_loc], 1); scale_t = torch.cat([Sq[:, end_ctx - N: end_ctx], fut_scale], 1)
            else:
                loc_t, scale_t = fut_loc, fut_scale
        else:
            loc_t = Lq[:, -1:].expand(B, n_pos); scale_t = Sq[:, -1:].expand(B, n_pos)
        out = {"q_norm": q_all, "loc": loc_t, "scale": scale_t}
        if self.edge and N > 0 and self.evidence_gate:
            out["gate"] = gate_out
        if self.edge and self.forecast_anchor:
            out["gate_fc"] = gate_fc_out; out["q_pre"] = q_pre
        if target is not None:
            yt = target.float(); ymask = torch.isfinite(yt).float()
            yn = torch.asinh((torch.nan_to_num(yt, nan=0.0) - loc_t) / scale_t)
            if self.edge and N > 0:
                ln = quantile_loss(q_all[:, :, :N], yn[:, :N], ymask[:, :N], self.qlevels)
                lf = quantile_loss(q_all[:, :, N:], yn[:, N:], ymask[:, N:], self.qlevels)
                out["loss"] = lf + self.now_weight * ln; out["loss_now"], out["loss_fc"] = ln.detach(), lf.detach(); out["lf"], out["ln"] = lf, ln
                if out_extra_qn_u is not None:
                    out["ln_u"] = quantile_loss(out_extra_qn_u.float(), yn[:, :N], ymask[:, :N], self.qlevels)
                    if out.get("gate") is not None:
                        out["lg"] = gate_supervision_loss(out["gate"], out_extra_qn_u.float(), anchor.float(), yn[:, :N], ymask[:, :N], self.qlevels)
            else:
                out["loss"] = quantile_loss(q_all, yn, ymask, self.qlevels)
        out["quantiles"] = torch.sinh(q_all.float().clamp(-4.6052, 4.6052)) * scale_t[:, None, :] + loc_t[:, None, :]
        return out
