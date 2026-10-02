from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from .common import (gate_supervision_loss, N_BUCKETS, LOG_AGE_SCALE, bucketize_age_diff, shared_loc_scale, AgeAxisAttention, quantile_loss, fix_timemoe_rotary, carry_forward_anchor)


class VATimeMoE(nn.Module):
    uses_arcsinh = False
    def __init__(self, path: str, age_channel=True, age_bias=True, shared_norm=True, age_attention=True, query_row=1, dtype=torch.float32):
        super().__init__()
        self.m = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=dtype, attn_implementation="eager")
        fix_timemoe_rotary(self.m)
        cfg = self.m.config
        self.hidden = cfg.hidden_size
        self.age_channel, self.age_bias, self.shared_norm, self.age_attention = age_channel, age_bias, shared_norm, age_attention
        self.query_row = query_row
        emb = self.m.model.embed_layer
        n_in = 5 if age_channel else 2
        for name in ("emb_layer", "gate_layer"):
            old = getattr(emb, name)
            new = nn.Linear(n_in, old.out_features, bias=False)
            with torch.no_grad():
                new.weight.zero_(); new.weight[:, :1] = old.weight
            setattr(emb, name, new.to(old.weight.dtype))
        emb.input_size = n_in
        self.n_in = n_in
        layers = self.m.model.layers
        self.n_layers = len(layers)
        self.n_heads = cfg.num_attention_heads
        self.d_kv = cfg.hidden_size // cfg.num_attention_heads
        if age_attention:
            self.age_layers = nn.ModuleList([AgeAxisAttention(cfg.hidden_size, self.n_heads, self.d_kv, 0.0) for _ in range(self.n_layers)])
            for i, l in enumerate(layers):
                l.register_forward_hook(self._make_hook(i))
        if age_bias:
            self.bias_emb = nn.Parameter(torch.zeros(self.n_layers, N_BUCKETS, self.n_heads))

        h64 = self.m.lm_heads[self.m.horizon_length_map[64]].out_layer
        self.pred_len = 64
        self.qhead = nn.Linear(cfg.hidden_size, 9 * 64, bias=False)
        with torch.no_grad():
            self.qhead.weight.copy_(h64.weight.repeat(9, 1))
        self.qhead = self.qhead.to(h64.weight.dtype)

        h1 = self.m.lm_heads[self.m.horizon_length_map[1]].out_layer
        self.now_head = nn.Linear(cfg.hidden_size, 9, bias=True)
        nn.init.zeros_(self.now_head.weight); nn.init.zeros_(self.now_head.bias)
        self.now_head = self.now_head.to(h1.weight.dtype)
        self.now_gate = nn.Parameter(torch.full((9,), -2.0))
        self.gate_head = nn.Linear(cfg.hidden_size, 9)
        nn.init.zeros_(self.gate_head.weight); nn.init.constant_(self.gate_head.bias, -2.0)
        self.evidence_gate = False
        self.forecast_anchor = False
        self.gate_fc_head = nn.Linear(cfg.hidden_size, 9)
        nn.init.zeros_(self.gate_fc_head.weight); nn.init.constant_(self.gate_fc_head.bias, -2.0)
        self.mult_head = False
        self.now_head2 = nn.Sequential(nn.Linear(cfg.hidden_size, cfg.hidden_size), nn.SiLU(), nn.Linear(cfg.hidden_size, 2 * 9))
        nn.init.zeros_(self.now_head2[2].weight); nn.init.zeros_(self.now_head2[2].bias)
        self._ctx = None
        self.edge = True
        self.now_weight = 1.0
        self.register_buffer("qlevels", torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]), persistent=False)
        self.target_row_default = 1
        self.median_idx = 4

    def new_parameters(self):
        ps = list(self.age_layers.parameters()) if self.age_attention else []
        if self.age_bias:
            ps.append(self.bias_emb)
        return ps + list(self.qhead.parameters()) + list(self.now_head.parameters()) + [self.now_gate] + list(self.gate_head.parameters()) + list(self.gate_fc_head.parameters()) + list(self.now_head2.parameters())

    def input_embedding_parameters(self):
        return list(self.m.model.embed_layer.parameters())

    def output_head_parameters(self):
        return list(self.qhead.parameters())

    def _make_hook(self, i):
        def hook(module, args, output):
            if self._ctx is None:
                return output
            c = self._ctx
            bias = None
            if c["idx"] is not None:
                bias = self.bias_emb[i][c["idx"]].permute(0, 1, 4, 2, 3).reshape(c["B"] * c["T"], self.n_heads, c["S"], c["S"])
            hs = self.age_layers[i](output[0], c["B"], c["S"], c["key_mask"], bias)
            return (hs,) + tuple(output[1:])
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
        x = vals.reshape(rows, P).float()
        loc, scale = shared_loc_scale(x, B, S, n_ctx, ref_row=1, shared=self.shared_norm)
        xn = ((x - loc) / scale)[:, :n_ctx]
        mk = mask.reshape(rows, P).float()[:, :n_ctx]; ar = age_raw.reshape(rows, P).float()[:, :n_ctx]; rv = rev.reshape(rows, P).float()[:, :n_ctx]
        xn = torch.nan_to_num(torch.where(mk > 0, xn, 0.0), nan=0.0)
        Lr = L.float().repeat_interleave(S)
        feats = [xn, mk]
        if self.age_channel:
            feats += [ar / Lr[:, None], torch.log1p(ar) / LOG_AGE_SCALE, rv]
        inp = torch.stack(feats, -1).to(next(m.parameters()).dtype)
        ta = ar.clone()
        nominal = stream_age.reshape(rows).float()
        ta = torch.where(mk > 0, ta, nominal[:, None])
        key_mask = torch.ones(B, n_ctx, S, device=dev)
        idx = None
        if self.age_bias:
            tb = ta.reshape(B, S, n_ctx).permute(0, 2, 1)
            idx = bucketize_age_diff(tb[:, :, :, None] - tb[:, :, None, :])
        self._ctx = dict(B=B, S=S, T=n_ctx, key_mask=key_mask, idx=idx)
        out_m = m.model(input_ids=inp, attention_mask=None, use_cache=False, return_dict=True)

        hs = out_m.last_hidden_state
        Hq = hs.reshape(B, S, n_ctx, -1)[:, self.query_row]
        q = self.qhead(Hq[:, -1]).view(B, 9, self.pred_len)
        gate_fc_out = None; q_pre = None
        if self.forecast_anchor:
            Hq_ = P - ctx_len if self.edge else self.pred_len
            gf = torch.sigmoid(self.gate_fc_head(Hq[:, -1]))[:, :, None]
            gate_fc_out = gf.expand(B, 9, Hq_); q_pre = q.float()[:, :, :Hq_]
            if q_anchor is not None:
                q = torch.cat([q_anchor.to(q.dtype) + gf.to(q.dtype) * (q[:, :, :Hq_] - q_anchor.to(q.dtype)), q[:, :, Hq_:]], dim=-1)
        loc_t = loc.reshape(B, S, 1)[:, self.query_row]; scale_t = scale.reshape(B, S, 1)[:, self.query_row]
        H = P - ctx_len if self.edge else None
        if self.edge and N > 0:
            anchor = carry_forward_anchor(xn.reshape(B, S, n_ctx)[:, self.query_row], mk.reshape(B, S, n_ctx)[:, self.query_row], n_ctx - N, N)
            if self.evidence_gate:
                g = torch.sigmoid(self.gate_head(Hq[:, -N:])).permute(0, 2, 1)
                gate_out = g
            else:
                g = torch.sigmoid(self.now_gate)[None, :, None].expand(B, 9, N)
            if self.mult_head:
                h2 = self.now_head2(Hq[:, -N:]).view(B, N, 2, 9).permute(0, 2, 3, 1)
                mm = h2[:, 0].float().clamp(-3, 3); dd = h2[:, 1].float()
                x_raw = anchor.float() * scale_t + loc_t
                q_raw = x_raw[:, None, :] * torch.exp(g.float() * mm) + scale_t[:, None, :] * g.float() * dd
                qn = ((q_raw - loc_t[:, None, :]) / scale_t[:, None, :]).clamp(-50, 50)
                q_raw_u = x_raw[:, None, :] * torch.exp(mm) + scale_t[:, None, :] * dd
                qn_ungated = ((q_raw_u - loc_t[:, None, :]) / scale_t[:, None, :]).clamp(-50, 50)
            else:
                qn = self.now_head(Hq[:, -N:]).permute(0, 2, 1)
                qn = anchor[:, None, :] + g.to(qn.dtype) * qn
            q_all = torch.cat([qn.float(), q[:, :, :H].float()], -1)
            if self.mult_head:
                out_extra_qn_u = qn_ungated
        elif self.edge:
            q_all = q[:, :, :H]
        else:
            q_all = q
        out = {"q_norm": q_all, "loc": loc_t, "scale": scale_t}
        if self.edge and N > 0 and self.evidence_gate:
            out["gate"] = gate_out
        if gate_fc_out is not None:
            out["gate_fc"] = gate_fc_out; out["q_pre"] = q_pre
        if target is not None:
            n = target.shape[1]
            yt = target.float(); ymask = torch.isfinite(yt).float()
            yn = (torch.nan_to_num(yt, nan=0.0) - loc_t) / scale_t
            if self.edge and N > 0:
                ln = quantile_loss(q_all[:, :, :N], yn[:, :N], ymask[:, :N], self.qlevels)
                lf = quantile_loss(q_all[:, :, N:n], yn[:, N:], ymask[:, N:], self.qlevels)
                out["loss"] = lf + self.now_weight * ln; out["loss_now"], out["loss_fc"] = ln.detach(), lf.detach(); out["lf"], out["ln"] = lf, ln
                if out_extra_qn_u is not None:
                    out["ln_u"] = quantile_loss(out_extra_qn_u.float(), yn[:, :N], ymask[:, :N], self.qlevels)
                    if out.get("gate") is not None:
                        out["lg"] = gate_supervision_loss(out["gate"], out_extra_qn_u.float(), anchor.float(), yn[:, :N], ymask[:, :N], self.qlevels)
            else:
                out["loss"] = quantile_loss(q_all[:, :, :n], yn, ymask, self.qlevels)
        out["quantiles"] = q_all.float() * scale_t[:, None, :] + loc_t[:, None, :]
        return out
