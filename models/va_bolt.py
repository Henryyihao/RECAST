"""VERA on Chronos-Bolt (T5 encoder-decoder, univariate): age-view streams + zero-initialised age-axis attention
inserted after every encoder block (forward hooks) + widened input embedding + retargeted joint output window.

Bolt reads context positions <= t (all C positions of the window) and its single decoder output patch of 64 steps
is re-targeted to the settled values of positions (t-N, t-N+64]; the loss covers the first N+H of them.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from chronos.chronos_bolt import ChronosBoltModelForForecasting, ResidualBlock
from .common import (gate_supervision_loss, N_BUCKETS, LOG_AGE_SCALE, bucketize_age_diff, shared_loc_scale, patch_1d, AgeAxisAttention,
                     token_ages, quantile_loss, carry_forward_anchor)


class VABolt(nn.Module):
    uses_arcsinh = False
    def __init__(self, path: str, age_channel: bool = True, age_bias: bool = True, shared_norm: bool = True,
                 age_attention: bool = True, query_row: int = 1, dtype=torch.float32):
        super().__init__()
        self.m = ChronosBoltModelForForecasting.from_pretrained(path, torch_dtype=dtype)
        cfg = self.m.chronos_config
        self.p = cfg.input_patch_size
        self.pred_len = cfg.prediction_length
        self.age_channel, self.age_bias, self.shared_norm, self.age_attention = age_channel, age_bias, shared_norm, age_attention
        self.query_row = query_row
        c = self.m.config
        if age_channel:
            old = self.m.input_patch_embedding
            new = ResidualBlock(in_dim=self.p * 5, h_dim=c.d_ff, out_dim=c.d_model, act_fn_name=c.dense_act_fn, dropout_p=c.dropout_rate)
            with torch.no_grad():
                for name in ("hidden_layer", "residual_layer"):
                    ln, lo = getattr(new, name), getattr(old, name)
                    ln.weight.zero_(); ln.weight[:, : self.p * 2] = lo.weight; ln.bias.copy_(lo.bias)
                new.output_layer.load_state_dict(old.output_layer.state_dict())
            self.m.input_patch_embedding = new.to(old.output_layer.weight.device, dtype=old.output_layer.weight.dtype)
        n_layers = len(self.m.encoder.block)
        self.n_layers, self.n_heads, self.d_kv = n_layers, c.num_heads, c.d_kv
        if age_attention:
            self.age_layers = nn.ModuleList([AgeAxisAttention(c.d_model, c.num_heads, c.d_kv, c.dropout_rate) for _ in range(n_layers)])
            for i, blk in enumerate(self.m.encoder.block):
                blk.register_forward_hook(self._make_hook(i))
        if age_bias:
            self.bias_emb = nn.Parameter(torch.zeros(n_layers, N_BUCKETS, c.num_heads))
        self._ctx = None
        self.edge = True
        self.now_weight = 1.0
        # nowcast head on encoder edge tokens: d_model -> 9 x patch, initialised from the first `patch` steps of the
        # pretrained decoder head
        nq = len(cfg.quantiles)
        self.now_head = ResidualBlock(in_dim=c.d_model, h_dim=c.d_ff, out_dim=nq * self.p, act_fn_name=c.dense_act_fn, dropout_p=c.dropout_rate)
        with torch.no_grad():
            oh = self.m.output_patch_embedding
            self.now_head.hidden_layer.load_state_dict(oh.hidden_layer.state_dict())
            W = oh.output_layer.weight.view(nq, self.pred_len, -1)[:, : self.p].reshape(nq * self.p, -1)
            bb = oh.output_layer.bias.view(nq, self.pred_len)[:, : self.p].reshape(-1)
            self.now_head.output_layer.weight.copy_(W); self.now_head.output_layer.bias.copy_(bb)
            Wr = oh.residual_layer.weight.view(nq, self.pred_len, -1)[:, : self.p].reshape(nq * self.p, -1)
            br = oh.residual_layer.bias.view(nq, self.pred_len)[:, : self.p].reshape(-1)
            self.now_head.residual_layer.weight.copy_(Wr); self.now_head.residual_layer.bias.copy_(br)
        # zero-initialised correction pathway
        with torch.no_grad():
            nn.init.zeros_(self.now_head.output_layer.weight); nn.init.zeros_(self.now_head.output_layer.bias)
            nn.init.zeros_(self.now_head.residual_layer.weight); nn.init.zeros_(self.now_head.residual_layer.bias)
        self.now_head = self.now_head.to(self.m.output_patch_embedding.output_layer.weight.dtype)
        self.now_gate = nn.Parameter(torch.full((nq,), -2.0))  # global (input-independent) gate, sigmoid-parameterised
        self.gate_head = nn.Linear(c.d_model, nq)
        nn.init.zeros_(self.gate_head.weight); nn.init.constant_(self.gate_head.bias, -2.0)
        self.evidence_gate = False
        self.forecast_anchor = False
        self.gate_fc_head = nn.Linear(c.d_model, nq)
        nn.init.zeros_(self.gate_fc_head.weight); nn.init.constant_(self.gate_fc_head.bias, -2.0)
        self.mult_head = False
        self.now_head2 = ResidualBlock(in_dim=c.d_model, h_dim=c.d_ff, out_dim=2 * nq * self.p, act_fn_name=c.dense_act_fn, dropout_p=c.dropout_rate)
        with torch.no_grad():
            self.now_head2.hidden_layer.load_state_dict(self.m.output_patch_embedding.hidden_layer.state_dict())
            nn.init.zeros_(self.now_head2.output_layer.weight); nn.init.zeros_(self.now_head2.output_layer.bias)
            nn.init.zeros_(self.now_head2.residual_layer.weight); nn.init.zeros_(self.now_head2.residual_layer.bias)
        self.now_head2 = self.now_head2.to(self.m.output_patch_embedding.output_layer.weight.dtype)
        self.register_buffer("qlevels", torch.tensor(cfg.quantiles), persistent=False)
        self.median_idx = 4

    def new_parameters(self):
        ps = list(self.age_layers.parameters()) if self.age_attention else []
        if self.age_bias:
            ps.append(self.bias_emb)
        return ps + list(self.now_head.parameters()) + [self.now_gate] + list(self.gate_head.parameters()) + list(self.gate_fc_head.parameters()) + list(self.now_head2.parameters())

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
        """vals/mask/age_raw [B, S, P]; Bolt reads positions [0, ctx_len + N) (everything <= t)."""
        m = self.m
        if rev is None:
            rev = torch.zeros_like(vals)
        B, S, P = vals.shape
        rows = B * S
        dev = vals.device
        n_ctx = ctx_len if self.edge else ctx_len + N
        x = vals.reshape(rows, P).float()[:, :n_ctx]
        mk = mask.reshape(rows, P).float()[:, :n_ctx]
        ar = age_raw.reshape(rows, P).float()[:, :n_ctx]
        rv = rev.reshape(rows, P).float()[:, :n_ctx]
        Lr = L.float().repeat_interleave(S)
        loc, scale = shared_loc_scale(vals.reshape(rows, P).float(), B, S, n_ctx, ref_row=1, shared=self.shared_norm)
        xn = (x - loc) / scale
        xn = torch.where(mk > 0, xn, 0.0)
        pv = patch_1d(xn, self.p); pm = torch.nan_to_num(patch_1d(mk, self.p), nan=0.0)
        pv = torch.nan_to_num(torch.where(pm > 0, pv, 0.0), nan=0.0)
        feats = [pv, pm]
        if self.age_channel:
            pa = torch.nan_to_num(patch_1d(ar, self.p), nan=0.0)
            pr = torch.nan_to_num(patch_1d(rv, self.p), nan=0.0)
            feats += [pa / Lr[:, None, None], torch.log1p(pa) / LOG_AGE_SCALE, pr]
        emb = m.input_patch_embedding(torch.cat(feats, -1).to(m.dtype))
        attn = (pm.sum(-1) > 0).to(m.dtype)
        reg = m.shared(torch.full((rows, 1), m.config.reg_token_id, device=dev))
        emb = torch.cat([emb, reg], 1)
        attn = torch.cat([attn, torch.ones(rows, 1, device=dev, dtype=m.dtype)], 1)
        T = emb.shape[1]
        nominal = stream_age.reshape(rows).float()
        pa = torch.nan_to_num(patch_1d(ar, self.p), nan=0.0)
        ta = torch.cat([token_ages(pa, pm, nominal), torch.zeros(rows, 1, device=dev)], 1)  # [rows, T]
        key_mask = attn.reshape(B, S, T).permute(0, 2, 1)
        idx = None
        if self.age_bias:
            tb = ta.reshape(B, S, T).permute(0, 2, 1)
            idx = bucketize_age_diff(tb[:, :, :, None] - tb[:, :, None, :])
        self._ctx = dict(B=B, S=S, T=T, key_mask=key_mask, idx=idx)
        enc = m.encoder(attention_mask=attn, inputs_embeds=emb)[0]
        self._ctx = None
        # decode from the query row only
        hq = enc.reshape(B, S, T, -1)[:, self.query_row]
        aq = attn.reshape(B, S, T)[:, self.query_row]
        dec = m.decode(hq, aq, hq)
        q = m.output_patch_embedding(dec).view(B, m.num_quantiles, self.pred_len)
        gate_fc_out = None; q_pre = None
        if self.forecast_anchor:
            Hq_ = P - ctx_len if self.edge else self.pred_len
            gf = torch.sigmoid(self.gate_fc_head(dec[:, 0]))[:, :, None]  # [B, Q, 1]
            gate_fc_out = gf.expand(B, m.num_quantiles, Hq_); q_pre = q.float()[:, :, :Hq_]
            if q_anchor is not None:
                q = torch.cat([q_anchor.to(q.dtype) + gf.to(q.dtype) * (q[:, :, :Hq_] - q_anchor.to(q.dtype)), q[:, :, Hq_:]], dim=-1)
        loc_t = loc.reshape(B, S, 1)[:, self.query_row]; scale_t = scale.reshape(B, S, 1)[:, self.query_row]
        out = {"q_norm": q, "loc": loc_t, "scale": scale_t}
        if gate_fc_out is not None:
            out["gate_fc"] = gate_fc_out; out["q_pre"] = q_pre
        H = P - ctx_len if self.edge else None
        if self.edge and N > 0:
            k = math.ceil(N / self.p)
            n_ctx_patches = T - 1
            edge_tok = hq[:, n_ctx_patches - k: n_ctx_patches]
            anchor = carry_forward_anchor(torch.nan_to_num(xn.reshape(B, S, n_ctx)[:, self.query_row], nan=0.0), mk.reshape(B, S, n_ctx)[:, self.query_row], n_ctx - N, N)
            if self.evidence_gate:
                g = torch.sigmoid(self.gate_head(edge_tok))  # [B, k, Q]
                g = g.permute(0, 2, 1).repeat_interleave(self.p, dim=-1)[:, :, -N:]
                out["gate"] = g
            else:
                g = torch.sigmoid(self.now_gate)[None, :, None].expand(B, m.num_quantiles, N)
            if self.mult_head:
                h2 = self.now_head2(edge_tok).view(B, k, 2, m.num_quantiles, self.p).permute(0, 2, 3, 1, 4).reshape(B, 2, m.num_quantiles, k * self.p)[:, :, :, -N:]
                mm = h2[:, 0].float().clamp(-3, 3); dd = h2[:, 1].float()
                x_raw = anchor.float() * scale_t + loc_t
                q_raw = x_raw[:, None, :] * torch.exp(g.float() * mm) + scale_t[:, None, :] * g.float() * dd
                qn = ((q_raw - loc_t[:, None, :]) / scale_t[:, None, :]).clamp(-50, 50)
                q_raw_u = x_raw[:, None, :] * torch.exp(mm) + scale_t[:, None, :] * dd  # un-gated correction (auxiliary)
                qn_ungated = ((q_raw_u - loc_t[:, None, :]) / scale_t[:, None, :]).clamp(-50, 50)
            else:
                qn = self.now_head(edge_tok)  # [B, k, 9*p]
                qn = qn.view(B, k, m.num_quantiles, self.p).permute(0, 2, 1, 3).reshape(B, m.num_quantiles, k * self.p)[:, :, -N:]
                qn = anchor[:, None, :] + g.to(qn.dtype) * qn
            q_all = torch.cat([qn.float(), q[:, :, :H].float()], dim=-1)
            if self.mult_head:
                out_extra_qn_u = qn_ungated
        elif self.edge:
            q_all = q[:, :, :H]
        else:
            q_all = q
        if target is not None:
            n = target.shape[1]
            yt = target.float(); ymask = torch.isfinite(yt).float()
            yn = (torch.nan_to_num(yt, nan=0.0) - loc_t) / scale_t
            if self.edge and N > 0:
                ln = quantile_loss(q_all[:, :, :N], yn[:, :N], ymask[:, :N], self.qlevels)
                lf = quantile_loss(q_all[:, :, N:n], yn[:, N:], ymask[:, N:], self.qlevels)
                out["loss"] = lf + self.now_weight * ln; out["loss_now"], out["loss_fc"] = ln.detach(), lf.detach(); out["lf"], out["ln"] = lf, ln
                if out_extra_qn_u is not None:
                    out["ln_u"] = quantile_loss(out_extra_qn_u.float(), yn[:, :N], ymask[:, :N], self.qlevels)  # un-gated auxiliary
                    if out.get("gate") is not None:
                        out["lg"] = gate_supervision_loss(out["gate"], out_extra_qn_u.float(), anchor.float(), yn[:, :N], ymask[:, :N], self.qlevels)
            else:
                out["loss"] = quantile_loss(q_all[:, :, :n], yn, ymask, self.qlevels)
        out["quantiles"] = q_all.float() * scale_t[:, None, :] + loc_t[:, None, :]  # [B, 9, N+H]
        return out
