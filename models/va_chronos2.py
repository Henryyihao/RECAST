"""VERA on Chronos-2: vintage-aware encoder with age-axis attention.

Modifications relative to the pretrained Chronos-2 encoder (all zero-initialised so that the
model is functionally identical to Chronos-2 at initialisation):
  * input patch embedding widened from [time, value, mask] to [time, value, mask, age_n, age_log]
  * relative-age bias added to the (pretrained) group self-attention, turning the unordered
    cross-variate attention into an ordered attention along the report-age axis; the attention is
    evaluated per sample over its S streams (equivalent to Chronos-2's group-masked attention)
  * shared (group-level) instance normalisation so that the level gap between vintages is visible
  * joint output window (t-N, t+H]: nowcasts and forecasts are one quantile output of the settled stream
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from chronos.chronos2.model import Chronos2Model
from chronos.chronos2.layers import ResidualBlock
from .common import carry_forward_anchor, gate_supervision_loss

# signed-log buckets of the raw age difference (in grid steps)
_EDGES = torch.tensor([0.5, 1.5, 2.5, 3.5, 5.5, 8.5, 12.5, 18.5, 27.5, 40.5, 60.5, 90.5])
N_BUCKETS = 2 * len(_EDGES) + 1
LOG_AGE_SCALE = math.log1p(100.0)


def bucketize_age_diff(d: torch.Tensor) -> torch.Tensor:
    mag = torch.bucketize(d.abs().contiguous(), _EDGES.to(d.device))  # 0..12
    return (len(_EDGES) + torch.sign(d).long() * mag).long()


class VAChronos2(nn.Module):
    uses_arcsinh = True
    def __init__(self, path: str, age_channel: bool = True, age_bias: bool = True, shared_norm: bool = True,
                 dtype=torch.float32, now_weight: float = 1.0):
        super().__init__()
        self.now_weight = now_weight
        self.edge = True
        self.m = Chronos2Model.from_pretrained(path, torch_dtype=dtype)
        self.m.config._attn_implementation = "sdpa"
        cfg = self.m.chronos_config
        self.p = cfg.input_patch_size
        self.age_channel, self.age_bias, self.shared_norm = age_channel, age_bias, shared_norm
        c = self.m.config
        if age_channel:
            old = self.m.input_patch_embedding
            new = ResidualBlock(in_dim=self.p * 6, h_dim=c.d_ff, out_dim=c.d_model, act_fn_name=c.dense_act_fn,
                                dropout_p=c.dropout_rate)
            with torch.no_grad():
                for name in ("hidden_layer", "residual_layer"):
                    ln, lo = getattr(new, name), getattr(old, name)
                    ln.weight.zero_(); ln.weight[:, : self.p * 3] = lo.weight; ln.bias.copy_(lo.bias)
                new.output_layer.load_state_dict(old.output_layer.state_dict())
            self.m.input_patch_embedding = new.to(self.m.device, dtype=old.output_layer.weight.dtype)
        if age_bias:
            self.bias_emb = nn.Parameter(torch.zeros(c.num_layers, N_BUCKETS, c.num_heads))
        self.n_layers, self.n_heads, self.d_kv = c.num_layers, c.num_heads, c.d_kv
        self.now_gate = nn.Parameter(torch.full((len(cfg.quantiles),), -2.0))  # global (input-independent) gate, sigmoid-parameterised  # anchored nowcast: q_now = anchor + gate * head
        # evidence gate: per-token sigmoid gate predicted from the edge token state (initialised nearly closed)
        self.gate_head = nn.Linear(c.d_model, len(cfg.quantiles))
        nn.init.zeros_(self.gate_head.weight); nn.init.constant_(self.gate_head.bias, -2.0)
        self.evidence_gate = False
        self.forecast_anchor = False
        self.gate_fc_head = nn.Linear(c.d_model, len(cfg.quantiles))
        nn.init.zeros_(self.gate_fc_head.weight); nn.init.constant_(self.gate_fc_head.bias, -2.0)
        # dedicated nowcast-correction head: hidden layer copied from the pretrained output head, output and residual
        # layers zero-initialised so that the correction pathway is exactly zero at initialisation
        oh = self.m.output_patch_embedding
        self.now_head = ResidualBlock(in_dim=c.d_model, h_dim=c.d_ff, out_dim=len(cfg.quantiles) * self.p, act_fn_name=c.dense_act_fn, dropout_p=c.dropout_rate)
        with torch.no_grad():
            self.now_head.hidden_layer.load_state_dict(oh.hidden_layer.state_dict())
            nn.init.zeros_(self.now_head.output_layer.weight); nn.init.zeros_(self.now_head.output_layer.bias)
            nn.init.zeros_(self.now_head.residual_layer.weight); nn.init.zeros_(self.now_head.residual_layer.bias)
        self.now_head = self.now_head.to(oh.output_layer.weight.dtype)
        self.separate_now_head = False
        self.mult_head = False
        # multiplicative-additive head: outputs (m, delta) per quantile and position: q_raw = x_raw * exp(g m) + scale * g * delta
        self.now_head2 = ResidualBlock(in_dim=c.d_model, h_dim=c.d_ff, out_dim=2 * len(cfg.quantiles) * self.p, act_fn_name=c.dense_act_fn, dropout_p=c.dropout_rate)
        with torch.no_grad():
            self.now_head2.hidden_layer.load_state_dict(oh.hidden_layer.state_dict())
            nn.init.zeros_(self.now_head2.output_layer.weight); nn.init.zeros_(self.now_head2.output_layer.bias)
            nn.init.zeros_(self.now_head2.residual_layer.weight); nn.init.zeros_(self.now_head2.residual_layer.bias)
        self.now_head2 = self.now_head2.to(oh.output_layer.weight.dtype)
        self.register_buffer("qlevels", torch.tensor(cfg.quantiles), persistent=False)
        self.median_idx = int(np.argmin(np.abs(np.array(cfg.quantiles) - 0.5)))

    # ----------------------------------------------------------------- helpers
    def new_parameters(self):
        return ([self.bias_emb] if self.age_bias else []) + [self.now_gate] + list(self.gate_head.parameters()) + list(self.now_head.parameters()) + list(self.gate_fc_head.parameters()) + list(self.now_head2.parameters())

    @property
    def device(self):
        return self.m.device

    def _patch(self, x):
        return self.m.patch(x)

    def _embed_block(self, vals, mask, age_raw, L, time_enc, rev=None):
        pv = self._patch(vals); pm = torch.nan_to_num(self._patch(mask), nan=0.0)
        pv = torch.nan_to_num(torch.where(pm > 0, pv, 0.0), nan=0.0)
        feats = [time_enc, pv, pm]
        if self.age_channel:
            pa = torch.nan_to_num(self._patch(age_raw), nan=0.0)
            pr = torch.nan_to_num(self._patch(rev), nan=0.0) if rev is not None else torch.zeros_like(pa)
            feats += [pa / L[:, None, None], torch.log1p(pa) / LOG_AGE_SCALE, pr]
        x = torch.cat(feats, dim=-1).to(self.m.dtype)
        return self.m.input_patch_embedding(x), pm

    def _age_attention(self, layer, hs, B, S, key_mask, bias, block=None):
        """Cross-stream (age-axis) attention within each sample. hs: [B*S, T, d]; key_mask: [B, T, S] (1 valid);
        bias: [B*T, heads, S, S] or None; block: [S, S] additive mask (0 / -inf) or None."""
        rows, T, d = hs.shape
        x = hs.reshape(B, S, T, d).permute(0, 2, 1, 3).reshape(B * T, S, d)
        normed = layer.layer_norm(x)
        mha = layer.self_attention
        def shape(z):
            return rearrange(z, "b s (h k) -> b h s k", h=self.n_heads, k=self.d_kv)
        q, k, v = shape(mha.q(normed)), shape(mha.k(normed)), shape(mha.v(normed))
        m = key_mask.reshape(B * T, 1, 1, S).to(q.dtype)
        m = (1.0 - m) * torch.finfo(q.dtype).min
        if block is not None:
            m = m + block.to(q.dtype)[None, None]
        if bias is not None:
            m = m + bias.to(q.dtype)
        out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=m, dropout_p=mha.dropout if self.training else 0.0, scale=1.0)
        out = mha.o(rearrange(out, "b h s k -> b s (h k)"))
        x = x + layer.dropout(out)
        return x.reshape(B, T, S, d).permute(0, 2, 1, 3).reshape(rows, T, d)

    def forward(self, vals, mask, age_raw, stream_age, L, ctx_len: int, N: int = 0, target=None, target_row: int = 1,
                output_hidden: bool = False, q_anchor=None, rev=None, **kw):
        out_extra_qn_u = None
        """vals/mask/age_raw: [B, S, P]; stream_age [B, S]; L [B]; target [B, N+H] (NaN allowed).
        Rows are ordered (sample, stream); stream `target_row` (default 0 = settled) is the forecast target."""
        m = self.m
        if not self.edge:
            N = 0
        B0, S0, P = vals.shape
        use_solo = False  # (solo-row anchor superseded by the repaired-context anchor computed outside the forward)
        if rev is None:
            rev = torch.zeros_like(vals)
        if use_solo:
            # append, for every sample, a solo group holding only the target (as-of) stream: its forecast is the anchor
            solo = lambda z: torch.cat([z, z[:, target_row:target_row + 1]], dim=1)
            vals, mask, age_raw, rev = solo(vals), solo(mask), solo(age_raw), solo(rev)
            stream_age = torch.cat([stream_age, stream_age[:, target_row:target_row + 1]], dim=1)
        B, S, P = vals.shape
        rows = B * S
        dev = vals.device
        x = vals.reshape(rows, P).float()
        mk = mask.reshape(rows, P).float()
        ar = age_raw.reshape(rows, P).float()
        rv = rev.reshape(rows, P).float()
        Lr = L.float().repeat_interleave(S)
        n_out = P - ctx_len
        n_out_patches = math.ceil(n_out / self.p)
        # ---------------------------------------------------------- normalisation
        ctx = x[:, :ctx_len]
        if self.shared_norm and S > 1:
            ref = ctx.reshape(B, S, ctx_len)[:, 1]  # as-of stream
            loc = torch.nan_to_num(torch.nanmean(ref, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num((ref - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
            scale = torch.where(scale <= 0, torch.ones_like(scale), scale)
            loc = loc.repeat_interleave(S, 0); scale = scale.repeat_interleave(S, 0)
        else:
            loc = torch.nan_to_num(torch.nanmean(ctx, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num((ctx - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
            scale = torch.where(scale <= 0, torch.ones_like(scale), scale)
        xn = torch.arcsinh((x - loc) / scale)
        # ---------------------------------------------------------- context tokens
        ctx_n, ctx_m, ctx_a, ctx_r = xn[:, :ctx_len], mk[:, :ctx_len], ar[:, :ctx_len], rv[:, :ctx_len]
        n_ctx_patches = math.ceil(ctx_len / self.p)
        final_ctx_len = n_ctx_patches * self.p
        te = torch.arange(-final_ctx_len, 0, device=dev, dtype=torch.float32)
        te = repeat(te, "(n p) -> r n p", r=rows, n=n_ctx_patches, p=self.p) / m.chronos_config.time_encoding_scale
        ctx_tok, ctx_pm = self._embed_block(ctx_n, ctx_m, ctx_a, Lr, te, ctx_r)
        attn = (ctx_pm.sum(-1) > 0).to(m.dtype)
        pa = torch.nan_to_num(self._patch(ctx_a), nan=0.0)
        tok_age_ctx = pa.sum(-1) / ctx_pm.sum(-1).clamp(min=1)
        nominal = stream_age.reshape(rows).float()
        tok_age_ctx = torch.where(ctx_pm.sum(-1) > 0, tok_age_ctx, nominal[:, None])
        reg = m.shared(torch.full((rows, 1), m.config.reg_token_id, device=dev))
        # ---------------------------------------------------------- future tokens
        fut_len = n_out_patches * self.p
        fut_n = torch.zeros(rows, fut_len, device=dev); fut_m = torch.zeros(rows, fut_len, device=dev); fut_a = torch.zeros(rows, fut_len, device=dev); fut_r = torch.zeros(rows, fut_len, device=dev)
        fut_n[:, :n_out] = torch.nan_to_num(xn[:, ctx_len:], nan=0.0); fut_m[:, :n_out] = mk[:, ctx_len:]; fut_a[:, :n_out] = ar[:, ctx_len:]; fut_r[:, :n_out] = rv[:, ctx_len:]
        fut_n = torch.where(fut_m > 0, fut_n, 0.0)
        tf = torch.arange(0, fut_len, device=dev, dtype=torch.float32)
        tf = repeat(tf, "(n p) -> r n p", r=rows, n=n_out_patches, p=self.p) / m.chronos_config.time_encoding_scale
        fut_tok, fut_pm = self._embed_block(fut_n, fut_m, fut_a, Lr, tf, fut_r)
        pa_f = torch.nan_to_num(self._patch(fut_a), nan=0.0)
        tok_age_fut = pa_f.sum(-1) / fut_pm.sum(-1).clamp(min=1)
        tok_age_fut = torch.where(fut_pm.sum(-1) > 0, tok_age_fut, nominal[:, None])
        # ---------------------------------------------------------- encoder
        h = torch.cat([ctx_tok, reg, fut_tok], dim=1)
        ones = lambda n: torch.ones(rows, n, device=dev, dtype=m.dtype)
        attn_all = torch.cat([attn, ones(1), ones(n_out_patches)], dim=1)  # [rows, T]
        tok_age = torch.cat([tok_age_ctx, torch.zeros(rows, 1, device=dev), tok_age_fut], dim=1)  # [rows, T]
        enc = m.encoder
        Tn = h.shape[1]
        pos = torch.arange(Tn, device=dev).unsqueeze(0)
        ext_mask = enc._expand_and_invert_time_attention_mask(attn_all, h.dtype)
        key_mask = attn_all.reshape(B, S, Tn).permute(0, 2, 1)  # [B, T, S]
        idx = None
        if self.age_bias:
            ta = tok_age.reshape(B, S, Tn).permute(0, 2, 1)  # [B, T, S]
            idx = bucketize_age_diff(ta[:, :, :, None] - ta[:, :, None, :])  # [B, T, S, S]
        block = None
        if use_solo:
            block = torch.zeros(S, S, device=dev)
            block[S - 1, : S - 1] = float("-inf"); block[: S - 1, S - 1] = float("-inf")
        hs = enc.dropout(h)
        for li, blk in enumerate(enc.block):
            hs = blk.layer[0](hs, position_ids=pos, attention_mask=ext_mask)[0]
            bias = None
            if idx is not None:
                bias = self.bias_emb[li][idx].permute(0, 1, 4, 2, 3).reshape(B * Tn, self.n_heads, S, S)
            hs = self._age_attention(blk.layer[1], hs, B, S, key_mask, bias, block)
            hs = blk.layer[2](hs)
        hs = enc.dropout(enc.final_layer_norm(hs))
        # ---------------------------------------------------------- output heads
        # forecast: the future tokens of the target row (standard Chronos-2 alignment)
        Hs = hs.reshape(B, S, Tn, -1)
        q = m.output_patch_embedding(Hs[:, target_row, -n_out_patches:])
        q = rearrange(q, "b n (q p) -> b q (n p)", n=n_out_patches, q=m.num_quantiles, p=self.p)[:, :, :n_out]
        loc_t = loc.reshape(B, S, 1)[:, target_row]; scale_t = scale.reshape(B, S, 1)[:, target_row]
        out = {"loc": loc_t, "scale": scale_t}
        if self.forecast_anchor:
            gf = torch.sigmoid(self.gate_fc_head(Hs[:, target_row, -n_out_patches:]))  # [B, n, Q]
            gf = gf.permute(0, 2, 1).repeat_interleave(self.p, dim=-1)[:, :, :n_out]
            out["gate_fc"] = gf
            out["q_pre"] = q.float()
            if q_anchor is not None:
                qa = q_anchor.to(q.dtype)
                q = qa + gf.to(q.dtype) * (q - qa)
        out["q_norm"] = q
        if N > 0:
            # nowcast: re-use the output head on the last k context tokens of the target row (edge tokens)
            k = math.ceil(N / self.p)
            xn_row = torch.nan_to_num(xn.reshape(B, S, P)[:, target_row], nan=0.0); mk_row = mk.reshape(B, S, P)[:, target_row]
            anchor = carry_forward_anchor(xn_row, mk_row, ctx_len - N, N)  # [B, N] (normalised space)
            edge_tok = Hs[:, target_row, n_ctx_patches - k: n_ctx_patches]
            if getattr(self, "no_now_gate", False):
                g = torch.ones(B, m.num_quantiles, N, device=dev)  # no gate: the zero-initialised head alone provides the invariance
            elif self.evidence_gate:
                g = torch.sigmoid(self.gate_head(edge_tok))  # [B, k, Q]
                g = g.permute(0, 2, 1).repeat_interleave(self.p, dim=-1)[:, :, -N:]  # [B, Q, N]
                out["gate"] = g
            else:
                g = torch.sigmoid(self.now_gate)[None, :, None].expand(B, m.num_quantiles, N)
            if self.mult_head:
                h2 = self.now_head2(edge_tok)
                h2 = rearrange(h2, "b n (c q p) -> b c q (n p)", c=2, q=m.num_quantiles, p=self.p)[:, :, :, -N:]
                mm = h2[:, 0].float().clamp(-3, 3); dd = h2[:, 1].float()
                x_raw = torch.sinh(anchor.float()) * scale_t + loc_t  # [B, N] raw anchor
                q_raw = x_raw[:, None, :] * torch.exp(g.float() * mm) + scale_t[:, None, :] * g.float() * dd
                qn = torch.arcsinh((q_raw - loc_t[:, None, :]) / scale_t[:, None, :])
                if not getattr(self, "no_now_gate", False):
                    q_raw_u = x_raw[:, None, :] * torch.exp(mm) + scale_t[:, None, :] * dd  # un-gated correction (auxiliary)
                    out_extra_qn_u = torch.arcsinh((q_raw_u - loc_t[:, None, :]) / scale_t[:, None, :])
            else:
                head = self.now_head if self.separate_now_head else m.output_patch_embedding
                qn = head(edge_tok)
                qn = rearrange(qn, "b n (q p) -> b q (n p)", n=k, q=m.num_quantiles, p=self.p)[:, :, -N:]
                qn = anchor[:, None, :].to(qn.dtype) + g.to(qn.dtype) * qn
            out["qn_norm"] = qn
            q_all = torch.cat([qn.float(), q.float()], dim=-1)
        else:
            q_all = q
        if output_hidden:
            out["hidden"] = Hs
        if target is not None:
            yt = target.float()
            ymask = torch.isfinite(yt).float()
            yn = torch.arcsinh((torch.nan_to_num(yt, nan=0.0) - loc_t) / scale_t)
            ql = self.qlevels.to(q_all.dtype)[None, :, None]
            diff = yn[:, None, :] - q_all
            pl = 2 * torch.abs(diff * ((diff <= 0).float() - ql)) * ymask[:, None, :]
            if N > 0:
                # equal weight to the nowcast block and the forecast block
                loss_n = (pl[:, :, :N].sum(-1) / ymask[:, :N].sum(-1).clamp(min=1)[:, None]).sum(1).mean()
                loss_f = (pl[:, :, N:].sum(-1) / ymask[:, N:].sum(-1).clamp(min=1)[:, None]).sum(1).mean()
                out["loss"] = loss_f + self.now_weight * loss_n
                out["loss_now"], out["loss_fc"] = loss_n.detach(), loss_f.detach(); out["lf"], out["ln"] = loss_f, loss_n
                if out_extra_qn_u is not None:
                    du = yn[:, None, :N] - out_extra_qn_u.float()
                    plu = 2 * torch.abs(du * ((du <= 0).float() - ql)) * ymask[:, None, :N]
                    out["ln_u"] = (plu.sum(-1) / ymask[:, :N].sum(-1).clamp(min=1)[:, None]).sum(1).mean()
                    if "gate" in out:
                        out["lg"] = gate_supervision_loss(out["gate"], out_extra_qn_u.float(), anchor.float(), yn[:, :N], ymask[:, :N], self.qlevels)
            else:
                out["loss"] = (pl.sum(-1) / ymask.sum(-1).clamp(min=1)[:, None]).sum(1).mean()
        out["quantiles"] = torch.sinh(q_all.float().clamp(-4.6052, 4.6052)) * scale_t[:, None, :] + loc_t[:, None, :]  # [B, Q, N+H]
        return out
