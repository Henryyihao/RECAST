from __future__ import annotations
import numpy as np, torch
from .registry import PATHS

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def clean(x, min_len=8):
    x = np.asarray(x, dtype=np.float32).copy()
    fin = np.isfinite(x)
    if not fin.any():
        return np.zeros(min_len, np.float32)
    x = x[int(np.argmax(fin)):]
    fin = np.isfinite(x)
    if not fin.all():
        idx = np.arange(len(x)); x[~fin] = np.interp(idx[~fin], idx[fin], x[fin])
    if len(x) < min_len:
        x = np.concatenate([np.full(min_len - len(x), x[0], np.float32), x])
    return x


class Plain:
    def __init__(self, arch, bs=64, device="cuda"):
        self.arch, self.bs, self.dev = arch, bs, device
        if arch == "chronos2":
            from chronos import Chronos2Pipeline
            self.pipe = Chronos2Pipeline.from_pretrained(PATHS[arch], device_map=device, dtype=torch.float32)
        elif arch in ("bolt_s", "bolt_b"):
            from chronos import ChronosBoltPipeline
            self.pipe = ChronosBoltPipeline.from_pretrained(PATHS[arch], device_map=device, dtype=torch.float32)
        elif arch in ("toto", "toto22"):
            from toto2 import Toto2Model
            self.model = Toto2Model.from_pretrained(PATHS[arch]).to(device).eval()
        elif arch == "timemoe":
            from transformers import AutoModelForCausalLM
            from .models.common import fix_timemoe_rotary
            self.model = AutoModelForCausalLM.from_pretrained(PATHS[arch], trust_remote_code=True, dtype=torch.float32,
                                                              attn_implementation="eager").to(device).eval()
            fix_timemoe_rotary(self.model)
        else:
            raise ValueError(arch)

    @torch.no_grad()
    def predict(self, contexts, H):
        ctxs = [clean(c) for c in contexts]
        out = np.full((len(ctxs), H, 9), np.nan, np.float32)
        order = np.argsort([len(c) for c in ctxs])
        for s in range(0, len(order), self.bs):
            idx = order[s: s + self.bs]
            batch = [ctxs[i] for i in idx]
            q = self._batch(batch, H)
            out[idx] = np.sort(q, axis=-1)
        return out

    def _batch(self, batch, H):
        if self.arch == "chronos2":
            t = [torch.tensor(c)[None, :] for c in batch]
            q, _ = self.pipe.predict_quantiles(t, prediction_length=H, quantile_levels=LEVELS)
            return np.stack([a[0].cpu().numpy() for a in q])
        if self.arch in ("bolt_s", "bolt_b"):
            t = [torch.tensor(c) for c in batch]
            q, _ = self.pipe.predict_quantiles(t, prediction_length=H, quantile_levels=LEVELS)
            return q.cpu().numpy()
        if self.arch in ("toto", "toto22"):
            n = len(batch); Lm = max(len(c) for c in batch)
            Lm = int(np.ceil(Lm / 32) * 32)
            X = np.zeros((n, 1, Lm), np.float32); M = np.zeros((n, 1, Lm), bool)
            for i, c in enumerate(batch):
                X[i, 0, Lm - len(c):] = c; M[i, 0, Lm - len(c):] = True
            inp = {"target": torch.tensor(X, device=self.dev), "target_mask": torch.tensor(M, device=self.dev),
                   "series_ids": torch.zeros(n, 1, dtype=torch.long, device=self.dev)}
            q = self.model.forecast(inp, horizon=H, decode_block_size=768, has_missing_values=True)
            return q[:, :, 0, :].permute(1, 2, 0).float().cpu().numpy()
        if self.arch == "timemoe":
            n = len(batch); Lm = max(len(c) for c in batch)
            X = np.zeros((n, Lm), np.float32)
            preds = np.zeros((n, H), np.float32)
            for i, c in enumerate(batch):
                mu, sd = c.mean(), c.std() + 1e-6
                X[i, Lm - len(c):] = (c - mu) / sd
                X[i, : Lm - len(c)] = X[i, Lm - len(c)]
            xt = torch.tensor(X, device=self.dev)

            hs = self.model.model(input_ids=xt[..., None], use_cache=False, return_dict=True).last_hidden_state[:, -1]
            hl = max(h for h in self.model.horizon_length_map if h <= max(H, 1)) if H < 64 else 64
            hl = 64 if H > 32 else hl
            head = self.model.lm_heads[self.model.horizon_length_map[hl]]
            p = head(hs).view(n, hl)
            if hl < H:
                cur = xt
                outs = []
                remaining = H
                while remaining > 0:
                    hs = self.model.model(input_ids=cur[..., None], use_cache=False, return_dict=True).last_hidden_state[:, -1]
                    step = min(64, remaining)
                    hh = max(h for h in self.model.horizon_length_map if h <= step)
                    pr = self.model.lm_heads[self.model.horizon_length_map[hh]](hs).view(n, hh)
                    outs.append(pr); cur = torch.cat([cur, pr], 1); remaining -= hh
                p = torch.cat(outs, 1)
            p = p[:, :H].float().cpu().numpy()
            for i, c in enumerate(batch):
                mu, sd = c.mean(), c.std() + 1e-6
                preds[i] = p[i] * sd + mu
            return np.repeat(preds[:, :, None], 9, axis=-1)
        raise ValueError(self.arch)
