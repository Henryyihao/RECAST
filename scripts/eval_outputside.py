import sys, os, json, argparse, time
sys.path.insert(0, ".")
import numpy as np, torch
from recast.datasets import load_bundle
from recast.protocol import make_tasks, metrics, DEFAULTS
from recast.views import as_of_series
from recast.synth import nowcast_window
from recast.outputside import OutputSideBaselines
from recast.registry import build_model, QIDX

HALF_LIFE = {"D": 90, "W": 26, "M": 36, "Q": 16, "h": 24 * 30}
ap = argparse.ArgumentParser()
ap.add_argument("--domain", required=True)
ap.add_argument("--arch", default="chronos2")
ap.add_argument("--out", default="data/outputside")
ap.add_argument("--max_origins", type=int, default=None)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--tag", default=None)
args = ap.parse_args()
tag = args.tag or {"chronos2": "chronos2", "bolt_s": "chronos_bolt", "bolt_b": "bolt_b",
                   "toto": "toto", "timemoe": "timemoe"}[args.arch]
b = load_bundle(args.domain)
freq, L = b["freq"], b["L"]
d = DEFAULTS[freq]; C, H = d["C"], d["H"]
N = nowcast_window(freq, L)
tasks = make_tasks(b, max_origins=args.max_origins)
index = []
for tk in tasks:
    for t in tk.origins:
        index.append((tk.sid, int(t)))
dev = "cuda"
model = build_model(args.arch, age_channel=False, age_bias=False, shared_norm=False,
                    age_attention=False, edge=True, query_row=0).to(dev).eval()


def predict(contexts):
    out = np.full((len(contexts), H, 9), np.nan, np.float32)
    with torch.no_grad():
        for s in range(0, len(contexts), args.bs):
            chunk = contexts[s: s + args.bs]
            X = np.stack(chunk).astype(np.float32)
            B = X.shape[0]
            vals = torch.tensor(np.concatenate([X, np.full((B, H), np.nan, np.float32)], 1))[:, None, :].to(dev)
            mask = torch.isfinite(vals).float()
            age = torch.zeros_like(vals)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(vals, mask, age, torch.zeros(B, 1, device=dev), torch.full((B,), L, device=dev),
                          ctx_len=C, target_row=0)
            q = o["quantiles"].float().cpu().numpy()[:, QIDX[args.arch], :].transpose(0, 2, 1)
            out[s: s + B] = np.sort(q, axis=-1)
    return out


states = {}
Y, S = [], []
row = 0
for tk in tasks:
    xa = tk.contexts("asof")
    y = tk.targets(); sc = tk.scale()
    Y.append(y)
    S.append(sc)

naive_ctx = [as_of_series(tk.A, tk.R, int(t), L, C) for tk in tasks for t in tk.origins]
Qn = predict(naive_ctx)
Y = np.concatenate(Y); S = np.concatenate(S)

import types
class _State:
    pass

class _States:
    def __init__(self, tasks):
        self._d = {}
        for tk in tasks:
            fin = tk.A[:, L]
            scale = np.nanmedian(np.abs(fin[np.isfinite(fin)])) if np.isfinite(fin).any() else 1.0
            st = _State()
            st.c = max(0.05 * scale, 1e-6)
            self._d[tk.sid] = st

    def __getitem__(self, key):
        return self._d[key]

osb = OutputSideBaselines(tasks, _States(tasks), Qn, Y, S, L, H, index, half_life=HALF_LIFE[freq])
out = osb.run()
os.makedirs(args.out, exist_ok=True)
np.savez_compressed(f"{args.out}/{tag}__{args.domain}.npz",
                    Q_b2f=out["b2f"], Q_ra_prov=out["ra_prov"], Q_ra_settled=out["ra_settled"],
                    Q_naive=Qn, Y=Y, S=S, index_sid=np.array([i[0] for i in index]),
                    index_t=np.array([i[1] for i in index]))
print("saved", f"{args.out}/{tag}__{args.domain}.npz")
