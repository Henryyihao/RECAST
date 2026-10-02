import sys, os, json, argparse, time
sys.path.insert(0, ".")
import numpy as np, torch
from recast.datasets import load_bundle
from recast.protocol import make_tasks, metrics, DEFAULTS
from recast.views import as_of_series
from recast.synth import nowcast_window
from recast.baselines import chain_ladder_nowcast, repaired_context, LEVELS
from recast.models.va_chronos2 import VAChronos2

QIDX = [2, 4, 6, 8, 10, 12, 14, 16, 18]
C2 = "models/chronos2"
ap = argparse.ArgumentParser()
ap.add_argument("--domain", required=True)
ap.add_argument("--out", default="results/rrbench")
ap.add_argument("--max_origins", type=int, default=None)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--C", type=int, default=None, help="restrict the chain ladder to the last C positions")
ap.add_argument("--tag", default="baselines")
ap.add_argument("--arch", default="chronos2", help="backbone for the two-stage pipeline (chronos2 | bolt_s | bolt_b | toto | timemoe)")
args = ap.parse_args()
if args.arch != "chronos2" and args.tag == "baselines":
    args.tag = f"baselines_{args.arch}"

b = load_bundle(args.domain)
freq, L = b["freq"], b["L"]
d = DEFAULTS[freq]; C, H = d["C"], d["H"]
N = nowcast_window(freq, L)
tasks = make_tasks(b, max_origins=args.max_origins)
dev = "cuda"
if args.arch == "chronos2":
    model = VAChronos2(C2, age_channel=False, age_bias=False, shared_norm=False).to(dev).eval()
elif args.arch != "none":
    from recast.plain import Plain
    plain = Plain(args.arch, bs=args.bs)

def predict_plain(ctxs):

    if args.arch != "chronos2":
        return plain.predict(ctxs, H)
    out = np.full((len(ctxs), H, 9), np.nan, np.float32)
    with torch.no_grad():
        for s in range(0, len(ctxs), args.bs):
            X = np.stack(ctxs[s: s + args.bs]).astype(np.float32)
            Bn = X.shape[0]
            vals = torch.tensor(np.concatenate([X, np.full((Bn, H), np.nan, np.float32)], 1))[:, None, :].to(dev)
            mask = torch.isfinite(vals).float(); age = torch.zeros_like(vals)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(vals, mask, age, torch.zeros(Bn, 1, device=dev), torch.full((Bn,), L, device=dev), ctx_len=C, target_row=0)
            q = o["quantiles"].float().cpu().numpy()[:, QIDX, :].transpose(0, 2, 1)
            out[s: s + Bn] = np.sort(q, axis=-1)
    return out

index, Y, S, EV, YN = [], [], [], [], []
Qn = []; ctx_med = []; ctx_mc = [[] for _ in range(5)]
t0 = time.time()
cache = f"{args.out}/clctx__{args.domain}{'_C' + str(args.C) if args.C else ''}.npz"
if os.path.exists(cache) and args.max_origins in (None, {"dv": 200, "chng_flu": 200, "hosp_cov": 200, "kit": 200, "eia930": 120}.get(args.domain)):
    z = np.load(cache, allow_pickle=True)
    index = [(str(a), int(b)) for a, b in zip(z["index_sid"], z["index_t"])]; Y = list(z["Y"]); S = list(z["S"]); EV = list(z["EV"]); YN = list(z["YN"]); Qn = list(z["Qn"])
    ctx_med = list(z["ctx_med"]); ctx_mc = [list(z["ctx_mc"][k]) for k in range(5)]
    tasks = []
    print(f"loaded chain-ladder cache {cache}", flush=True)
for tk in tasks:
    y = tk.targets(); sc = tk.scale(); ev = tk.eval_mask()
    for i, t in enumerate(tk.origins):
        t = int(t)
        index.append((tk.sid, t)); Y.append(y[i]); S.append(sc[i]); EV.append(bool(ev[i]))
        u = np.arange(t - N + 1, t + 1)
        yn = np.full(N, np.nan, np.float32); ok = (u >= 0) & (u < tk.A.shape[0]); yn[ok] = tk.A[u[ok], L]; YN.append(yn)
        q, point_all, f = chain_ladder_nowcast(tk.A, tk.R, t, L, N, W=(args.C - L) if args.C else None)
        Qn.append(q)
        x = as_of_series(tk.A, tk.R, t, L, C)
        u_ctx = np.arange(t - C + 1, t + 1)
        xr = repaired_context(x, None, point_all, u_ctx)
        ctx_med.append(xr)

        for k, tau_i in enumerate([0, 2, 4, 6, 8]):
            xk = xr.copy()
            for j in range(N):
                jj = C - N + j
                if np.isfinite(q[j, tau_i]) and np.isfinite(xk[jj]):
                    xk[jj] = q[j, tau_i]
            ctx_mc[k].append(xk)
Y = np.stack(Y); S = np.asarray(S); EV = np.asarray(EV); YN = np.stack(YN); Qn = np.stack(Qn).astype(np.float32)
n = len(index)
print(f"domain {args.domain} L {L} N {N} origins {n} chain ladder done {time.time()-t0:.0f}s", flush=True)
if not os.path.exists(cache):
    np.savez_compressed(cache, index_sid=np.array([i[0] for i in index]), index_t=np.array([i[1] for i in index]), Y=Y, S=S, EV=EV, YN=YN, Qn=Qn,
                        ctx_med=np.stack(ctx_med), ctx_mc=np.stack([np.stack(c) for c in ctx_mc]))
if args.arch == "none":
    sys.exit(0)
Qf_2s = predict_plain(ctx_med)
Qmc = np.stack([predict_plain(c) for c in ctx_mc])

samp = Qmc.transpose(1, 2, 0, 3).reshape(n, H, 45)
Qf_mc = np.quantile(samp, LEVELS, axis=-1).transpose(1, 2, 0).astype(np.float32)

res = dict(domain=args.domain, freq=freq, L=L, N=N, H=H, n=n, n_eval=int(EV.sum()))
res["nowcast_cl"] = metrics(Qn[EV], YN[EV], S[EV])
res["forecast_2s_point"] = metrics(Qf_2s[EV], Y[EV], S[EV])
res["forecast_2s_mc"] = metrics(Qf_mc[EV], Y[EV], S[EV])
for k, v in res.items():
    if isinstance(v, dict):
        print(k, {kk: round(vv, 4) for kk, vv in v.items() if not isinstance(vv, list)})
os.makedirs(args.out, exist_ok=True)
json.dump(res, open(f"{args.out}/{args.tag}__{args.domain}.json", "w"), indent=1)
np.savez_compressed(f"{args.out}/{args.tag}__{args.domain}.npz", Qn_cl=Qn, Qf_2s=Qf_2s, Qf_mc=Qf_mc, Y=Y, YN=YN, S=S, EV=EV,
                    index_sid=np.array([i[0] for i in index]), index_t=np.array([i[1] for i in index]))
print("saved", f"{time.time()-t0:.0f}s")
