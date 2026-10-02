import sys, os, json, argparse, time
sys.path.insert(0, ".")
import numpy as np, torch
from recast.datasets import load_bundle
from recast.protocol import make_tasks, metrics, DEFAULTS
from recast.views import build_streams, as_of_series, settled_series
from recast.synth import nowcast_window
from recast.registry import build_model, QIDX, NMAX_OVERRIDE
from recast.anchor import repaired_anchor, repaired_anchor_mc, combine_forecast, repaired_anchor_paths, repaired_anchor_mc_copula, repaired_anchor_moment
from recast.synth import model_nowcast_window

ap = argparse.ArgumentParser()
ap.add_argument("--domain", required=True)
ap.add_argument("--arch", default="chronos2")
ap.add_argument("--model", default="zeroshot", help="zeroshot | naive | oracle | path to checkpoint")
ap.add_argument("--tag", default=None)
ap.add_argument("--out", default="results/rrbench")
ap.add_argument("--max_origins", type=int, default=None)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--N", type=int, default=None, help="override nowcast window")
ap.add_argument("--C", type=int, default=None, help="override context length (data-efficiency experiment)")
ap.add_argument("--no_age_channel", action="store_true")
ap.add_argument("--no_age_bias", action="store_true")
ap.add_argument("--no_age_attention", action="store_true")
ap.add_argument("--per_series_norm", action="store_true")
ap.add_argument("--query_row", type=int, default=None)
ap.add_argument("--ages", default=None, help="comma-separated view ages override; 'none' = as-of only")
ap.add_argument("--target_row", type=int, default=None)
ap.add_argument("--shift", action="store_true")
ap.add_argument("--evidence_gate", action="store_true")
ap.add_argument("--separate_now_head", action="store_true")
ap.add_argument("--forecast_anchor", action="store_true")
ap.add_argument("--mult_head", action="store_true")
ap.add_argument("--no_rev", action="store_true")
ap.add_argument("--mc_anchor", action="store_true", help="propagate nowcast uncertainty through the anchor (3 quantile paths)")
ap.add_argument("--gate_off", action="store_true", help="use the anchor alone as the forecast (diagnostic)")
ap.add_argument("--pre_only", action="store_true", help="use the adapted model's direct forecast alone, no anchor (diagnostic)")
ap.add_argument("--anchor_mode", default=None, help="paths | mc | moment (overrides --mc_anchor); see recast/anchor.py")
ap.add_argument("--anchor_K", type=int, default=3, help="number of nowcast paths for the anchor (paths: 1,2,3,5,9; mc: any)")
ap.add_argument("--anchor_rho", type=float, default=0.8, help="AR(1) copula correlation across nowcast positions (mc mode)")
ap.add_argument("--n_views", type=int, default=16, help="number of fixed-age views (age-grid size)")
ap.add_argument("--L_model", type=int, default=None, help="settlement age assumed by the model (mis-specification experiment); scoring keeps the domain's L")
ap.add_argument("--no_now_gate", action="store_true")
args = ap.parse_args()

b = load_bundle(args.domain)
freq, L = b["freq"], b["L"]
d = DEFAULTS[freq]; C, H = d["C"], d["H"]
plain_mode = args.model in ("naive", "oracle")
def plain_mode_flag(a):
    return a.model in ("naive", "oracle") or a.shift
N_score = args.N or nowcast_window(freq, L)
Lm = args.L_model or L
N = N_score if plain_mode else max(N_score, model_nowcast_window(freq, Lm)) if Lm >= N_score else model_nowcast_window(freq, Lm)


def with_L_model(A, L, Lm):


    if Lm == L:
        return A
    if Lm > L:
        return np.concatenate([A, np.repeat(A[:, L:L + 1], Lm - L, axis=1)], axis=1)
    Ap = A[:, :Lm + 1].copy()


    miss = ~np.isfinite(Ap[:, Lm]) & np.isfinite(A[:, L])
    Ap[miss, Lm] = A[miss, L]
    return Ap
tasks = make_tasks(b, max_origins=args.max_origins)
if args.C:
    C = args.C
tag = args.tag or (f"{args.arch}_{args.model}" if args.model in ("zeroshot", "naive", "oracle") else os.path.basename(args.model).replace(".pt", ""))
os.makedirs(args.out, exist_ok=True)
dev = "cuda"


if plain_mode:
    from recast.plain import Plain
    plain = Plain(args.arch, bs=args.bs)
else:
    model = build_model(args.arch, age_channel=not args.no_age_channel, age_bias=not args.no_age_bias,
                        shared_norm=not args.per_series_norm, age_attention=not args.no_age_attention, query_row=args.query_row, edge=not args.shift, evidence_gate=args.evidence_gate, separate_now_head=args.separate_now_head, forecast_anchor=args.forecast_anchor, mult_head=args.mult_head, no_now_gate=args.no_now_gate).to(dev).eval()
    if args.model != "zeroshot":
        sd = torch.load(args.model, map_location=dev)
        sd = sd.get("model", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("loaded", args.model, "missing", len(missing), "unexpected", len(unexpected))
        if len(unexpected):
            print("unexpected keys sample", unexpected[:5])
    target_row = args.target_row if args.target_row is not None else getattr(model, "target_row_default", 0)
ages = None
if args.ages:
    ages = [] if args.ages == "none" else [int(a) for a in args.ages.split(",")]
anchor_model = None
if (not plain_mode) and args.forecast_anchor:
    anchor_model = build_model(args.arch, age_channel=False, age_bias=False, shared_norm=False, age_attention=False, edge=True, query_row=0).to(dev).eval()


index, Y, S, EV, YN, XN, items, ctxs = [], [], [], [], [], [], [], []
for tk in tasks:
    y = tk.targets(); sc = tk.scale(); ev = tk.eval_mask()
    for i, t in enumerate(tk.origins):
        t = int(t)
        index.append((tk.sid, t)); Y.append(y[i]); S.append(sc[i]); EV.append(bool(ev[i]))
        if plain_mode:
            ctxs.append(as_of_series(tk.A, tk.R, t, L, C) if args.model == "naive" else settled_series(tk.A, tk.R, t, L, C))
        else:
            items.append(build_streams(with_L_model(tk.A, L, Lm), tk.R, t, Lm, C, N, H, ages=ages, with_target=False, edge=not args.shift, n_views=args.n_views))
        u = np.arange(t - N_score + 1, t + 1)
        yn = np.full(N_score, np.nan, np.float32); xn = np.full(N_score, np.nan, np.float32)
        ok = (u >= 0) & (u < tk.A.shape[0])
        yn[ok] = tk.A[u[ok], L]
        Rv = np.where(np.isfinite(tk.R), tk.R, np.inf)
        rel = ok & (Rv[np.clip(u, 0, tk.A.shape[0] - 1)] <= t)
        xn[rel] = tk.A[u[rel], np.minimum(t - u, L)[rel]]

        xa = as_of_series(tk.A, tk.R, t, L, C)
        last = xa[np.isfinite(xa)][-1] if np.isfinite(xa).any() else np.nan
        for k in range(N_score):
            if not np.isfinite(xn[k]):
                xn[k] = xn[k - 1] if k > 0 and np.isfinite(xn[k - 1]) else last
            else:
                last = xn[k]
        YN.append(yn); XN.append(xn)
Y = np.stack(Y); S = np.asarray(S); EV = np.asarray(EV); YN = np.stack(YN); XN = np.stack(XN)
n = len(index)
print(f"domain {args.domain} freq {freq} L {L} C {C} H {H} N {N} origins {n} eval {EV.sum()} model {tag}", flush=True)


Qf = np.full((n, H, 9), np.nan, np.float32); Qn = np.full((n, N_score, 9), np.nan, np.float32); Qn_full = np.full((n, N, 9), np.nan, np.float32)
t0 = time.time()
if plain_mode:
    Qf = plain.predict(ctxs, H)
else:
    qidx = QIDX[args.arch]
    with torch.no_grad():
        for s in range(0, n, args.bs):
            its = items[s: s + args.bs]
            B = len(its)
            vals = torch.tensor(np.stack([it["vals"] for it in its])).to(dev)
            mask = torch.tensor(np.stack([it["mask"] for it in its])).to(dev)
            age = torch.tensor(np.stack([it["age_raw"] for it in its])).to(dev)
            rev = torch.tensor(np.stack([it["rev"] for it in its])).to(dev)
            if args.no_rev:
                rev = torch.zeros_like(rev)
            sage = torch.tensor(np.stack([it["stream_age"] for it in its])).to(dev)
            Lt = torch.full((B,), Lm, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(vals, mask, age, sage, Lt, ctx_len=its[0]['ctx_len'], N=N, target_row=target_row, rev=rev)
            if anchor_model is not None:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    aq = out["quantiles"][:, :, :N].float(); cl_ = its[0]['ctx_len']
                    if args.anchor_mode == "paths":
                        q_anchor = repaired_anchor_paths(anchor_model, vals, mask, Lt, cl_, N, H, aq, out["loc"], out["scale"], model.uses_arcsinh, model.qlevels, target_row, K=args.anchor_K)
                    elif args.anchor_mode == "mc":
                        q_anchor = repaired_anchor_mc_copula(anchor_model, vals, mask, Lt, cl_, N, H, aq, out["loc"], out["scale"], model.uses_arcsinh, model.qlevels, target_row, K=args.anchor_K, rho=args.anchor_rho, seed=s)
                    elif args.anchor_mode == "moment":
                        q_anchor = repaired_anchor_moment(anchor_model, vals, mask, Lt, cl_, N, H, aq, out["loc"], out["scale"], model.uses_arcsinh, model.qlevels, target_row, K=args.anchor_K)
                    elif args.mc_anchor:
                        q_anchor = repaired_anchor_mc(anchor_model, vals, mask, Lt, its[0]['ctx_len'], N, H, out["quantiles"][:, :, :N].float(), out["loc"], out["scale"], model.uses_arcsinh, model.qlevels, target_row)
                    else:
                        now_med = out["quantiles"][:, model.median_idx, :N].float()
                        q_anchor = repaired_anchor(anchor_model, vals, mask, Lt, its[0]['ctx_len'], N, H, now_med, out["loc"], out["scale"], model.uses_arcsinh, target_row)
                if args.gate_off:
                    out["gate_fc"] = torch.zeros_like(out["gate_fc"])
                if args.pre_only:
                    out["gate_fc"] = torch.ones_like(out["gate_fc"])
                combine_forecast(out, q_anchor, N, H, model.uses_arcsinh, model.qlevels)
            q = out["quantiles"].float().cpu().numpy()[:, qidx, :]
            q = np.sort(q.transpose(0, 2, 1), axis=-1)
            Qn_full[s: s + B] = q[:, :N]; Qf[s: s + B] = q[:, N: N + H]
            if N >= N_score:
                Qn[s: s + B] = q[:, N - N_score: N]
            else:
                Qn[s: s + B, N_score - N:] = q[:, :N]; Qn[s: s + B, :N_score - N] = XN[s: s + B, :N_score - N, None]
sec = time.time() - t0


res = dict(domain=args.domain, arch=args.arch, model=tag, freq=freq, L=L, L_model=Lm, C=C, H=H, N=N_score, N_model=N, n=n, n_eval=int(EV.sum()), seconds=sec, n_views=args.n_views, anchor_mode=args.anchor_mode or ("mc3" if args.mc_anchor else "median"), anchor_K=args.anchor_K)
res["forecast"] = metrics(Qf[EV], Y[EV], S[EV])
if not plain_mode:
    res["nowcast"] = metrics(Qn[EV], YN[EV], S[EV])
    ae = np.abs(XN[EV] - YN[EV]) / S[EV][:, None]
    res["nowcast_naive_MASE"] = float(np.nanmean(ae))
    res["nowcast_naive_MASE_h"] = [float(np.nanmean(ae[:, k])) for k in range(N_score)]
print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}))
print("forecast:", {k: round(v, 4) for k, v in res["forecast"].items() if not isinstance(v, list)})
if "nowcast" in res:
    print("nowcast :", {k: round(v, 4) for k, v in res["nowcast"].items() if not isinstance(v, list)}, "naive-now MASE", round(res["nowcast_naive_MASE"], 4))
json.dump(res, open(f"{args.out}/{tag}__{args.domain}.json", "w"), indent=1)
np.savez_compressed(f"{args.out}/{tag}__{args.domain}.npz", Qf=Qf, Qn=Qn, Qn_full=Qn_full, Y=Y, YN=YN, XN=XN, S=S, EV=EV,
                    index_sid=np.array([i[0] for i in index]), index_t=np.array([i[1] for i in index]))
print("saved", f"{args.out}/{tag}__{args.domain}.npz", f"{sec:.0f}s")
