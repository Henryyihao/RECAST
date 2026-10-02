import sys, os, json, argparse, numpy as np, torch
sys.path.insert(0, ".")
from recast.synth import _epi, _kernel, _delay_pmf, apply_release
from recast.views import build_streams
from recast.baselines import chain_ladder_nowcast, LEVELS
from recast.registry import build_model, QIDX
from recast.anchor import repaired_anchor, repaired_anchor_mc, combine_forecast
from recast.synth import model_nowcast_window
from synth_metrics import pinball_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--arch", default="chronos2")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--out", default="results/mech")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--evidence_gate", action="store_true")
ap.add_argument("--separate_now_head", action="store_true")
ap.add_argument("--forecast_anchor", action="store_true")
ap.add_argument("--mult_head", action="store_true")
ap.add_argument("--mc_anchor", action="store_true")
ap.add_argument("--no_rev", action="store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = "cuda"
rng = np.random.default_rng(args.seed)
L, C, H, freq = 40, 256, 28, "D"
N = model_nowcast_window(freq, L)


def controlled_backfill(rng, y, L, pmf, kappa=100.0, switch=None, pmf2=None):

    T = len(y)
    A = np.full((T, L + 1), np.nan)
    for u in range(T):
        p = pmf if (switch is None or u < switch) else pmf2
        n = int(np.round(y[u]))
        if n > 0:
            pp = rng.dirichlet(np.maximum(kappa * p, 1e-3))
            cnt = rng.multinomial(n, pp).astype(float)
        else:
            cnt = np.zeros(L + 1)
        A[u] = np.cumsum(cnt); A[u, L] = n
    return A


def make_series(rng, T, pmf, switch=None, pmf2=None, kappa=100.0):
    y = _epi(rng, T, "D")
    y = np.clip(y, 0, None)
    A = controlled_backfill(rng, y, L, pmf, kappa, switch, pmf2)
    R = np.arange(T, dtype=float)
    return y, A.astype(np.float32), R


anchor_model = None
if args.forecast_anchor:
    anchor_model = build_model(args.arch, age_channel=False, age_bias=False, shared_norm=False, age_attention=False, edge=True, query_row=0).to(dev).eval()


def predict_recast(model, items, bs=32):
    out = []
    with torch.no_grad():
        for s in range(0, len(items), bs):
            its = items[s: s + bs]
            tt = lambda k: torch.tensor(np.stack([it[k] for it in its])).to(dev)
            o = model(tt("vals"), tt("mask"), tt("age_raw"), tt("stream_age"), torch.full((len(its),), L, device=dev), ctx_len=C, N=N, target_row=1, output_hidden=True, rev=tt("rev"))
            if anchor_model is not None:
                Lt = torch.full((len(its),), L, device=dev)
                if args.mc_anchor:
                    qa = repaired_anchor_mc(anchor_model, tt("vals"), tt("mask"), Lt, C, N, H, o["quantiles"][:, :, :N].float(), o["loc"], o["scale"], model.uses_arcsinh, model.qlevels, 1)
                else:
                    now_med = o["quantiles"][:, model.median_idx, :N].float()
                    qa = repaired_anchor(anchor_model, tt("vals"), tt("mask"), Lt, C, N, H, now_med, o["loc"], o["scale"], model.uses_arcsinh, 1)
                combine_forecast(o, qa, N, H, model.uses_arcsinh, model.qlevels)
            q = np.sort(o["quantiles"].float().cpu().numpy()[:, QIDX[args.arch]].transpose(0, 2, 1), -1)
            hid = o["hidden"][:, 1].float().cpu().numpy()
            for i in range(len(its)):
                out.append((q[i, :N], q[i, N:], hid[i]))
    return out


def oracle_quantiles(x_a, a, F, y_u, kappa=100.0):

    lam = max(y_u * (1 - F[a]), 1e-9)
    from scipy.stats import poisson
    return x_a + poisson.ppf(LEVELS, lam)


def oracle_cl(x_a, a, F):
    return np.full(len(LEVELS), x_a / max(F[a], 1e-6))


MK = dict(evidence_gate=args.evidence_gate, separate_now_head=args.separate_now_head, forecast_anchor=args.forecast_anchor, mult_head=args.mult_head)
model = build_model(args.arch, **MK).to(dev).eval()
init_model = None
sd = torch.load(args.model, map_location=dev)["model"]; print(model.load_state_dict(sd, strict=False))
res = {}

recs = []
items, meta = [], []
for i in range(args.n):
    pmf = _delay_pmf(rng, L)
    mean_delay = float((np.arange(L + 1) * pmf).sum())
    F = np.cumsum(pmf)
    T = C + L + H + 60
    y, A, R = make_series(rng, T, pmf)
    t = int(rng.integers(C, T - H))
    items.append(build_streams(A, R, t, L, C, N, H))
    meta.append(dict(pmf=pmf, F=F, y=y, A=A, R=R, t=t, mean_delay=mean_delay))
preds = predict_recast(model, items)
rows = []
hid_feats, hid_targets = [], []
for (qn, qf, hid), md in zip(preds, meta):
    t, A, F, y = md["t"], md["A"], md["F"], md["y"]
    pos = np.arange(t - N + 1, t + 1)
    ycl, _, _ = chain_ladder_nowcast(A, md["R"], t, L, N)
    sc = np.nanmean(np.abs(np.diff(A[max(0, t - 2 * C): t + 1, L]))) + 1e-6
    for k, u in enumerate(pos):
        a = t - u
        x_a = A[u, a]
        truth = A[u, L]
        qo = oracle_quantiles(x_a, a, F, y[u])
        qc = oracle_cl(x_a, a, F)
        rows.append(dict(age=a, truth=truth, x=x_a, recast=qn[k], cl=ycl[k], oracle=qo, oracle_cl=qc, scale=sc))
    hid_feats.append(hid.mean(0)); hid_targets.append(np.log(md["mean_delay"] + 1e-3))
def agg(rows, key):
    m = np.stack([r[key] for r in rows]); y = np.array([r["truth"] for r in rows]); s = np.array([r["scale"] for r in rows])
    ok = np.isfinite(m).all(1) & np.isfinite(y)
    return pinball_metrics(m[ok], y[ok], s[ok])
res["M1"] = {k: agg(rows, k) for k in ["recast", "cl", "oracle", "oracle_cl"]}
res["M1"]["naive"] = pinball_metrics(np.repeat(np.array([[r["x"] for r in rows]]).T, 9, 1), np.array([r["truth"] for r in rows]), np.array([r["scale"] for r in rows]))

res["M1_by_age"] = {}
for k in ["recast", "cl", "oracle", "oracle_cl", "naive"]:
    by = []
    for a in range(N):
        rr = [r for r in rows if r["age"] == a]
        if k == "naive":
            m = np.repeat(np.array([[r["x"] for r in rr]]).T, 9, 1)
        else:
            m = np.stack([r[k] for r in rr])
        by.append(pinball_metrics(m, np.array([r["truth"] for r in rr]), np.array([r["scale"] for r in rr]))["MASE"])
    res["M1_by_age"][k] = by
print("M1", {k: {m: round(v[m], 4) for m in ["MASE", "CRPS_s", "COV80"]} for k, v in res["M1"].items()})

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
X = np.stack(hid_feats); yv = np.array(hid_targets)
pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), X, yv, cv=5)
r2_recast = 1 - ((pred - yv) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
init_model = build_model(args.arch, **MK).to(dev).eval()
preds0 = predict_recast(init_model, items)
X0 = np.stack([h.mean(0) for (_, _, h) in preds0])
pred0 = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), X0, yv, cv=5)
r2_init = 1 - ((pred0 - yv) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()

raw = np.array([[np.nanmean(it["vals"][2, :C] / np.maximum(it["vals"][0, :C], 1e-6)) if np.isfinite(it["vals"][0, :C]).any() else 0.0] for it in items])
raw = np.nan_to_num(raw)
predr = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), raw, yv, cv=5)
r2_raw = 1 - ((predr - yv) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
res["M3"] = dict(r2_recast=float(r2_recast), r2_init=float(r2_init), r2_raw_ratio=float(r2_raw), n=len(yv))
print("M3 probing R2: RECAST", round(r2_recast, 3), "init", round(r2_init, 3), "raw-ratio", round(r2_raw, 3))
del init_model; torch.cuda.empty_cache()

rng2 = np.random.default_rng(args.seed + 1)
K = 61
err = {"recast": np.zeros(K), "cl": np.zeros(K), "oracle_cl": np.zeros(K), "naive": np.zeros(K)}; cnt = np.zeros(K)
items2, meta2 = [], []
for i in range(args.n // 2):
    pmf1 = _delay_pmf(rng2, L);

    fac = float(np.exp(rng2.uniform(np.log(0.4), np.log(2.5))))
    d = np.arange(L + 1); cdf1 = np.cumsum(pmf1)
    c2 = np.interp(d / fac, d, cdf1, left=0, right=1); c2[-1] = 1
    pmf2 = np.diff(np.concatenate([[0], c2])); pmf2 = np.maximum(pmf2, 1e-9); pmf2 /= pmf2.sum()
    T = C + L + H + K + 20
    ts = C + 10
    y, A, R = make_series(rng2, T, pmf1, switch=ts, pmf2=pmf2)
    for k in range(0, K, 4):
        t = ts + k
        items2.append(build_streams(A, R, t, L, C, N, H)); meta2.append((k, A, R, t, np.cumsum(pmf2), y))
preds2 = predict_recast(model, items2)
for (qn, qf, _), (k, A, R, t, F2, y) in zip(preds2, meta2):
    pos = np.arange(t - N + 1, t + 1)
    ycl, _, _ = chain_ladder_nowcast(A, R, t, L, N)
    sc = np.nanmean(np.abs(np.diff(A[max(0, t - 2 * C): t + 1, L]))) + 1e-6
    for j, u in enumerate(pos):
        a = t - u; x_a = A[u, a]; truth = A[u, L]
        if not np.isfinite(x_a):
            continue
        err["recast"][k] += abs(qn[j, 4] - truth) / sc; err["cl"][k] += abs(ycl[j, 4] - truth) / sc
        err["oracle_cl"][k] += abs(x_a / max(F2[a], 1e-6) - truth) / sc; err["naive"][k] += abs(x_a - truth) / sc; cnt[k] += 1
res["M2"] = {k: [float(v[i] / cnt[i]) if cnt[i] > 0 else None for i in range(K)] for k, v in err.items()}
print("M2 (steps since switch: 0, 8, 16, 32, 60):", {k: [round(v[i], 3) for i in [0, 8, 16, 32, 60] if v[i] is not None] for k, v in res["M2"].items()})
json.dump(res, open(f"{args.out}/mech_{os.path.basename(args.model).replace('.pt','')}.json", "w"), indent=1)
print("saved")
