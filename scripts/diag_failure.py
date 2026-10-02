import sys, os, json, argparse
sys.path.insert(0, ".")
import numpy as np, torch
from recast.datasets import load_bundle
from recast.protocol import make_tasks, metrics, DEFAULTS, per_origin_loss
from recast.views import build_streams, as_of_series
from recast.synth import nowcast_window, model_nowcast_window
from recast.registry import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--domain", required=True); ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--arch", default="chronos2"); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--max_origins", type=int, default=None)
ap.add_argument("--out", default=None)
args = ap.parse_args()
R = "results/rrbench"
b = load_bundle(args.domain); freq, L = b["freq"], b["L"]
d = DEFAULTS[freq]; C, H = d["C"], d["H"]
N = max(nowcast_window(freq, L), model_nowcast_window(freq, L))
tasks = make_tasks(b, max_origins=args.max_origins)
dev = "cuda"
model = build_model(args.arch, evidence_gate=True, separate_now_head=True, forecast_anchor=True, mult_head=True).to(dev).eval()
sd = torch.load(args.model, map_location=dev); model.load_state_dict(sd.get("model", sd), strict=False)
tr = getattr(model, "target_row_default", 1)


zv = dict(np.load(f"{R}/{args.tag}__{args.domain}.npz", allow_pickle=True)); zn = dict(np.load(f"{R}/{args.arch}_naive__{args.domain}.npz", allow_pickle=True))
kv = {(str(s), int(t)): i for i, (s, t) in enumerate(zip(zv["index_sid"], zv["index_t"]))}
kn = {(str(s), int(t)): i for i, (s, t) in enumerate(zip(zn["index_sid"], zn["index_t"]))}

rows = []
with torch.no_grad():
    for tk in tasks:
        items, meta = [], []
        for t in tk.origins:
            t = int(t)
            st = build_streams(tk.A, tk.R, t, L, C, N, H, with_target=False)
            x = as_of_series(tk.A, tk.R, t, L, C)
            n_settled = int(np.isfinite(st["vals"][0]).sum()); n_ctx = int(np.isfinite(x).sum())
            rev_edge = float(st["rev"][1][C - 1]) if st["mask"][1][C - 1] > 0 else float("nan")
            rev_win = float(np.nanmean(np.where(st["mask"][1][C - N:C] > 0, st["rev"][1][C - N:C], np.nan)))
            items.append(st); meta.append(dict(t=t, n_settled=n_settled, n_ctx=n_ctx, rev_edge=rev_edge, rev_win=rev_win))
        for s in range(0, len(items), args.bs):
            its = items[s: s + args.bs]; B = len(its)
            vals = torch.tensor(np.stack([it["vals"] for it in its])).to(dev); mask = torch.tensor(np.stack([it["mask"] for it in its])).to(dev)
            age = torch.tensor(np.stack([it["age_raw"] for it in its])).to(dev); rev = torch.tensor(np.stack([it["rev"] for it in its])).to(dev)
            sage = torch.tensor(np.stack([it["stream_age"] for it in its])).to(dev); Lt = torch.full((B,), L, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(vals, mask, age, sage, Lt, ctx_len=C, N=N, target_row=tr, rev=rev)
            g_now = out["gate"].float().mean(dim=(1, 2)).cpu().numpy(); g_fc = out["gate_fc"].float().mean(dim=(1, 2)).cpu().numpy()

            qn = out["quantiles"][:, model.median_idx, :N].float().cpu().numpy()
            xr = vals[:, 1, C - N:C].float().cpu().numpy()
            corr = np.nanmean(np.abs(qn - xr) / (np.abs(xr) + 1e-9), axis=1)
            for j in range(B):
                m = meta[s + j]; key = (tk.sid, m["t"])
                if key not in kv or key not in kn:
                    continue
                iv, inn = kv[key], kn[key]
                if not zv["EV"][iv]:
                    continue
                Y = zv["Y"][iv][None]; S = zv["S"][iv][None]
                lv = per_origin_loss(zv["Qf"][iv][None], Y, S)[0]; ln = per_origin_loss(zn["Qf"][inn][None], Y, S)[0]
                wv = float(np.nanmean(zv["Qf"][iv][:, 8] - zv["Qf"][iv][:, 0]) / S[0]); wn = float(np.nanmean(zn["Qf"][inn][:, 8] - zn["Qf"][inn][:, 0]) / S[0])
                cv = float(np.nanmean((Y[0] >= zv["Qf"][iv][:, 0]) & (Y[0] <= zv["Qf"][iv][:, 8]))); cn = float(np.nanmean((Y[0] >= zn["Qf"][inn][:, 0]) & (Y[0] <= zn["Qf"][inn][:, 8])))
                rows.append(dict(sid=tk.sid, **m, gate_now=float(g_now[j]), gate_fc=float(g_fc[j]), corr=float(corr[j]), crps_recast=float(lv), crps_naive=float(ln), width_recast=wv, width_naive=wn, cov_recast=cv, cov_naive=cn))

import collections
by = collections.defaultdict(list)
for r in rows:
    by[r["sid"]].append(r)
summ = []
for sid, rs in by.items():
    f = lambda k: float(np.nanmean([r[k] for r in rs]))
    summ.append(dict(sid=sid, n=len(rs), crps_naive=f("crps_naive"), crps_recast=f("crps_recast"), rel=100 * (f("crps_recast") / f("crps_naive") - 1), gate_now=f("gate_now"), gate_fc=f("gate_fc"),
                     corr=f("corr"), width_ratio=f("width_recast") / f("width_naive"), cov_naive=f("cov_naive"), cov_recast=f("cov_recast"), n_settled=f("n_settled"), rev_edge=f("rev_edge"), n_ctx=f("n_ctx")))
summ.sort(key=lambda r: -r["rel"])
print(f"{'series':40s} {'n':>5s} {'naive':>7s} {'RECAST':>7s} {'rel%':>6s} {'g_now':>6s} {'g_fc':>6s} {'corr%':>6s} {'width':>6s} {'covN':>5s} {'covV':>5s} {'settl':>5s} {'rev':>5s}")
for r in summ:
    print(f"{r['sid'][:40]:40s} {r['n']:5d} {r['crps_naive']:7.3f} {r['crps_recast']:7.3f} {r['rel']:+6.1f} {r['gate_now']:6.3f} {r['gate_fc']:6.3f} {100*r['corr']:6.2f} {r['width_ratio']:6.2f} {r['cov_naive']:5.2f} {r['cov_recast']:5.2f} {r['n_settled']:5.0f} {r['rev_edge']:5.2f}")
allr = dict(n=len(rows), crps_naive=float(np.mean([r["crps_naive"] for r in rows])), crps_recast=float(np.mean([r["crps_recast"] for r in rows])),
            gate_now=float(np.mean([r["gate_now"] for r in rows])), gate_fc=float(np.mean([r["gate_fc"] for r in rows])), width_ratio=float(np.mean([r["width_recast"] for r in rows]) / np.mean([r["width_naive"] for r in rows])),
            frac_origins_worse=float(np.mean([r["crps_recast"] > r["crps_naive"] for r in rows])))

dl = np.array([r["crps_recast"] - r["crps_naive"] for r in rows]); gf = np.array([r["gate_fc"] for r in rows]); gn = np.array([r["gate_now"] for r in rows])
allr["corr_dcrps_gate_fc"] = float(np.corrcoef(dl, gf)[0, 1]); allr["corr_dcrps_gate_now"] = float(np.corrcoef(dl, gn)[0, 1])

order = np.argsort(-dl); k = max(1, int(0.05 * len(dl)))
allr["share_worst5pct"] = float(dl[order[:k]].sum() / dl.sum()) if dl.sum() > 0 else float("nan")
allr["median_dcrps"] = float(np.median(dl))
print(json.dumps(allr, indent=1))
if args.out:
    json.dump(dict(domain=args.domain, tag=args.tag, overall=allr, series=summ, origins=rows), open(args.out, "w"), indent=1)
    print("saved", args.out)
