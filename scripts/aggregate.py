import sys, os, json, argparse
sys.path.insert(0, ".")
import numpy as np
from scipy import stats
from recast.protocol import metrics, per_origin_loss

R = "results/rrbench"
FPD = "data/outputside"
DOMAINS = ["kit", "dv", "chng_flu", "hosp_cov", "respinow", "nssp", "nhsn", "macro_m", "rtdsm_q", "alfred_w", "eia930"]
FPD_ARCH = {"chronos2": "chronos2", "bolt_s": "chronos_bolt", "bolt_b": "bolt_b", "toto": "toto", "timemoe": "timemoe"}

ap = argparse.ArgumentParser()
ap.add_argument("--tags", required=True, help="comma-separated RECAST result tags")
ap.add_argument("--arch", default="chronos2")
ap.add_argument("--out", default=None)
ap.add_argument("--domains", default=",".join(DOMAINS))
ap.add_argument("--min_scale_frac", type=float, default=0.0, help="drop cells whose protocol scale S is below this fraction of the domain median (0 = full protocol)")
ap.add_argument("--no_bayes", action="store_true", help="ignore the Bayesian nowcaster results even where available")
ap.add_argument("--no_fpd", action="store_true", help="ignore the output-side baselines (B2F, residual adapter)")
ap.add_argument("--bayes_subset", action="store_true", help="allow Bayesian results on a subset of the origins (all methods are then compared on that subset)")
args = ap.parse_args()
tags = args.tags.split(",")
doms = args.domains.split(",")


def load_npz(path):
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def key_index(d):
    return {(str(s), int(t)): i for i, (s, t) in enumerate(zip(d["index_sid"], d["index_t"]))}


def nowcast_mase_naive(XN, YN, S):
    ae = np.abs(XN - YN) / S[:, None]
    return float(np.nanmean(ae))


out = {}
for dom in doms:
    ref = load_npz(f"{R}/{tags[0]}__{dom}.npz")
    if ref is None:
        print("missing", tags[0], dom); continue
    kref = key_index(ref)
    keys = list(kref.keys())

    srcs = {}
    for tag in tags:
        d = load_npz(f"{R}/{tag}__{dom}.npz")
        if d is not None:
            srcs[tag] = (d, key_index(d), {"fc": "Qf", "now": "Qn"})
    for mode in ["naive", "oracle"]:
        d = load_npz(f"{R}/{args.arch}_{mode}__{dom}.npz")
        if d is not None:
            srcs[f"plain_{mode}"] = (d, key_index(d), {"fc": "Qf"})
    d = load_npz(f"{R}/baselines__{dom}.npz")
    if d is not None:
        srcs["cl"] = (d, key_index(d), {"now": "Qn_cl"})
        if args.arch == "chronos2":
            srcs["twostage"] = (d, key_index(d), {"fc": "Qf_2s"}); srcs["twostage_mc"] = (d, key_index(d), {"fc": "Qf_mc"})
    if args.arch != "chronos2":
        d = load_npz(f"{R}/baselines_{args.arch}__{dom}.npz")
        if d is not None:
            srcs["twostage"] = (d, key_index(d), {"fc": "Qf_2s"}); srcs["twostage_mc"] = (d, key_index(d), {"fc": "Qf_mc"})
    fa = FPD_ARCH.get(args.arch)
    d = load_npz(f"{FPD}/{fa}__{dom}.npz") if (fa and not args.no_fpd) else None
    if d is not None:
        kf = key_index(d)
        for k, nm in [("Q_b2f", "b2f"), ("Q_ra_prov", "resid_adapter")]:
            if k in d:
                srcs[nm] = (d, kf, {"fc": k})
    d = None if args.no_bayes else load_npz(f"{R}/bayes__{dom}.npz")
    if d is not None:
        kb = key_index(d)
        if args.bayes_subset or len(set(kb) & set(keys)) >= 0.95 * len(keys):
            srcs["bayes"] = (d, kb, {"now": "Qn_bayes"}); srcs["bayes_twostage"] = (d, kb, {"fc": "Qf_2s"}); srcs["bayes_twostage_mc"] = (d, kb, {"fc": "Qf_mc"})
        else:
            print(f"  bayes results for {dom} cover only {len(set(kb) & set(keys))}/{len(keys)} origins: skipped")

    common = set(keys)
    for nm, (d, kd, m) in srcs.items():
        common &= set(kd.keys())
    common = sorted(common, key=lambda k: kref[k])
    idx_ref = np.array([kref[k] for k in common])
    EV = ref["EV"][idx_ref].copy()
    Y = ref["Y"][idx_ref]; S = ref["S"][idx_ref]; YN = ref["YN"][idx_ref]; XN = ref["XN"][idx_ref]
    if args.min_scale_frac > 0:
        thr = args.min_scale_frac * np.median(S[EV]); EV = EV & (S >= thr)
    res = {"n_common": int(len(common)), "n_eval": int(EV.sum()), "L": None}
    per_origin = {}
    for nm, (d, kd, m) in srcs.items():
        ii = np.array([kd[k] for k in common])
        r = {}
        if "fc" in m:
            Q = d[m["fc"]][ii]
            mt = metrics(Q[EV], Y[EV], S[EV])
            r["forecast"] = {k: v for k, v in mt.items() if not isinstance(v, list)}
            r["forecast_h"] = {k: mt[k] for k in ["MASE_h", "CRPS_h", "COV80_h"]}
            per_origin[nm] = per_origin_loss(Q[EV], Y[EV], S[EV])
        if "now" in m:
            Q = d[m["now"]][ii]
            mt = metrics(Q[EV], YN[EV], S[EV])
            r["nowcast"] = {k: v for k, v in mt.items() if not isinstance(v, list)}
            r["nowcast_h"] = {k: mt[k] for k in ["MASE_h", "CRPS_h", "COV80_h"]}
            per_origin[nm + "@now"] = per_origin_loss(Q[EV], YN[EV], S[EV])
        res[nm] = r
    res["naive_now"] = {"MASE": nowcast_mase_naive(XN[EV], YN[EV], S[EV])}

    tests = {}
    if tags[0] in per_origin:
        a = per_origin[tags[0]]
        for nm in ["plain_naive", "twostage", "twostage_mc", "b2f", "resid_adapter", "bayes_twostage", "bayes_twostage_mc", "plain_oracle"] + tags[1:]:
            if nm in per_origin:
                b = per_origin[nm]; ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() > 10:
                    t, p = stats.ttest_rel(a[ok], b[ok]); w = stats.wilcoxon(a[ok], b[ok]).pvalue if ok.sum() < 5000 else np.nan
                    tests[nm] = {"t": float(t), "p": float(p), "wilcoxon_p": float(w), "mean_diff": float(np.mean(a[ok] - b[ok])), "n": int(ok.sum())}
        for bn in ["cl", "bayes"]:
            if f"{bn}@now" in per_origin and f"{tags[0]}@now" in per_origin:
                a = per_origin[f"{tags[0]}@now"]; b = per_origin[f"{bn}@now"]; ok = np.isfinite(a) & np.isfinite(b)
                t, p = stats.ttest_rel(a[ok], b[ok]); tests[f"{bn}@now"] = {"t": float(t), "p": float(p), "mean_diff": float(np.mean(a[ok] - b[ok])), "n": int(ok.sum())}
    res["tests"] = tests
    out[dom] = res
    fc = lambda nm, k="MASE": res.get(nm, {}).get("forecast", {}).get(k, float("nan"))
    nw = lambda nm, k="MASE": res.get(nm, {}).get("nowcast", {}).get(k, float("nan"))
    print(f"{dom:9s} n={len(common):5d} | fc MASE naive {fc('plain_naive'):.3f} 2s {fc('twostage'):.3f} RECAST {fc(tags[0]):.3f} oracle {fc('plain_oracle'):.3f} | CRPS naive {fc('plain_naive','CRPS_s'):.3f} 2s {fc('twostage','CRPS_s'):.3f} bayes2s {fc('bayes_twostage','CRPS_s'):.3f} RECAST {fc(tags[0],'CRPS_s'):.3f} | now naive {res['naive_now']['MASE']:.3f} cl {nw('cl'):.3f} bayes {nw('bayes'):.3f} RECAST {nw(tags[0]):.3f}")
if args.out:
    json.dump(out, open(args.out, "w"), indent=1)
    print("saved", args.out)
