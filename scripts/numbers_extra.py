import sys, os, json, argparse, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--agg", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--timemoe", default=None, help="agg json:tag for Time-MoE")
ap.add_argument("--bayes", default=None)
ap.add_argument("--bb", default=None, help="comma-separated arch:agg:tag for the backbone summary macros")
ap.add_argument("--dataeff", default=None, help="prefix of the data-efficiency runs (e.g. v12 -> v12_C96__dv.json)")
args = ap.parse_args()
ORDER = ["kit", "dv", "chng_flu", "hosp_cov", "respinow", "nssp", "nhsn", "macro_m", "rtdsm_q", "alfred_w", "eia930"]
KEY = {"kit": "kit", "dv": "dv", "chng_flu": "flu", "hosp_cov": "hosp", "respinow": "resp", "nssp": "nssp", "nhsn": "nhsn", "macro_m": "macroM", "rtdsm_q": "macroQ", "alfred_w": "ui", "eia930": "grid"}
A = json.load(open(args.agg)); m = {}
gaps = []
for d in ORDER:
    if d not in A:
        continue
    nv = A[d]["plain_naive"]["forecast"]["CRPS_s"]; orc = A[d]["plain_oracle"]["forecast"]["CRPS_s"]
    gaps.append(100 * (nv - orc) / nv); m[f"rtgap{KEY[d]}"] = f"{gaps[-1]:.0f}"
    m[f"fc{KEY[d]}Rel"] = f"{abs(100 * (A[d][args.tag]['forecast']['CRPS_s'] / nv - 1)):.1f}"; m[f"fc{KEY[d]}RelSigned"] = f"{100 * (A[d][args.tag]['forecast']['CRPS_s'] / nv - 1):+.1f}"

    for nm, lab in [(f"{args.tag.replace('_final', '')}_preonly", "Pre"), (f"{args.tag.replace('_final', '')}_anchoronly", "Anc")]:
        if nm in A[d]:
            m[f"fc{KEY[d]}{lab}"] = f"{A[d][nm]['forecast']['CRPS_s']:.3f}"; m[f"cov{KEY[d]}{lab}"] = f"{A[d][nm]['forecast']['COV80']:.2f}"

    if "b2f" in A[d]:
        m[f"fc{KEY[d]}BTFrel"] = f"{100 * (A[d]['b2f']['forecast']['CRPS_s'] / nv - 1):+.0f}"

m["realTimeGapMax"] = f"{max(gaps):.0f}"; m["realTimeGapMean"] = f"{np.mean(gaps):.0f}"

tot = 0.0; n = 0
for d in ORDER:
    p_ = f"results/rrbench/{args.tag}__{d}.json"
    if os.path.exists(p_):
        r = json.load(open(p_)); tot += r["seconds"]; n += r["n"]
m["msPerOrigin"] = f"{1000 * tot / n:.1f}" if n else "--"
if args.timemoe:
    p, tag = args.timemoe.split(":"); T = json.load(open(p)); w = 0; n = 0; gm = []
    for d in ORDER:
        if d in T and tag in T[d]:
            nv = T[d]["plain_naive"]["forecast"]["MASE"]; v = T[d][tag]["forecast"]["MASE"]; orc = T[d]["plain_oracle"]["forecast"]["MASE"]
            n += 1; w += int(v < nv)
            if (nv - orc) / nv >= 0.02:
                gm.append(100 * (nv - v) / (nv - orc))
    m["tmMASEwins"] = str(w); m["tmMASEgap"] = f"{np.mean(gm):.0f}"
    covs = [T[d][tag]["forecast"]["COV80"] for d in ORDER if d in T and tag in T[d]]
    m["tmCovMin"] = f"{min(covs):.2f}"; m["tmCovMax"] = f"{max(covs):.2f}"
if args.bayes and os.path.exists(args.bayes):
    Bj = json.load(open(args.bayes))
    for d in Bj:
        if "bayes" not in Bj[d]:
            continue
        r = Bj[d]; k = KEY[d]; nv = r["plain_naive"]["forecast"]["CRPS_s"]; orc = r["plain_oracle"]["forecast"]["CRPS_s"]; nvn = r["naive_now"]["MASE"]
        m[f"now{k}Bayes"] = f"{r['bayes']['nowcast']['MASE']:.3f}"; m[f"now{k}BayesCRPS"] = f"{r['bayes']['nowcast']['CRPS_s']:.3f}"; m[f"now{k}BayesCov"] = f"{r['bayes']['nowcast']['COV80']:.2f}"
        m[f"now{k}RecastCRPS"] = f"{r[args.tag]['nowcast']['CRPS_s']:.3f}"; m[f"now{k}RecastCov"] = f"{r[args.tag]['nowcast']['COV80']:.2f}"; m[f"now{k}CLCRPS"] = f"{r['cl']['nowcast']['CRPS_s']:.3f}"; m[f"now{k}CLCov"] = f"{r['cl']['nowcast']['COV80']:.2f}"
        m[f"nowImpr{k}Bayes"] = f"{100 * (1 - r['bayes']['nowcast']['MASE'] / nvn):.0f}"; m[f"nowImpr{k}Recast"] = f"{100 * (1 - r[args.tag]['nowcast']['MASE'] / nvn):.0f}"; m[f"nowImpr{k}CL"] = f"{100 * (1 - r['cl']['nowcast']['MASE'] / nvn):.0f}"
        m[f"now{k}RecastB"] = f"{r[args.tag]['nowcast']['MASE']:.3f}"; m[f"now{k}NaiveB"] = f"{nvn:.3f}"; m[f"now{k}CLB"] = f"{r['cl']['nowcast']['MASE']:.3f}"
        m[f"fc{k}RecastB"] = f"{r[args.tag]['forecast']['CRPS_s']:.3f}"; m[f"fc{k}NaiveB"] = f"{nv:.3f}"
        for nm, lab in [("bayes_twostage", "BayesTwoS"), ("bayes_twostage_mc", "BayesTwoSMC"), ("twostage_mc", "TwoSMC")]:
            if nm in r:
                v = r[nm]["forecast"]["CRPS_s"]; m[f"fc{k}{lab}"] = f"{v:.3f}"
                if (nv - orc) / nv >= 0.02:
                    m[f"gap{k}{lab}"] = f"{100 * (nv - v) / (nv - orc):.0f}"
        if (nv - orc) / nv >= 0.02:
            m[f"gap{k}RecastB"] = f"{100 * (nv - r[args.tag]['forecast']['CRPS_s']) / (nv - orc):.0f}"
        m[f"nbayes{k}"] = str(r["n_common"])
    nf = sum(Bj[d]["n_common"] for d in Bj if "bayes" in Bj[d]); m["NbayesFits"] = f"{nf:,}".replace(",", "{,}")
    ndiv = 0; ntot = 0; nfb = 0
    for d in Bj:
        pj = f"results/rrbench/bayes__{d}.json"
        if "bayes" in Bj[d] and os.path.exists(pj):
            r = json.load(open(pj)); ndiv += r.get("n_divergent", 0) + r.get("n_error", 0); nfb += r.get("n_fallback", 0); ntot += r["n"]
            m[f"bayesFallback{KEY[d]}"] = f"{100 * r.get('n_fallback', 0) / r['n']:.0f}"
    if ntot:
        m["bayesDivergentPct"] = f"{100 * ndiv / ntot:.1f}"; m["bayesFallbackPct"] = f"{100 * nfb / ntot:.0f}"
if args.dataeff:
    R_ = "results/rrbench"
    CN = {96: "Short", 128: "Mid", 192: "Long", 256: "Full"}
    for d in ["dv", "kit"]:
        for C in [96, 128, 192]:
            fv = f"{R_}/{args.dataeff}_C{C}__{d}.json"; fc = f"{R_}/baselines_C{C}__{d}.json"
            if os.path.exists(fv):
                v = json.load(open(fv)); m[f"dataeffVnow{KEY[d]}{CN[C]}"] = f"{v['nowcast']['MASE']:.3f}"; m[f"dataeffVfc{KEY[d]}{CN[C]}"] = f"{v['forecast']['MASE']:.3f}"
            if os.path.exists(fc):
                c = json.load(open(fc)); m[f"dataeffCLnow{KEY[d]}{CN[C]}"] = f"{c['nowcast_cl']['MASE']:.3f}"; m[f"dataeffCLfc{KEY[d]}{CN[C]}"] = f"{c['forecast_2s_point']['MASE']:.3f}"
        fc = f"{R_}/baselines__{d}.json"
        if os.path.exists(fc):
            c = json.load(open(fc)); m[f"dataeffCLnow{KEY[d]}Full"] = f"{c['nowcast_cl']['MASE']:.3f}"; m[f"dataeffCLfc{KEY[d]}Full"] = f"{c['forecast_2s_point']['MASE']:.3f}"
        m[f"dataeffVnow{KEY[d]}"] = m.get(f"dataeffVnow{KEY[d]}Short", "--"); m[f"dataeffVfc{KEY[d]}"] = m.get(f"dataeffVfc{KEY[d]}Short", "--")
        m[f"dataeffCLnow{KEY[d]}"] = m.get(f"dataeffCLnow{KEY[d]}Short", "--"); m[f"dataeffCLfc{KEY[d]}"] = m.get(f"dataeffCLfc{KEY[d]}Short", "--")
if args.bb:
    PRETTYK = {"chronos2": "cTwo", "bolt_s": "boltS", "bolt_b": "boltB", "toto": "toto", "timemoe": "tm"}
    for item in args.bb.split(","):
        arch, path, tag = item.split(":"); B = json.load(open(path)); g2, gv, wn, w2, n, nh, nc = [], [], 0, 0, 0, [], []
        for d in ORDER:
            if d not in B or tag not in B[d]:
                continue
            nv = B[d]["plain_naive"]["forecast"]["CRPS_s"]; orc = B[d]["plain_oracle"]["forecast"]["CRPS_s"]; v = B[d][tag]["forecast"]["CRPS_s"]; t2 = B[d].get("twostage", {}).get("forecast", {}).get("CRPS_s", np.nan)
            n += 1; wn += int(v < nv); w2 += int(np.isfinite(t2) and v < t2)
            if (nv - orc) / nv >= 0.02:
                gv.append(100 * (nv - v) / (nv - orc)); g2.append(100 * (nv - t2) / (nv - orc) if np.isfinite(t2) else np.nan)
            nvn = B[d]["naive_now"]["MASE"]; nh.append(100 * (1 - B[d][tag]["nowcast"]["MASE"] / nvn)); nc.append(100 * (1 - B[d]["cl"]["nowcast"]["MASE"] / nvn))
        k = PRETTYK[arch]
        m[f"bb{k}Gap"] = f"{np.mean(gv):.0f}"; m[f"bb{k}GapTwoS"] = f"{np.nanmean(g2):.0f}" if np.isfinite(g2).any() else "--"; m[f"bb{k}WinsNaive"] = str(wn); m[f"bb{k}WinsTwoS"] = str(w2)
        m[f"bb{k}Now"] = f"{np.mean(nh):.0f}"; m[f"bb{k}NowCL"] = f"{np.mean(nc):.0f}"
        for d in ["kit", "dv", "eia930"]:
            if d in B and tag in B[d]:
                m[f"bb{k}Now{KEY[d]}"] = f"{B[d][tag]['nowcast']['MASE']:.3f}"; m[f"bb{k}NowCL{KEY[d]}"] = f"{B[d]['cl']['nowcast']['MASE']:.3f}"
with open(args.out, "a") as f:
    for k, v in m.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("appended", len(m), "macros to", args.out)
