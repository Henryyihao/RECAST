import sys, json, argparse, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--agg", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--mech", default=None)
args = ap.parse_args()
A = json.load(open(args.agg))
ORDER = ["kit", "dv", "chng_flu", "hosp_cov", "respinow", "nssp", "nhsn", "macro_m", "rtdsm_q", "alfred_w", "eia930"]
HEAVY = ["kit", "dv", "chng_flu", "hosp_cov", "eia930"]
def num(x, d=3):
    return f"{x:.{d}f}"
m = {}
gaps, gaps_rc, gaps_2s, wins_naive, wins_rc, wins_2s, n = [], [], [], 0, 0, 0, 0
rel, rel_rc, rel_2s, gap_doms = [], [], [], []
now_impr, now_impr_cl, now_wins_cl = [], [], 0
zs_rel, zs_wins, zs_loss_max = [], 0, 0.0
for d in ORDER:
    if d not in A: continue
    r = A[d]; nv = r["plain_naive"]["forecast"]["CRPS_s"]; orc = r["plain_oracle"]["forecast"]["CRPS_s"]; ve = r[args.tag]["forecast"]["CRPS_s"]
    rc = r.get("b2f", {}).get("forecast", {}).get("CRPS_s", np.nan); ts = r.get("twostage", {}).get("forecast", {}).get("CRPS_s", np.nan)
    n += 1; wins_naive += ve < nv; wins_rc += np.isfinite(rc) and ve < rc; wins_2s += np.isfinite(ts) and ve < ts
    rel.append(ve / nv); rel_rc.append(rc / nv if np.isfinite(rc) else np.nan); rel_2s.append(ts / nv if np.isfinite(ts) else np.nan)
    if (nv - orc) / nv >= 0.02:
        gaps.append(100 * (nv - ve) / (nv - orc)); gaps_rc.append(100 * (nv - rc) / (nv - orc) if np.isfinite(rc) else np.nan); gaps_2s.append(100 * (nv - ts) / (nv - orc) if np.isfinite(ts) else np.nan)
        gap_doms.append(d)
    nvn = r["naive_now"]["MASE"]; vn = r[args.tag]["nowcast"]["MASE"]; cln = r.get("cl", {}).get("nowcast", {}).get("MASE", np.nan)
    now_impr.append(100 * (1 - vn / nvn)); now_impr_cl.append(100 * (1 - cln / nvn) if np.isfinite(cln) else np.nan); now_wins_cl += np.isfinite(cln) and vn < cln
    key = {"kit": "kit", "dv": "dv", "chng_flu": "flu", "hosp_cov": "hosp", "respinow": "resp", "nssp": "nssp", "nhsn": "nhsn", "macro_m": "macroM", "rtdsm_q": "macroQ", "alfred_w": "ui", "eia930": "grid"}[d]
    m[f"fc{key}Naive"] = num(nv); m[f"fc{key}Recast"] = num(ve); m[f"fc{key}Oracle"] = num(orc); m[f"fc{key}BTF"] = num(rc) if np.isfinite(rc) else "--"; m[f"fc{key}TwoS"] = num(ts) if np.isfinite(ts) else "--"

    m[f"fcMASE{key}Naive"] = num(r["plain_naive"]["forecast"]["MASE"]); m[f"fcMASE{key}Recast"] = num(r[args.tag]["forecast"]["MASE"]); m[f"fcMASE{key}Oracle"] = num(r["plain_oracle"]["forecast"]["MASE"])
    m[f"fcMASE{key}BTF"] = num(r["b2f"]["forecast"]["MASE"]) if "b2f" in r else "--"; m[f"fcMASE{key}TwoS"] = num(r["twostage"]["forecast"]["MASE"]) if "twostage" in r else "--"
    m[f"now{key}Naive"] = num(nvn); m[f"now{key}Recast"] = num(vn); m[f"now{key}CL"] = num(cln) if np.isfinite(cln) else "--"
    m[f"cov{key}Recast"] = num(r[args.tag]["forecast"]["COV80"], 2); m[f"cov{key}Naive"] = num(r["plain_naive"]["forecast"]["COV80"], 2)
    m[f"gap{key}"] = f"{gaps[-1]:.0f}" if (nv - orc) / nv >= 0.02 else "--"
    zs = r.get("chronos2_zsviews_psn", {}).get("forecast", {}).get("CRPS_s", np.nan)
    m[f"fc{key}ZS"] = num(zs) if np.isfinite(zs) else "--"; m[f"fc{key}ZSrel"] = f"{100*(zs/nv-1):+.0f}" if np.isfinite(zs) else "--"; m[f"fc{key}ZSabs"] = f"{abs(100*(zs/nv-1)):.0f}" if np.isfinite(zs) else "--"
    zs_rel.append(zs / nv if np.isfinite(zs) else np.nan); zs_wins += int(np.isfinite(zs) and zs < nv); zs_loss_max = max(zs_loss_max, 100*(zs/nv-1) if np.isfinite(zs) else 0)
m["gapMean"] = f"{np.nanmean(gaps):.0f}"; m["gapMeanBTF"] = f"{np.nanmean(gaps_rc):.0f}"; m["gapMeanTwoS"] = f"{np.nanmean(gaps_2s):.0f}"
m["gapMeanHeavy"] = f"{np.nanmean([g for g, d in zip(gaps, gap_doms) if d in HEAVY]):.0f}"
m["gapMeanHeavyBTF"] = f"{np.nanmean([g for g, d in zip(gaps_rc, gap_doms) if d in HEAVY]):.0f}"
m["gapMeanHeavyTwoS"] = f"{np.nanmean([g for g, d in zip(gaps_2s, gap_doms) if d in HEAVY]):.0f}"
m["nGapDoms"] = str(len(gap_doms))
m["zsWins"] = f"{zs_wins}/{n}"; m["zsLossMax"] = f"{zs_loss_max:.0f}"; m["relCRPSMeanZS"] = f"{100*(np.nanmean(zs_rel)-1):+.1f}"
m["relCRPSRed"] = f"{abs(100*(np.nanmean(rel)-1)):.1f}"; m["relCRPSRedTwoS"] = f"{abs(100*(np.nanmean(rel_2s)-1)):.1f}"; m["relCRPSIncBTF"] = f"{abs(100*(np.nanmean(rel_rc)-1)):.1f}"
m["nScoredPairs"] = f"{sum(A[d]['n_eval'] for d in A):,}".replace(",", "{,}"); m["nProcessedPairs"] = f"{sum(A[d]['n_common'] for d in A):,}".replace(",", "{,}")
m["relCRPSMean"] = f"{100*(np.nanmean(rel)-1):+.1f}"; m["relCRPSMeanBTF"] = f"{100*(np.nanmean(rel_rc)-1):+.1f}"; m["relCRPSMeanTwoS"] = f"{100*(np.nanmean(rel_2s)-1):+.1f}"
m["winsNaive"] = f"{wins_naive}/{n}"; m["winsBTF"] = f"{wins_rc}/{n}"; m["winsTwoS"] = f"{wins_2s}/{n}"
m["nowImprMean"] = f"{np.nanmean(now_impr):.0f}"; m["nowImprMeanCL"] = f"{np.nanmean(now_impr_cl):.0f}"; m["nowWinsCL"] = f"{now_wins_cl}/{n}"
m["nowImprHeavy"] = f"{np.nanmean([v for v, d in zip(now_impr, [d for d in ORDER if d in A]) if d in HEAVY]):.0f}"
if args.mech:
    M = json.load(open(args.mech))
    for k in ["recast", "cl", "oracle", "oracle_cl", "naive"]:
        m[f"mechMASE{k.replace('_','')}"] = num(M["M1"][k]["MASE"]); m[f"mechCRPS{k.replace('_','')}"] = num(M["M1"][k]["CRPS_s"]); m[f"mechCov{k.replace('_','')}"] = num(M["M1"][k]["COV80"], 2)
    m["probeRecast"] = num(M["M3"]["r2_recast"], 2); m["probeInit"] = num(M["M3"]["r2_init"], 2); m["probeRaw"] = num(M["M3"]["r2_raw_ratio"], 2)
    for i, nm in [(0, "zero"), (8, "eight"), (16, "sixteen"), (32, "thirtytwo"), (60, "sixty")]:
        for k in ["recast", "cl", "naive"]:
            v = M["M2"][k][i]
            if v is not None: m[f"switch{k}{nm}"] = num(v, 2)
with open(args.out, "w") as f:
    for k, v in m.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("wrote", len(m), "macros to", args.out)
