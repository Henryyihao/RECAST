import sys, json, argparse, os, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--spec", required=True)
ap.add_argument("--out", default="results/tables/tab_backbones.tex")
args = ap.parse_args()
PRETTY = {"chronos2": "Chronos-2 (120M)", "bolt_s": "Chronos-Bolt-S (48M)", "bolt_b": "Chronos-Bolt-B (205M)", "toto": "Toto-2 (313M)", "timemoe": "Time-MoE (50M)"}
ORDER = ["kit", "dv", "chng_flu", "hosp_cov", "respinow", "nssp", "nhsn", "macro_m", "rtdsm_q", "alfred_w", "eia930"]
rows = []
detail = {}
for item in args.spec.split(","):
    arch, path, tag = item.split(":")
    A = json.load(open(path))
    rel = {"twostage": [], tag: [], "plain_oracle": []}; gap = {"twostage": [], tag: []}; now = {"cl": [], tag: []}
    wins_naive = wins_rc = n = 0
    for d in ORDER:
        if d not in A or "plain_naive" not in A[d]:
            continue
        nv = A[d]["plain_naive"]["forecast"]["CRPS_s"]; orc = A[d].get("plain_oracle", {}).get("forecast", {}).get("CRPS_s", np.nan)
        for k in rel:
            v = A[d].get(k, {}).get("forecast", {}).get("CRPS_s", np.nan)
            rel[k].append(v / nv)
            if k in gap and np.isfinite(orc) and (nv - orc) / nv >= 0.02:
                gap[k].append(100 * (nv - v) / (nv - orc))
        nvn = A[d]["naive_now"]["MASE"]
        for k in now:
            v = A[d].get(k, {}).get("nowcast", {}).get("MASE", np.nan); now[k].append(100 * (1 - v / nvn))
        ve = A[d].get(tag, {}).get("forecast", {}).get("CRPS_s", np.nan); rc = A[d].get("twostage", {}).get("forecast", {}).get("CRPS_s", np.nan)
        n += 1; wins_naive += int(ve < nv); wins_rc += int(np.isfinite(rc) and ve < rc)
        detail.setdefault(arch, {})[d] = dict(naive=nv, twostage=rc, recast=ve, oracle=orc)
    f = lambda x: f"{np.nanmean(x):.3f}" if len(x) and np.isfinite(x).any() else "--"
    g = lambda x: f"{np.nanmean(x):.0f}" if len(x) and np.isfinite(x).any() else "--"
    n_rc = int(np.isfinite(rel["twostage"]).sum())
    rc_rel = f(rel["twostage"]) if n_rc >= 6 else "--"; rc_gap = g(gap["twostage"]) if n_rc >= 6 else "--"; rc_wins = f"{wins_rc}/{n}" if n_rc >= 6 else "--"
    rows.append(f"{PRETTY.get(arch, arch)} & {rc_rel} & {f(rel[tag])} & {f(rel['plain_oracle'])} & {rc_gap} & {g(gap[tag])} & {wins_naive}/{n} & {rc_wins} & {g(now['cl'])} & {g(now[tag])} \\\\")
hdr = ["\\begin{tabular*}{\\tblwidth}{@{}lrrrrrrrrr@{}}\n\\toprule",
       " & \\multicolumn{3}{c}{CRPS relative to \\naive{}} & \\multicolumn{2}{c}{gap closed (\\%)} & \\multicolumn{2}{c}{\\recast{} wins} & \\multicolumn{2}{c}{nowcast impr. (\\%)} \\\\",
       "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
       "Backbone & CL$\\to$TSFM & \\recast{} & \\oracle{} & CL$\\to$TSFM & \\recast{} & vs \\naive{} & vs CL$\\to$TSFM & CL & \\recast{} \\\\\n\\midrule"]
open(args.out, "w").write("\n".join(hdr + rows + ["\\bottomrule\n\\end{tabular*}"]))
json.dump(detail, open(args.out.replace(".tex", ".json"), "w"), indent=1)
print("\n".join(rows))
