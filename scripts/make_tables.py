import sys, os, json, argparse, glob
import numpy as np
PRETTY = {"kit": "DE-Hosp (D)", "dv": "US-CLI (D)", "chng_flu": "US-Flu (D)", "hosp_cov": "US-Hosp (D)", "respinow": "DE-RESP (W)",
          "nssp": "US-NSSP (W)", "nhsn": "US-NHSN (W)", "macro_m": "US-Macro (M)", "rtdsm_q": "US-Macro (Q)", "alfred_w": "US-UI (W)", "eia930": "US-Grid (h)"}
ORDER = ["kit", "dv", "chng_flu", "hosp_cov", "respinow", "nssp", "nhsn", "macro_m", "rtdsm_q", "alfred_w", "eia930"]
ap = argparse.ArgumentParser()
ap.add_argument("--main", required=True)
ap.add_argument("--tag", required=True, help="RECAST tag inside the aggregated file")
ap.add_argument("--out", default="results/tables")
ap.add_argument("--name", default="main")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
agg = json.load(open(args.main))


def g(dom, nm, blk, k):
    try:
        return agg[dom][nm][blk][k]
    except KeyError:
        return float("nan")


def fmt(v, best=False, mark=""):
    if not np.isfinite(v):
        return "--"
    s = f"{v:.3f}"
    if best:
        s = "\\textbf{" + s + "}"
    return s + mark


def sig(dom, nm):
    t = agg[dom].get("tests", {}).get(nm)
    if not t:
        return ""
    p = t["p"]
    if p < 0.01:
        return "$^{\\dagger}$" if t["mean_diff"] < 0 else "$^{\\ddagger}$"
    if p < 0.05:
        return "$^{*}$" if t["mean_diff"] < 0 else "$^{\\circ}$"
    return ""


methods = [("plain_naive", "\\naive{}"), ("chronos2_zsviews_psn", "Views (0-shot)"), ("twostage", "CL$\\to$TSFM"), ("b2f", "B2F"), (args.tag, "\\recast{}"), ("plain_oracle", "\\oracle{}")]
methods = [m for m in methods if any(m[0] in agg[d] for d in agg)]
lines = []
lines.append("\\begin{tabular*}{\\tblwidth}{@{}ll" + "r" * len(methods) + "r@{}}\n\\toprule")
lines.append("Domain & metric & " + " & ".join(m[1] for m in methods) + " & gap closed \\\\\n\\midrule")
gaps = []
for dom in ORDER:
    if dom not in agg:
        continue
    for met in ["MASE", "CRPS_s"]:
        vals = [g(dom, nm, "forecast", met) for nm, _ in methods]
        feas = vals[:-1]
        best = np.nanargmin(feas) if np.isfinite(feas).any() else -1
        cells = []
        for i, (nm, _) in enumerate(methods):
            mark = sig(dom, nm) if nm in ("b2f", "plain_naive", "twostage", "chronos2_zsviews_psn") and met == "CRPS_s" else ""
            cells.append(fmt(vals[i], best=(i == best), mark=mark))
        gap = ""
        if met == "CRPS_s":
            nv, orc, ve = vals[0], vals[-1], vals[methods.index((args.tag, "\\recast{}"))]
            if np.isfinite(nv) and np.isfinite(orc) and (nv - orc) / nv >= 0.02:
                gc = 100 * (nv - ve) / (nv - orc); gaps.append(gc); gap = f"{gc:.0f}\\%"
            else:
                gap = "n/a"
        label = PRETTY.get(dom, dom) if met == "MASE" else ""
        lines.append(f"{label} & {'MASE' if met == 'MASE' else 'CRPS'} & " + " & ".join(cells) + f" & {gap} \\\\")
lines.append("\\bottomrule\n\\end{tabular*}")
open(f"{args.out}/tab_{args.name}.tex", "w").write("\n".join(lines))
print("gap closed (CRPS) per domain:", [round(x) for x in gaps], "mean", round(float(np.mean(gaps)), 1) if gaps else None)


lines = ["\\begin{tabular*}{\\tblwidth}{@{}lrrrrrrrr@{}}\n\\toprule",
         " & & \\multicolumn{3}{c}{MASE} & \\multicolumn{2}{c}{CRPS} & \\multicolumn{2}{c}{Cov80} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
         "Domain & $N$ & latest report & CL & \\recast{} & CL & \\recast{} & CL & \\recast{} \\\\\n\\midrule"]
for dom in ORDER:
    if dom not in agg:
        continue
    nv = agg[dom]["naive_now"]["MASE"]; cl = g(dom, "cl", "nowcast", "MASE"); ve = g(dom, args.tag, "nowcast", "MASE")
    clc = g(dom, "cl", "nowcast", "CRPS_s"); vec = g(dom, args.tag, "nowcast", "CRPS_s"); cov = g(dom, args.tag, "nowcast", "COV80"); covc = g(dom, "cl", "nowcast", "COV80")
    try:
        N = int(np.load(f"results/rrbench/{args.tag}__{dom}.npz")["YN"].shape[1])
    except Exception:
        N = None
    b = [nv, cl, ve]; best = int(np.nanargmin(b))
    cells = [fmt(v, best=(i == best)) for i, v in enumerate(b)]
    mark = sig(dom, "cl@now")
    lines.append(f"{PRETTY.get(dom, dom)} & {N if N is not None else ''} & " + " & ".join(cells) + f" & {fmt(clc)} & {fmt(vec, mark=mark)} & {covc:.2f} & {cov:.2f} \\\\")
lines.append("\\bottomrule\n\\end{tabular*}")
open(f"{args.out}/tab_{args.name}_nowcast.tex", "w").write("\n".join(lines))
print("saved tables to", args.out)
