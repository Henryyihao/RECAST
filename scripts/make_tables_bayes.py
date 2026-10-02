import sys, json, argparse, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--agg", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--out", default="results/tables/tab_bayes.tex")
ap.add_argument("--domains", default="kit,nhsn")
args = ap.parse_args()
A = json.load(open(args.agg))
PRETTY = {"kit": "DE-Hosp (D)", "dv": "US-CLI (D)", "chng_flu": "US-Flu (D)", "hosp_cov": "US-Hosp (D)", "respinow": "DE-RESP (W)",
          "nssp": "US-NSSP (W)", "nhsn": "US-NHSN (W)", "macro_m": "US-Macro (M)", "rtdsm_q": "US-Macro (Q)", "alfred_w": "US-UI (W)", "eia930": "US-Grid (h)"}


def g(d, nm, blk, k):
    try:
        return A[d][nm][blk][k]
    except KeyError:
        return float("nan")


def fmt(v, best=False, d=3):
    if not np.isfinite(v):
        return "--"
    s = f"{v:.{d}f}"
    return "\\textbf{" + s + "}" if best else s


def sig(d, nm):
    t = A[d].get("tests", {}).get(nm)
    if not t:
        return ""
    p = t["p"]
    if p < 0.01:
        return "$^{\\dagger}$" if t["mean_diff"] < 0 else "$^{\\ddagger}$"
    if p < 0.05:
        return "$^{*}$" if t["mean_diff"] < 0 else "$^{\\circ}$"
    return ""


rows = ["\\begin{tabular*}{\\tblwidth}{@{}llrrrrrrr@{}}\n\\toprule",
        " & & \\multicolumn{4}{c}{nowcast (last $N$ periods)} & \\multicolumn{3}{c}{forecast (two-stage vs.\\ joint)} \\\\",
        "\\cmidrule(lr){3-6}\\cmidrule(lr){7-9}",
        "Domain & metric & latest & CL & Bayes & \\recast{} & CL$\\to$TSFM & Bayes$\\to$TSFM & \\recast{} \\\\\n\\midrule"]
for d in args.domains.split(","):
    if d not in A or "bayes" not in A[d]:
        continue
    r = A[d]
    n = r["n_eval"]
    for met, lab in [("MASE", "MASE"), ("CRPS_s", "CRPS"), ("COV80", "Cov80")]:
        nv = r["naive_now"]["MASE"] if met == "MASE" else float("nan")
        cl, by, ve = g(d, "cl", "nowcast", met), g(d, "bayes", "nowcast", met), g(d, args.tag, "nowcast", met)
        cand = [nv, cl, by, ve]
        best = int(np.nanargmin(cand)) if met != "COV80" else -1
        cells = [fmt(v, best=(i == best), d=(2 if met == "COV80" else 3)) for i, v in enumerate(cand)]
        cells[3] = cells[3] + (sig(d, "bayes@now") if met == "CRPS_s" else "")
        f2, fb, fv = g(d, "twostage", "forecast", met), g(d, "bayes_twostage_mc", "forecast", met), g(d, args.tag, "forecast", met)
        fc = [f2, fb, fv]
        bestf = int(np.nanargmin(fc)) if met != "COV80" else -1
        fcells = [fmt(v, best=(i == bestf), d=(2 if met == "COV80" else 3)) for i, v in enumerate(fc)]
        fcells[2] = fcells[2] + (sig(d, "bayes_twostage_mc") if met == "CRPS_s" else "")
        label = f"{PRETTY.get(d, d)} ($n$={n})" if met == "MASE" else ""
        rows.append(f"{label} & {lab} & " + " & ".join(cells) + " & " + " & ".join(fcells) + " \\\\")
rows.append("\\bottomrule\n\\end{tabular*}")
open(args.out, "w").write("\n".join(rows))
print("\n".join(rows))
