import sys, os, json, argparse, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--agg", required=True); ap.add_argument("--main", required=True); ap.add_argument("--prefix", required=True)
ap.add_argument("--abl", default=None, help="aggregated ablation json (7.5k-step models with 4/8 views, for the view table)")
ap.add_argument("--abl_main", default="abl2_main")
ap.add_argument("--out", default="results/tables")
ap.add_argument("--numbers", default=None)
ap.add_argument("--time_prefix", default="v12t", help="prefix of dedicated timing runs (<prefix>_views<K>__<dom>.json, GPU otherwise idle)")
args = ap.parse_args()
MACROS = {}
A = json.load(open(args.agg))
ORDER = ["kit", "dv", "chng_flu", "hosp_cov", "respinow", "nssp", "nhsn", "macro_m", "rtdsm_q", "alfred_w", "eia930"]
HEAVY = ["kit", "dv", "chng_flu", "hosp_cov", "eia930"]
R = "results/rrbench"


def stats(A, tag, ref="plain_naive"):

    rel_h, rel_l, gap, now_h, now_l, cov_h, wins = [], [], [], [], [], [], 0
    for d in ORDER:
        if d not in A or tag not in A[d] or "forecast" not in A[d][tag]:
            continue
        nv = A[d][ref]["forecast"]["CRPS_s"]; orc = A[d]["plain_oracle"]["forecast"]["CRPS_s"]; v = A[d][tag]["forecast"]["CRPS_s"]
        (rel_h if d in HEAVY else rel_l).append(v / nv); wins += int(v < nv)
        if (nv - orc) / nv >= 0.02:
            gap.append(100 * (nv - v) / (nv - orc))
        if d in HEAVY:
            cov_h.append(A[d][tag]["forecast"]["COV80"])
        if "nowcast" in A[d][tag]:
            nvn = A[d]["naive_now"]["MASE"]; (now_h if d in HEAVY else now_l).append(100 * (1 - A[d][tag]["nowcast"]["MASE"] / nvn))
    f = lambda x: f"{np.mean(x):.3f}" if len(x) else "--"; g = lambda x: f"{np.mean(x):.0f}" if len(x) else "--"; c = lambda x: f"{np.mean(x):.2f}" if len(x) else "--"
    return dict(rel_h=f(rel_h), rel_l=f(rel_l), gap=g(gap), now_h=g(now_h), now_l=g(now_l), cov_h=c(cov_h), wins=f"{wins}/{len(rel_h) + len(rel_l)}")


def seconds(tag):
    tot = 0.0; n = 0
    for d in ORDER:
        p = f"{R}/{tag}__{d}.json"
        if os.path.exists(p):
            r = json.load(open(p)); tot += r["seconds"]; n += r["n"]
    return 1000 * tot / n if n else np.nan


def seconds_views(k):

    t = f"{args.time_prefix}_views{k}"
    v = seconds(t)
    return v


P = args.prefix
rows = ["\\begin{tabular*}{\\tblwidth}{@{}lrrrrrr@{}}\n\\toprule",
        "Forecast pathway / anchoring scheme & passes & \\multicolumn{2}{c}{CRPS / \\naive{}} & gap closed & Cov80 & wins \\\\",
        "\\cmidrule(lr){3-4}", " & & heavy-rev. & light-rev. & (\\%) & heavy-rev. & vs \\naive{} \\\\\n\\midrule"]
for tag, lab, passes in [(f"{P}_preonly", "direct forecast only (no anchor)", 0), (f"{P}_anchoronly", "anchor only (frozen backbone on repaired context)", 3),
                         (f"{P}_anc_med", "anchored: median nowcast path", 1), (args.main, "anchored: 3 quantile paths (0.1/0.5/0.9)", 3),
                         (f"{P}_anc_p9", "anchored: 9 quantile paths", 9), (f"{P}_anc_mc16", "anchored: 16 MC paths, Gaussian copula ($\\rho$=0.8)", 16),
                         (f"{P}_anc_mc16i", "anchored: 16 MC paths, independent", 16), (f"{P}_anc_mom", "anchored: 3 paths, moment matching", 3)]:
    if not any(tag in A[d] for d in A):
        continue
    s = stats(A, tag)
    rows.append(f"{lab} & {passes} & {s['rel_h']} & {s['rel_l']} & {s['gap']} & {s['cov_h']} & {s['wins']} \\\\")
rows.append("\\bottomrule\n\\end{tabular*}")
open(f"{args.out}/tab_anchor.tex", "w").write("\n".join(rows)); print("\n".join(rows))
covs = []
for tag, nm in [(f"{P}_anc_med", "Med"), (args.main, "Main"), (f"{P}_anc_p9", "Pnine"), (f"{P}_anc_mc16", "Mc"), (f"{P}_anc_mc16i", "Mci"), (f"{P}_anc_mom", "Mom"), (f"{P}_preonly", "Pre"), (f"{P}_anchoronly", "Anc")]:
    if any(tag in A[d] for d in A):
        s_ = stats(A, tag); MACROS[f"anc{nm}Cov"] = s_["cov_h"]; MACROS[f"anc{nm}RelH"] = s_["rel_h"]; MACROS[f"anc{nm}RelL"] = s_["rel_l"]; MACROS[f"anc{nm}Gap"] = s_["gap"]
        if nm in ("Main", "Pnine", "Mc", "Mci"):
            covs.append(float(s_["cov_h"]))
if covs:
    MACROS["ancCovRange"] = f"{max(covs) - min(covs):.2f}"


rows = ["\\begin{tabular*}{\\tblwidth}{@{}lrrrrrrrr@{}}\n\\toprule",
        "Age grid & views $K$ & streams $S$ & ms/origin & \\multicolumn{2}{c}{CRPS / \\naive{}} & gap closed & \\multicolumn{2}{c}{nowcast impr. (\\%)} \\\\",
        "\\cmidrule(lr){5-6}\\cmidrule(lr){8-9}", " & & & & heavy-rev. & light-rev. & (\\%) & heavy-rev. & light-rev. \\\\\n\\midrule"]
for tag, k in [(f"{P}_views2", 2), (f"{P}_views4", 4), (f"{P}_views8", 8), (args.main, 16)]:
    if not any(tag in A[d] for d in A):
        continue
    s = stats(A, tag)
    rows.append(f"16-view model evaluated with $K$ views & {k} & {k + 2} & {seconds_views(k):.1f} & {s['rel_h']} & {s['rel_l']} & {s['gap']} & {s['now_h']} & {s['now_l']} \\\\")
if args.abl and os.path.exists(args.abl):
    B = json.load(open(args.abl))
    rows.append("\\midrule")
    for tag, k in [("abl2_views4", 4), ("abl2_views8", 8), (args.abl_main, 16)]:
        if not any(tag in B[d] for d in B):
            continue
        s = stats(B, tag)
        rows.append(f"trained with $K$ views (7.5k steps) & {k} & {k + 2} & {seconds_views(k):.1f} & {s['rel_h']} & {s['rel_l']} & {s['gap']} & {s['now_h']} & {s['now_l']} \\\\")
rows.append("\\bottomrule\n\\end{tabular*}")
open(f"{args.out}/tab_views.tex", "w").write("\n".join(rows)); print("\n".join(rows))
for tag, nm in [(f"{P}_views2", "Two"), (f"{P}_views4", "Four"), (f"{P}_views8", "Eight"), (args.main, "Main")]:
    if any(tag in A[d] for d in A):
        s_ = stats(A, tag); MACROS[f"views{nm}RelH"] = s_["rel_h"]; MACROS[f"views{nm}RelL"] = s_["rel_l"]; MACROS[f"views{nm}NowH"] = s_["now_h"]; MACROS[f"views{nm}NowL"] = s_["now_l"]; MACROS[f"views{nm}Gap"] = s_["gap"]; MACROS[f"views{nm}Ms"] = f"{seconds_views({'Two': 2, 'Four': 4, 'Eight': 8, 'Main': 16}[nm]):.1f}"


PRETTY = {"kit": "DE-Hosp", "dv": "US-CLI", "chng_flu": "US-Flu", "hosp_cov": "US-Hosp", "eia930": "US-Grid"}


def rel_dom(A, tag, d):
    try:
        return A[d][tag]["forecast"]["CRPS_s"] / A[d]["plain_naive"]["forecast"]["CRPS_s"]
    except KeyError:
        return float("nan")


rows = ["\\begin{tabular*}{\\tblwidth}{@{}lrrrrrrrr@{}}\n\\toprule",
        "Settlement age & \\multicolumn{6}{c}{forecast CRPS / \\naive{}} & \\multicolumn{2}{c}{nowcast impr. (\\%)} \\\\",
        "\\cmidrule(lr){2-7}\\cmidrule(lr){8-9}", "assumed by the model & DE-Hosp & US-CLI & US-Flu & US-Hosp & US-Grid & light-rev.\\ (mean) & heavy-rev. & light-rev. \\\\\n\\midrule"]
for tag, lab in [(f"{P}_L050", "$0.5\\,L$"), (f"{P}_L075", "$0.75\\,L$"), (args.main, "$L$ (correct)"), (f"{P}_L125", "$1.25\\,L$"), (f"{P}_L150", "$1.5\\,L$")]:
    if not any(tag in A[d] for d in A):
        continue
    s = stats(A, tag)
    cells = [f"{rel_dom(A, tag, d):.3f}" for d in ["kit", "dv", "chng_flu", "hosp_cov", "eia930"]]
    rows.append(f"{lab} & " + " & ".join(cells) + f" & {s['rel_l']} & {s['now_h']} & {s['now_l']} \\\\")
rows.append("\\bottomrule\n\\end{tabular*}")
open(f"{args.out}/tab_L.tex", "w").write("\n".join(rows)); print("\n".join(rows))
rels = []
KEYD = {"kit": "kit", "dv": "dv", "chng_flu": "flu", "hosp_cov": "hosp", "eia930": "grid"}
for tag, nm in [(f"{P}_L050", "Half"), (f"{P}_L075", "ThreeQ"), (args.main, "Main"), (f"{P}_L125", "FiveQ"), (f"{P}_L150", "Sesq")]:
    if any(tag in A[d] for d in A):
        s_ = stats(A, tag); MACROS[f"L{nm}RelH"] = s_["rel_h"]; MACROS[f"L{nm}RelL"] = s_["rel_l"]; MACROS[f"L{nm}NowH"] = s_["now_h"]; MACROS[f"L{nm}NowL"] = s_["now_l"]; MACROS[f"L{nm}Gap"] = s_["gap"]
        for d, k in KEYD.items():
            MACROS[f"L{nm}Rel{k}"] = f"{rel_dom(A, tag, d):.3f}"

        vals = [rel_dom(A, tag, d) for d in ["kit", "dv", "chng_flu", "hosp_cov"]]; MACROS[f"L{nm}RelHnoGrid"] = f"{np.nanmean(vals):.3f}"
        if nm in ("Half", "ThreeQ", "Main", "FiveQ", "Sesq"):
            rels.append(np.nanmean(vals))
if rels:
    MACROS["LRelRange"] = f"{100 * (max(rels) - min(rels)):.1f}"
if args.numbers:
    with open(args.numbers, "a") as f:
        for k, v in MACROS.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("appended", len(MACROS), "sensitivity macros")
