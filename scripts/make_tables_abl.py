import sys, json, argparse, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--agg", required=True); ap.add_argument("--main", required=True); ap.add_argument("--tags", required=True)
ap.add_argument("--out", default="results/tables/tab_ablation.tex")
ap.add_argument("--numbers", default=None, help="append macros abl<Name><RelH|RelL|Gap|NowH|NowL> to this file")
args = ap.parse_args()
MAC = {"abl2_main": "Main", "abl2_noviews": "Noviews", "abl2_noage": "Noage", "abl2_norev": "Norev", "abl2_nodistill": "Nodistill", "abl2_noanchorfc": "Noanchor", "abl2_nogate2": "NogateTwo",
       "abl2_nogate0": "Nogate", "abl_main7500": "Noaux", "abl2_backfill": "Backfill", "abl2_nobackfill": "Nobackfill", "abl2_nomod": "Nomod", "abl2_synthbase": "Synthbase", "abl2_light": "Light", "abl_shift": "Shift", "abl2_views4": "ViewsFour", "abl2_views8": "ViewsEight", "abl2_gatefc": "Gatefc"}
A = json.load(open(args.agg))
ORDER = ["kit", "dv", "chng_flu", "hosp_cov", "respinow", "nssp", "nhsn", "macro_m", "rtdsm_q", "alfred_w", "eia930"]
HEAVY = ["kit", "dv", "chng_flu", "hosp_cov", "eia930"]; LIGHT = [d for d in ORDER if d not in HEAVY]
LAB = {"abl_noviews": "no age-view streams (as-of + settled only)", "abl_noage": "no age channels / relative-age bias", "abl_norev": "no in-context revision-profile channel",
       "abl_nodistill": "true-value forecast loss instead of oracle distillation", "abl_noanchorfc": "no forecast anchor", "abl_nogate2": "global (input-independent) gate instead of evidence gate", "abl_nogate": "global gate (degenerate zero init)",
       "abl_backfill": "prior: delayed reporting only", "abl_nobackfill": "prior: without delayed reporting", "abl_nomod": "prior: no modifiers", "abl_synthbase": "synthetic base signals only",
       "abl_light": "lightweight: pretrained weights frozen", "abl_shift": "shifted output window (no edge heads, no anchors)", "v7a_7500": "additive correction head", "abl_aux": "+ auxiliary un-gated losses (Section~\\ref{sec:training})",
       "abl2_noviews": "no age-view streams (as-of + settled only)", "abl2_noage": "no age channels / relative-age bias", "abl2_norev": "no in-context revision-profile channel",
       "abl2_nodistill": "true-value forecast loss instead of oracle distillation", "abl2_noanchorfc": "no forecast anchor", "abl2_nogate2": "global (input-independent) gate instead of evidence gate", "abl2_nogate0": "no nowcast gate (zero-initialised correction only)",
       "abl2_backfill": "prior: delayed reporting only", "abl2_nobackfill": "prior: without delayed reporting", "abl2_nomod": "prior: no modifiers", "abl2_synthbase": "synthetic base signals only",
       "abl2_light": "lightweight: pretrained weights frozen", "abl_main7500": "gated outputs only (no arm-wise supervision of the gates)", "abl2_views4": "4 age views instead of 16", "abl2_views8": "8 age views instead of 16", "abl2_gatefc": "+ supervision of the forecast gate (as for the nowcast gate)"}
def stats(tag):
    rel_h, rel_l, gap, now_h, now_l = [], [], [], [], []
    for d in ORDER:
        if d not in A or tag not in A[d] or "forecast" not in A[d][tag]:
            continue
        nv = A[d]["plain_naive"]["forecast"]["MASE"]; orc = A[d]["plain_oracle"]["forecast"]["MASE"]; v = A[d][tag]["forecast"]["MASE"]
        (rel_h if d in HEAVY else rel_l).append(v / nv)
        if (nv - orc) / nv >= 0.02:
            gap.append(100 * (nv - v) / (nv - orc))
        if "nowcast" in A[d][tag]:
            nvn = A[d]["naive_now"]["MASE"]; (now_h if d in HEAVY else now_l).append(100 * (1 - A[d][tag]["nowcast"]["MASE"] / nvn))
    f = lambda x: f"{np.mean(x):.3f}" if len(x) else "--"; g = lambda x: f"{np.mean(x):.0f}" if len(x) else "--"
    return f(rel_h), f(rel_l), g(gap), g(now_h), g(now_l)
rows = ["\\begin{tabular*}{\\tblwidth}{@{}lrrrrr@{}}\n\\toprule",
        " & \\multicolumn{2}{c}{forecast MASE / \\naive{}} & gap closed & \\multicolumn{2}{c}{nowcast impr. (\\%)} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}",
        "Variant & heavy-rev. & light-rev. & (\\%) & heavy-rev. & light-rev. \\\\\n\\midrule"]
rows.append("\\recast{} (full recipe) & " + " & ".join(stats(args.main)) + " \\\\\n\\midrule")
for t in args.tags.split(","):
    rows.append(f"{LAB.get(t, t)} & " + " & ".join(stats(t)) + " \\\\")
rows.append("\\bottomrule\n\\end{tabular*}")
open(args.out, "w").write("\n".join(rows)); print("\n".join(rows))
if args.numbers:
    with open(args.numbers, "a") as f:
        for t in [args.main] + args.tags.split(",") + ["abl2_views4", "abl2_views8"]:
            if t not in MAC or not any(t in A[d] for d in A):
                continue
            rh, rl, gp, nh, nl = stats(t); nm = MAC[t]
            f.write(f"\\newcommand{{\\abl{nm}RelH}}{{{rh}}}\n\\newcommand{{\\abl{nm}RelL}}{{{rl}}}\n\\newcommand{{\\abl{nm}Gap}}{{{gp}}}\n\\newcommand{{\\abl{nm}NowH}}{{{nh}}}\n\\newcommand{{\\abl{nm}NowL}}{{{nl}}}\n")
            if rl != "--":
                f.write(f"\\newcommand{{\\abl{nm}RelLpct}}{{{100 * (float(rl) - 1):.0f}}}\n")
            if nl != "--":
                f.write(f"\\newcommand{{\\abl{nm}NowLabs}}{{{abs(float(nl)):.0f}}}\n")
    print("appended ablation macros")
