import sys, json, argparse, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--diags", required=True); ap.add_argument("--out", required=True); ap.add_argument("--numbers", default=None)
ap.add_argument("--max_series", type=int, default=6)
args = ap.parse_args()
PRETTY = {"alfred_w": "US-UI (W)", "rtdsm_q": "US-Macro (Q)", "macro_m": "US-Macro (M)", "nhsn": "US-NHSN (W)", "nssp": "US-NSSP (W)", "respinow": "DE-RESP (W)", "chng_flu": "US-Flu (D)"}
rows = ["\\begin{tabular*}{\\tblwidth}{@{}llrrrrrrr@{}}\n\\toprule",
        "Domain & series & origins & \\naive{} & \\recast{} & $\\Delta$ (\\%) & worst 5\\% share & gate$^{\\mathrm{fc}}$ & width ratio \\\\\n\\midrule"]
macros = {}
for p in args.diags.split(","):
    D = json.load(open(p)); dom = D["domain"]; o = D["overall"]
    ser = D["series"]

    ser_sorted = sorted(ser, key=lambda r: -(r["crps_recast"] - r["crps_naive"]) * r["n"])
    shown = ser_sorted[:args.max_series]
    for i, r in enumerate(shown):

        rs = [x for x in D["origins"] if x["sid"] == r["sid"]]
        dl = np.array([x["crps_recast"] - x["crps_naive"] for x in rs]); k = max(1, int(round(0.05 * len(dl)))); order = np.argsort(-dl)
        share = dl[order[:k]].sum() / dl.sum() if dl.sum() > 0 else float("nan")
        lab = PRETTY.get(dom, dom) if i == 0 else ""
        name = r["sid"].replace("_", "\\_").replace("|", " ")
        rows.append(f"{lab} & {name} & {r['n']} & {r['crps_naive']:.3f} & {r['crps_recast']:.3f} & {r['rel']:+.1f} & {100 * share:.0f}\\% & {r['gate_fc']:.2f} & {r['width_ratio']:.2f} \\\\")
    if len(ser_sorted) > args.max_series:
        rest = ser_sorted[args.max_series:]
        n = sum(r["n"] for r in rest); cn = sum(r["crps_naive"] * r["n"] for r in rest) / n; cv = sum(r["crps_recast"] * r["n"] for r in rest) / n
        gf = sum(r["gate_fc"] * r["n"] for r in rest) / n; wr = sum(r["width_ratio"] * r["n"] for r in rest) / n
        rows.append(f" & other {len(rest)} series & {n} & {cn:.3f} & {cv:.3f} & {100 * (cv / cn - 1):+.1f} & -- & {gf:.2f} & {wr:.2f} \\\\")
    rows.append(f" & all & {o['n']} & {o['crps_naive']:.3f} & {o['crps_recast']:.3f} & {100 * (o['crps_recast'] / o['crps_naive'] - 1):+.1f} & {100 * o['share_worst5pct']:.0f}\\% & {o['gate_fc']:.2f} & {o['width_ratio']:.2f} \\\\\n\\midrule")
    key = {"alfred_w": "ui", "rtdsm_q": "macroQ", "chng_flu": "flu", "nssp": "nssp"}.get(dom, dom)
    macros[f"fail{key}Share"] = f"{100 * o['share_worst5pct']:.0f}"; macros[f"fail{key}FracWorse"] = f"{100 * o['frac_origins_worse']:.0f}"; macros[f"fail{key}FracBetter"] = f"{100 * (1 - o['frac_origins_worse']):.0f}"
    macros[f"fail{key}MedianD"] = f"{o['median_dcrps']:.4f}"; macros[f"fail{key}GateFc"] = f"{o['gate_fc']:.2f}"; macros[f"fail{key}Width"] = f"{o['width_ratio']:.2f}"

    dl = np.array([x["crps_recast"] - x["crps_naive"] for x in D["origins"]]); cn = np.array([x["crps_naive"] for x in D["origins"]]); cv = np.array([x["crps_recast"] for x in D["origins"]])
    k = max(1, int(round(0.05 * len(dl)))); keep = np.argsort(-dl)[k:]
    macros[f"fail{key}RelExcl"] = f"{100 * (cv[keep].mean() / cn[keep].mean() - 1):+.1f}"; macros[f"fail{key}RelExclAbs"] = f"{abs(100 * (cv[keep].mean() / cn[keep].mean() - 1)):.1f}"
    macros[f"fail{key}MedianDabs"] = f"{abs(o['median_dcrps']):.3f}"; macros[f"fail{key}MedianDpct"] = f"{abs(100 * np.median((cv - cn) / cn)):.1f}"

    j = int(np.argmax(dl)); w = D["origins"][j]
    macros[f"fail{key}WorstShare"] = f"{100 * dl[j] / dl.sum():.0f}" if dl.sum() > 0 else "--"
    macros[f"fail{key}WorstSid"] = w["sid"].replace("_", "\\_"); macros[f"fail{key}WorstGateFc"] = f"{w['gate_fc']:.2f}"; macros[f"fail{key}WorstWidth"] = f"{w['width_recast'] / max(w['width_naive'], 1e-9):.0f}"
rows[-1] = rows[-1].replace("\n\\midrule", "")
rows.append("\\bottomrule\n\\end{tabular*}")
open(args.out, "w").write("\n".join(rows)); print("\n".join(rows))
if args.numbers:
    with open(args.numbers, "a") as f:
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("appended", len(macros), "macros")
