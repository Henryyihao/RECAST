import sys, json, numpy as np
sys.path.insert(0, ".")
from recast.datasets import load_bundle
from recast.protocol import make_tasks
ORDER = ["kit", "dv", "chng_flu", "hosp_cov", "respinow", "nssp", "nhsn", "macro_m", "rtdsm_q", "alfred_w", "eia930"]
PRETTY = {"kit": ("DE-Hosp", "day"), "dv": ("US-CLI", "day"), "chng_flu": ("US-Flu", "day"), "hosp_cov": ("US-Hosp", "day"), "respinow": ("DE-RESP", "week"),
          "nssp": ("US-NSSP", "week"), "nhsn": ("US-NHSN", "week"), "macro_m": ("US-Macro-M", "month"), "rtdsm_q": ("US-Macro-Q", "quarter"), "alfred_w": ("US-UI", "week"), "eia930": ("US-Grid", "hour")}
MAXO = {"dv": 200, "chng_flu": 200, "hosp_cov": 200, "kit": 200, "eia930": 120}
rows = []
for d in ORDER:
    b = load_bundle(d); L = b["L"]; ser = b["series"]
    rels = []; T = 0
    for sid, v in ser.items():
        A, R = v["A"], v["R"]; T = max(T, A.shape[0]); settled = A[:, L]
        u = np.arange(A.shape[0]); Rf = np.where(np.isfinite(R), R, u); a0 = np.clip(Rf - u, 0, L).astype(int)
        first = A[u, a0]
        ok = np.isfinite(R) & np.isfinite(first) & np.isfinite(settled) & (np.abs(settled) > 1e-9)
        rels.append(np.abs(first[ok] - settled[ok]) / np.abs(settled[ok]) * 100)
    rel = np.concatenate(rels)
    tasks = make_tasks(b, max_origins=MAXO.get(d)); n_cells = sum(len(t.origins) for t in tasks)
    frac_rev = 100 * np.mean(rel > 0.5)
    mean_rev_given = np.median(rel[rel > 0.5]) if (rel > 0.5).any() else 0.0
    rows.append((PRETTY[d][0], PRETTY[d][1], len(ser), T, L, np.median(rel), np.mean(rel), frac_rev, mean_rev_given, n_cells))
    print(rows[-1], flush=True)
lines = ["\\begin{tabular*}{\\tblwidth}{@{}llrrrrrrrr@{}}\n\\toprule",
         "Domain & step & series & $T$ & $L$ & median $|$rev$|$ (\\%) & mean $|$rev$|$ (\\%) & revised (\\%) & median if revised (\\%) & origins \\\\\n\\midrule"]
for r in rows:
    lines.append(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]:.1f} & {r[6]:.1f} & {r[7]:.1f} & {r[8]:.1f} & {r[9]} \\\\".replace("& 69.5 &", "& 69.5$^{a}$ &"))
lines.append("\\bottomrule\n\\end{tabular*}")
open("results/tables/tab_domains.tex", "w").write("\n".join(lines)); print("\n".join(lines))
