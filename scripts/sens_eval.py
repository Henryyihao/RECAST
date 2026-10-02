import sys, os, json, argparse, subprocess
sys.path.insert(0, ".")
DOMAINS = ["kit", "dv", "chng_flu", "hosp_cov", "respinow", "nssp", "nhsn", "macro_m", "rtdsm_q", "alfred_w", "eia930"]
LDOM = {"kit": 80, "dv": 75, "chng_flu": 75, "hosp_cov": 75, "respinow": 10, "nssp": 8, "nhsn": 12, "macro_m": 3, "rtdsm_q": 3, "alfred_w": 4, "eia930": 72}
MAXO = {"dv": 200, "chng_flu": 200, "hosp_cov": 200, "kit": 200, "eia930": 120}
BASE = "--evidence_gate --separate_now_head --forecast_anchor --mult_head"
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--prefix", required=True)
ap.add_argument("--group", default="all")
ap.add_argument("--domains", default=",".join(DOMAINS))
ap.add_argument("--gpu", default="0")
args = ap.parse_args()
R = "results/rrbench"

runs = []
if args.group in ("anchor", "all"):
    runs += [(f"{args.prefix}_anc_med", f"{BASE} --anchor_mode paths --anchor_K 1", None),
             (f"{args.prefix}_anc_p9", f"{BASE} --anchor_mode paths --anchor_K 9", None),
             (f"{args.prefix}_anc_mc16", f"{BASE} --anchor_mode mc --anchor_K 16 --anchor_rho 0.8", None),
             (f"{args.prefix}_anc_mc16i", f"{BASE} --anchor_mode mc --anchor_K 16 --anchor_rho 0.0", None),
             (f"{args.prefix}_anc_mom", f"{BASE} --anchor_mode moment --anchor_K 3", None)]
if args.group in ("paths", "all"):
    runs += [(f"{args.prefix}_preonly", f"{BASE} --mc_anchor --pre_only", None),
             (f"{args.prefix}_anchoronly", f"{BASE} --mc_anchor --gate_off", None)]
if args.group in ("views", "all"):
    runs += [(f"{args.prefix}_views{k}", f"{BASE} --mc_anchor --n_views {k}", None) for k in (2, 4, 8)]
if args.group in ("L", "all"):
    for f, nm in [(0.5, "050"), (0.75, "075"), (1.25, "125"), (1.5, "150")]:
        runs.append((f"{args.prefix}_L{nm}", f"{BASE} --mc_anchor", (lambda dom, f=f: f"--L_model {max(1, int(LDOM[dom] * f + 0.5))}")))
for tag, extra, fn in runs:
    for dom in args.domains.split(","):
        if os.path.exists(f"{R}/{tag}__{dom}.json"):
            continue
        mo = f"--max_origins {MAXO[dom]}" if dom in MAXO else ""
        pd = fn(dom) if fn else ""
        cmd = f"cd . && CUDA_VISIBLE_DEVICES={args.gpu} python scripts/eval_rrbench.py --domain {dom} --arch chronos2 --model {args.model} --tag {tag} {mo} {extra} {pd}"
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        lines = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("forecast:") or l.startswith("nowcast :") or "Error" in l]
        print(tag, dom, " | ".join(l[:120] for l in lines[-2:]), flush=True)
print("SENS DONE", args.group)
