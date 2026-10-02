import sys, os, json, argparse, subprocess
sys.path.insert(0, ".")
DOMAINS = ["kit", "dv", "chng_flu", "hosp_cov", "respinow", "nssp", "nhsn", "macro_m", "rtdsm_q", "alfred_w", "eia930"]
PRETTY = {"kit": "DE-Hosp(D)", "dv": "US-CLI(D)", "chng_flu": "US-Flu(D)", "hosp_cov": "US-Hosp(D)", "respinow": "DE-RESP(W)",
          "nssp": "US-NSSP(W)", "nhsn": "US-NHSN(W)", "macro_m": "US-Macro(M)", "rtdsm_q": "US-Macro(Q)", "alfred_w": "US-UI(W)", "eia930": "US-Grid(h)"}
ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="chronos2")
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--domains", default=",".join(DOMAINS))
ap.add_argument("--extra", default="")
ap.add_argument("--skip_run", action="store_true")
args = ap.parse_args()
MAXO = {"dv": 200, "chng_flu": 200, "hosp_cov": 200, "kit": 200, "eia930": 120}
doms = args.domains.split(",")
if not args.skip_run:
    for dom in doms:
        mo = f"--max_origins {MAXO[dom]}" if dom in MAXO else ""
        cmd = f"python scripts/eval_rrbench.py --domain {dom} --arch {args.arch} --model {args.model} --tag {args.tag} {mo} {args.extra}"
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        lines = [l for l in (r.stdout + r.stderr).splitlines() if "forecast:" in l or "nowcast :" in l or "Error" in l or "error" in l]
        print(dom, " | ".join(lines[-3:]), flush=True)
print(f"{'domain':14s} {'naive':>7s} {'oracle':>7s} | {'MASE':>7s} {'CRPS':>7s} {'cov80':>6s} | {'now':>6s} {'nowNaive':>8s} {'nowCL':>6s} {'sec':>6s}")
for dom in doms:
    f = f"results/rrbench/{args.tag}__{dom}.json"
    if not os.path.exists(f):
        continue
    r = json.load(open(f))
    nv = orc = float("nan")
    pn, po = f"results/rrbench/{args.arch}_naive__{dom}.json", f"results/rrbench/{args.arch}_oracle__{dom}.json"
    if os.path.exists(pn):
        nv = json.load(open(pn))["forecast"]["MASE"]
    if os.path.exists(po):
        orc = json.load(open(po))["forecast"]["MASE"]
    bl = f"results/rrbench/baselines__{dom}.json"
    cl = json.load(open(bl))["nowcast_cl"]["MASE"] if os.path.exists(bl) else float("nan")
    fc, nc = r["forecast"], r.get("nowcast", {})
    print(f"{PRETTY.get(dom, dom):14s} {nv:7.3f} {orc:7.3f} | {fc['MASE']:7.3f} {fc['CRPS_s']:7.3f} {fc['COV80']:6.2f} | "
          f"{nc.get('MASE', float('nan')):6.3f} {r.get('nowcast_naive_MASE', float('nan')):8.3f} {cl:6.3f} {r.get('seconds', float('nan')):6.0f}")
