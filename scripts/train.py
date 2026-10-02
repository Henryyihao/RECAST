import sys, os, json, time, argparse, math
sys.path.insert(0, ".")
import numpy as np, torch
from torch.utils.data import DataLoader
from recast.models.va_chronos2 import VAChronos2
from recast.data import TriangleBatches
from recast.registry import build_model, NMAX_OVERRIDE
from recast.anchor import repaired_anchor, combine_forecast

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--arch", default="chronos2", help="chronos2 | bolt_s | bolt_b | toto | timemoe")
ap.add_argument("--backbone", default=None)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--lr_new", type=float, default=1e-4)
ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--workers", type=int, default=32)
ap.add_argument("--token_budget", type=int, default=14000)
ap.add_argument("--pool", default="data/pool_all.npz")
ap.add_argument("--no_pool", action="store_true")
ap.add_argument("--no_age_channel", action="store_true")
ap.add_argument("--no_age_bias", action="store_true")
ap.add_argument("--per_series_norm", action="store_true")
ap.add_argument("--mech", default=None, help="comma-separated allowed mechanisms (prior ablation)")
ap.add_argument("--no_modifiers", action="store_true")
ap.add_argument("--freeze", default="none", help="none | backbone (train input embedding + age bias + output head only)")
ap.add_argument("--no_age_attention", action="store_true")
ap.add_argument("--query_row", type=int, default=None)
ap.add_argument("--ages", default=None, help="view ages override, e.g. none")
ap.add_argument("--shift", action="store_true", help="shifted-window formulation (context ends at t-N) instead of edge heads")
ap.add_argument("--now_weight", type=float, default=1.0)
ap.add_argument("--p_real", type=float, default=0.4)
ap.add_argument("--evidence_gate", action="store_true")
ap.add_argument("--gate_penalty", type=float, default=0.0, help="sparsity penalty on the evidence gate (mean sigmoid activation)")
ap.add_argument("--separate_now_head", action="store_true")
ap.add_argument("--forecast_anchor", action="store_true")
ap.add_argument("--mult_head", action="store_true")
ap.add_argument("--no_rev", action="store_true", help="zero the in-context revision-profile channel")
ap.add_argument("--freeze_old_cols", action="store_true", help="freeze the pretrained columns of widened input embeddings")
ap.add_argument("--freeze_output_head", action="store_true")
ap.add_argument("--distill", type=float, default=0.0, help="weight of the oracle-distillation loss on the forecast block (0 = off)")
ap.add_argument("--distill_pre", type=float, default=0.0, help="additional oracle-distillation weight applied to the direct (un-anchored) forecast q_pre")
ap.add_argument("--now_aux", type=float, default=0.0, help="weight of the auxiliary pinball loss on the un-gated nowcast correction (trains the head irrespective of the gate)")
ap.add_argument("--min_bs", type=int, default=8, help="minimum batch size (token budget floor)")
ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing on the backbone (Time-MoE)")
ap.add_argument("--gate_sup", type=float, default=0.0, help="weight of the BCE that trains the evidence gate to predict whether the un-gated correction beats the latest report")
ap.add_argument("--w_true", type=float, default=1.0, help="weight of the true-value quantile loss on the forecast block")
ap.add_argument("--gate_fc_sup", type=float, default=0.0, help="weight of the BCE that trains the forecast gate to predict whether the direct forecast is closer to the oracle than the anchor")
ap.add_argument("--n_views", type=int, default=16, help="number of fixed-age views (age grid size)")
ap.add_argument("--no_now_gate", action="store_true", help="remove the nowcast gate entirely (g = 1); invariance from the zero-initialised head only")
ap.add_argument("--save_every", type=int, default=2500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--resume", default=None)
ap.add_argument("--out", default="checkpoints")
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
dev = "cuda"
os.makedirs(args.out, exist_ok=True)
os.makedirs("logs", exist_ok=True)
log_path = f"logs/train_{args.name}.log"
logf = open(log_path, "a")
def log(s):
    print(s, flush=True); logf.write(s + "\n"); logf.flush()
log(json.dumps(vars(args)))

teacher = None
if args.distill > 0 or args.forecast_anchor:
    from recast.plain import Plain
    teacher = build_model(args.arch, backbone=args.backbone, age_channel=False, age_bias=False, shared_norm=False, age_attention=False, edge=True, query_row=0).to(dev).eval()
    for p_ in teacher.parameters():
        p_.requires_grad = False
model = build_model(args.arch, backbone=args.backbone, age_channel=not args.no_age_channel, age_bias=not args.no_age_bias,
                    shared_norm=not args.per_series_norm, age_attention=not args.no_age_attention, query_row=args.query_row,
                    edge=not args.shift, now_weight=args.now_weight, evidence_gate=args.evidence_gate, separate_now_head=args.separate_now_head, forecast_anchor=args.forecast_anchor, mult_head=args.mult_head, no_now_gate=args.no_now_gate).to(dev)
if args.grad_ckpt:
    model.m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); log("gradient checkpointing enabled")
new_ids = {id(p) for p in model.new_parameters()}
emb_ids = {id(p) for p in model.input_embedding_parameters()}
if args.freeze == "backbone":
    for p in model.parameters():
        p.requires_grad = False
    for p in list(model.input_embedding_parameters()) + list(model.output_head_parameters()) + model.new_parameters():
        p.requires_grad = True
if args.freeze_output_head:
    for p in model.output_head_parameters():
        p.requires_grad = False
if args.freeze_old_cols:
    for w, n_old in model.old_input_cols:
        mask = torch.ones_like(w); mask[:, :n_old] = 0.0
        w.register_hook(lambda g, mask=mask: g * mask)
    log(f"froze pretrained columns of {len(model.old_input_cols)} widened input layers")
params_new = [p for p in model.parameters() if p.requires_grad and (id(p) in new_ids or id(p) in emb_ids)]
params_base = [p for p in model.parameters() if p.requires_grad and not (id(p) in new_ids or id(p) in emb_ids)]
log(f"trainable params: base {sum(p.numel() for p in params_base)/1e6:.1f}M new/emb {sum(p.numel() for p in params_new)/1e6:.2f}M")
opt = torch.optim.AdamW([{"params": params_base, "lr": args.lr, "weight_decay": 0.01}, {"params": params_new, "lr": args.lr_new, "weight_decay": 0.0}], betas=(0.9, 0.98))
def lr_lambda(step):
    if step < args.warmup:
        return step / max(1, args.warmup)
    pr = (step - args.warmup) / max(1, args.steps - args.warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, pr)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
step0 = 0
if args.resume:
    ck = torch.load(args.resume, map_location=dev)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); step0 = ck["step"]
    log(f"resumed from {args.resume} at step {step0}")

mech = args.mech.split(",") if args.mech else None
ages = None
if args.ages:
    ages = [] if args.ages == "none" else [int(a) for a in args.ages.split(",")]
ds = TriangleBatches(pool_path=None if args.no_pool else args.pool, seed=args.seed + step0, token_budget=args.token_budget,
                     mech_filter=mech, nmax=NMAX_OVERRIDE.get(args.arch), modifiers=not args.no_modifiers, ages=ages, edge=not args.shift, p_real=args.p_real, no_rev=args.no_rev,
                     patch=(1 if args.arch == "timemoe" else 16), min_bs=args.min_bs, n_views=args.n_views)
import multiprocessing as _mp
try:
    _mp.set_start_method('fork', force=True); torch.multiprocessing.set_start_method('fork', force=True)
except Exception as _e:
    log(f'start method: {_e}')
dl = DataLoader(ds, batch_size=None, num_workers=args.workers, prefetch_factor=4, persistent_workers=True, multiprocessing_context='fork')

model.train()
t0 = time.time(); run = {"loss": 0.0, "n": 0}
per_freq = {}
it = iter(dl)
for step in range(step0, args.steps):
    b = next(it)
    freq = b.pop("freq")
    b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
    oc = b.pop("oracle_ctx")
    tr = getattr(model, "target_row_default", 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**b, target_row=tr)
    if args.forecast_anchor:
        Nn = b["N"]; Hh0 = b["vals"].shape[-1] - b["ctx_len"]
        now_med = out["quantiles"][:, model.median_idx, :Nn].float().detach()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q_anchor = repaired_anchor(teacher, b["vals"], b["mask"], b["L"], b["ctx_len"], Nn, Hh0, now_med, out["loc"], out["scale"], model.uses_arcsinh, tr)
        combine_forecast(out, q_anchor, Nn, Hh0, model.uses_arcsinh, model.qlevels)
    loss = out["loss"]
    if teacher is not None:
        Nn = b["N"]; Hh = b["vals"].shape[-1] - b["ctx_len"]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            Bn = oc.shape[0]
            tv = torch.cat([oc, torch.full((Bn, Hh), float("nan"), device=dev)], -1)[:, None, :]
            to = teacher(tv, torch.isfinite(tv).float(), torch.zeros_like(tv), torch.zeros(Bn, 1, device=dev), b["L"], ctx_len=b["ctx_len"], N=0, target_row=0)
            qt_raw = to["quantiles"].float()[:, :, :Hh]
        loc_t, scale_t = out["loc"], out["scale"]
        if loc_t.shape[-1] == 1:
            qt_norm = (qt_raw - loc_t[:, None, :]) / scale_t[:, None, :]
        else:
            qt_norm = (qt_raw - loc_t[:, None, -Hh:]) / scale_t[:, None, -Hh:]
        if getattr(model, "uses_arcsinh", True):
            qt_norm = torch.arcsinh(qt_norm)
        qs = out["q_norm"].float()
        qs = qs[:, :, -Hh:] if qs.shape[-1] > Hh else qs

        okd = (qt_norm.abs().amax(dim=(1, 2)) < 30.0).float()
        dl = ((qs - qt_norm).abs().mean(-1).sum(1) * okd).sum() / okd.sum().clamp(min=1.0)
        loss = (args.w_true * out["lf"] + model.now_weight * out["ln"] + args.distill * dl) if "lf" in out else loss + args.distill * dl
        run["dl"] = run.get("dl", 0.0) + dl.item()
        if args.distill_pre > 0 and "q_pre" in out:
            qp = out["q_pre"].float()[:, :, :Hh]
            if not getattr(model, "uses_arcsinh", True):
                qp = qp.clamp(-50, 50)
            dlp = ((qp - qt_norm).abs().mean(-1).sum(1) * okd).sum() / okd.sum().clamp(min=1.0)
            loss = loss + args.distill_pre * dlp
            run["dlp"] = run.get("dlp", 0.0) + dlp.item()
        if args.gate_fc_sup > 0 and "q_pre" in out and "gate_fc" in out and args.forecast_anchor:
            qp = out["q_pre"].float()[:, :, :Hh]
            if not getattr(model, "uses_arcsinh", True):
                qp = qp.clamp(-50, 50)
            lab = ((qp - qt_norm).abs() < (q_anchor.float()[:, :, :Hh] - qt_norm).abs()).float()
            gf = out["gate_fc"].float()[:, :, :Hh].clamp(1e-4, 1 - 1e-4)
            bce = -(lab * torch.log(gf) + (1 - lab) * torch.log(1 - gf)).mean(dim=(1, 2))
            lgf = (bce * okd).sum() / okd.sum().clamp(min=1.0)
            loss = loss + args.gate_fc_sup * lgf
            run["lgf"] = run.get("lgf", 0.0) + lgf.item()
    if args.now_aux > 0 and "ln_u" in out:
        loss = loss + args.now_aux * out["ln_u"]; run["lnu"] = run.get("lnu", 0.0) + out["ln_u"].item()
    if args.gate_sup > 0 and "lg" in out:
        loss = loss + args.gate_sup * out["lg"]; run["lg"] = run.get("lg", 0.0) + out["lg"].item()
    if args.gate_penalty > 0 and "gate" in out:
        gp = out["gate"].float().mean() * out["gate"].shape[1]
        if "gate_fc" in out:
            gp = gp + out["gate_fc"].float().mean() * out["gate_fc"].shape[1]
            run["gpf"] = run.get("gpf", 0.0) + out["gate_fc"].float().mean().item()
        loss = loss + args.gate_penalty * gp
        run["gp"] = run.get("gp", 0.0) + out["gate"].float().mean().item()
    if not torch.isfinite(loss):
        log(f"step {step} non-finite loss, skipping"); opt.zero_grad(); continue
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    run["loss"] += loss.item(); run["n"] += 1
    if "loss_now" in out:
        run["ln"] = run.get("ln", 0.0) + out["loss_now"].item(); run["lf"] = run.get("lf", 0.0) + out["loss_fc"].item()
    pf = per_freq.setdefault(freq, [0.0, 0]); pf[0] += loss.item(); pf[1] += 1
    if (step + 1) % 100 == 0:
        el = time.time() - t0
        pfs = " ".join(f"{k}:{v[0]/max(1,v[1]):.3f}" for k, v in sorted(per_freq.items()))
        extra = f" now {run.get('ln',0)/run['n']:.3f} fc {run.get('lf',0)/run['n']:.3f}" if "ln" in run else ""
        extra += f" dist {run['dl']/run['n']:.3f}" if "dl" in run else ""
        extra += f" gate {run['gp']/run['n']:.3f}" if "gp" in run else ""
        extra += f" dpre {run['dlp']/run['n']:.3f}" if "dlp" in run else ""
        extra += f" nowU {run['lnu']/run['n']:.3f}" if "lnu" in run else ""
        extra += f" gsup {run['lg']/run['n']:.3f}" if "lg" in run else ""
        extra += f" gfc {run['gpf']/run['n']:.3f}" if "gpf" in run else ""
        extra += f" gfsup {run['lgf']/run['n']:.3f}" if "lgf" in run else ""
        log(f"step {step+1} loss {run['loss']/run['n']:.4f}{extra} [{pfs}] gn {gn:.2f} lr {sched.get_last_lr()[0]:.2e} {el/ (step+1-step0):.2f}s/step mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
        run = {"loss": 0.0, "n": 0}; per_freq = {}
    if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
        torch.save({"model": {k: v for k, v in model.state_dict().items()}, "step": step + 1, "args": vars(args)}, f"{args.out}/{args.name}.pt")
        log(f"saved {args.out}/{args.name}.pt at step {step+1}")
log("DONE")
