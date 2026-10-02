import sys, os, json, argparse, time
sys.path.insert(0, ".")
import numpy as np

LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


_counter = None


def _init_worker(counter, n_cores, first_core):


    with counter.get_lock():
        i = counter.value; counter.value += 1
    core = first_core + (i % n_cores)
    try:
        os.sched_setaffinity(0, {core})
    except Exception:
        pass
    os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["OMP_NUM_THREADS"] = "1"


_CACHE = {}


def _model(inc, obs, log_mean0, W, D):
    import jax, jax.numpy as jnp
    import numpyro, numpyro.distributions as dist
    tau = numpyro.sample("tau", dist.HalfNormal(0.5))
    r = numpyro.sample("r", dist.Gamma(2.0, 0.1))

    sig_xi = numpyro.sample("sig_xi", dist.HalfNormal(1.0))
    xi0 = numpyro.sample("xi0", dist.Normal(0.0, 3.0))
    eta = numpyro.sample("eta", dist.Normal(0.0, 1.0).expand([D]))
    xi = jnp.concatenate([jnp.array([xi0]), xi0 + sig_xi * jnp.cumsum(eta)])
    p = jax.nn.softmax(xi)
    eps = numpyro.sample("eps", dist.Normal(0.0, 1.0).expand([W]))
    l0 = numpyro.sample("l0", dist.Normal(log_mean0, 2.0))
    loglam = l0 + tau * jnp.cumsum(eps)
    mean = jnp.exp(loglam)[:, None] * p[None, :] + 1e-6
    with numpyro.handlers.mask(mask=obs):
        numpyro.sample("n", dist.NegativeBinomial2(mean, r), obs=inc)
    numpyro.deterministic("lam", jnp.exp(loglam))
    numpyro.deterministic("p", p)


def _get_sampler(infer, W, D, warmup, samples):

    key = (infer, W, D, warmup, samples)
    if key in _CACHE:
        return _CACHE[key]
    import jax, numpyro
    from functools import partial
    from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive, init_to_median
    from numpyro.infer.autoguide import AutoNormal
    numpyro.set_platform("cpu")
    model = partial(_model, W=W, D=D)
    if infer == "nuts":
        kern = NUTS(model, target_accept_prob=0.8, max_tree_depth=7)
        mc = MCMC(kern, num_warmup=warmup, num_samples=samples, num_chains=1, progress_bar=False, jit_model_args=True)
        obj = {"mc": mc}
    else:
        guide = AutoNormal(model, init_loc_fn=init_to_median(num_samples=15), init_scale=0.05)
        svi = SVI(model, guide, numpyro.optim.ClippedAdam(0.01, clip_norm=10.0), Trace_ELBO(num_particles=1))

        def run(key, inc, obs, log_mean0):
            st = svi.init(key, inc, obs, log_mean0)
            st, _ = jax.lax.scan(lambda s_, _: (svi.update(s_, inc, obs, log_mean0)[0], None), st, None, length=warmup)
            return svi.get_params(st)
        obj = {"run": jax.jit(run), "guide": guide, "model": model}
    _CACHE[key] = obj
    return obj


def fit_one(args):

    try:
        return _fit_one(args)
    except Exception as e:
        L = args[3]
        return np.full((L, len(LEVELS)), np.nan), {"fallback": True, "n_obs": -1, "error": repr(e)[:200]}


def _fit_one(args):

    A, R, t, L, W, warmup, samples, seed = args[:8]
    infer = args[8] if len(args) > 8 else "nuts"
    import jax, jax.numpy as jnp
    T = A.shape[0]
    D = L
    u = np.arange(t - W + 1, t + 1)
    valid = (u >= 0) & (u < T)
    uu = np.clip(u, 0, T - 1)
    Rv = np.where(np.isfinite(R), R, np.inf)[uu]

    Av = A[uu].astype(np.float64)
    a_vis = np.minimum(t - u, D)
    inc = np.zeros((W, D + 1)); obs = np.zeros((W, D + 1), bool)
    prev = np.zeros(W)
    for d in range(D + 1):
        col = Av[:, d]
        ok = valid & (d <= a_vis) & (Rv <= u + d) & np.isfinite(col) & (Rv <= u)
        inc[:, d] = np.where(ok, np.maximum(col - prev, 0.0), 0.0)
        obs[:, d] = ok
        prev = np.where(ok, col, prev)
    inc = np.round(inc)
    n_obs_periods = int((obs.sum(1) > 0).sum())
    fallback = n_obs_periods < 2 * min(L, 8) + 4
    q_now = np.full((L, len(LEVELS)), np.nan)
    if fallback:
        return q_now, {"fallback": True, "n_obs": n_obs_periods}
    tot_obs = inc.sum(1)
    log_mean0 = float(np.log(max(tot_obs[obs.sum(1) > 0].mean(), 1.0)))
    inc_j = jnp.asarray(inc, dtype=jnp.float32); obs_j = jnp.asarray(obs); lm0 = jnp.float32(log_mean0)
    sampler = _get_sampler(infer, W, D, warmup, samples)
    if infer == "nuts":
        mc = sampler["mc"]
        mc.run(jax.random.PRNGKey(seed), inc_j, obs_j, lm0)
        sm = mc.get_samples()
    else:
        from numpyro.infer import Predictive
        params = sampler["run"](jax.random.PRNGKey(seed), inc_j, obs_j, lm0)
        sm = Predictive(sampler["model"], guide=sampler["guide"], params=params, num_samples=samples, return_sites=["lam", "p", "r", "tau"])(jax.random.PRNGKey(seed + 1), inc_j, obs_j, lm0)
    lam = np.asarray(sm["lam"]); p = np.asarray(sm["p"]); r = np.asarray(sm["r"])
    if not (np.isfinite(lam).all() and np.isfinite(p).all() and np.isfinite(r).all()):
        return q_now, {"fallback": True, "n_obs": n_obs_periods, "nonfinite": True}
    rng = np.random.default_rng(seed)

    for k, pos in enumerate(range(t - L + 1, t + 1)):
        j = pos - (t - W + 1)
        if j < 0 or not valid[j]:
            continue
        unobs = ~obs[j]
        if Rv[j] > pos:
            unobs = np.ones(D + 1, bool)
        m = np.minimum(lam[:, j][:, None] * p[:, unobs], 1e12)
        rr = r[:, None]

        g = np.minimum(rng.gamma(np.broadcast_to(rr, m.shape), m / rr), 1e12)
        big = g > 1e8
        draws = np.where(big, np.maximum(g + rng.standard_normal(g.shape) * np.sqrt(g), 0.0), rng.poisson(np.where(big, 1.0, g))).sum(1)
        tot = inc[j, obs[j]].sum() + draws
        q_now[k] = np.quantile(tot, LEVELS)
    diag = {"fallback": False, "n_obs": n_obs_periods, "tau": float(np.mean(sm["tau"])), "r": float(np.mean(r)),
            "mean_delay": float(np.mean(p @ np.arange(D + 1)))}
    return q_now, diag


if __name__ == "__main__":
    import multiprocessing as mp
    from recast.datasets import load_bundle
    from recast.protocol import make_tasks, metrics, DEFAULTS
    from recast.views import as_of_series
    from recast.synth import nowcast_window
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", default="results/rrbench")
    ap.add_argument("--tag", default="bayes")
    ap.add_argument("--max_origins", type=int, default=None)
    ap.add_argument("--W", type=int, default=None, help="moving window (periods); default 1.5 L + 8, capped at the context")
    ap.add_argument("--warmup", type=int, default=200, help="NUTS warm-up iterations, or SVI optimisation steps")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--stage", default="all", help="fit | tsfm | all")
    ap.add_argument("--arch", default="chronos2")
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first n origins")
    ap.add_argument("--infer", default="nuts", help="nuts | svi (mean-field VI; --warmup = optimisation steps)")
    ap.add_argument("--subsample", type=int, default=1, help="use every k-th origin of the protocol (limited-scope MCMC runs)")
    ap.add_argument("--refit", action="store_true", help="stage fit: only re-fit origins whose first fit failed or diverged (tau > --tau_max), with a new seed")
    ap.add_argument("--tau_max", type=float, default=1.0, help="posterior-mean random-walk scale above which a fit is treated as divergent")
    ap.add_argument("--delay_max_frac", type=float, default=0.5, help="fits whose posterior mean delay exceeds this fraction of L (mass in the unobserved tail) are treated as divergent")
    args = ap.parse_args()

    b = load_bundle(args.domain)
    freq, L = b["freq"], b["L"]
    d = DEFAULTS[freq]; C, H = d["C"], d["H"]
    N = nowcast_window(freq, L)
    W = args.W or int(min(C, 1.2 * L + 8 if freq in ("D", "h") else max(2 * L + 8, 40)))
    tasks = make_tasks(b, max_origins=args.max_origins)
    index, Y, S, EV, YN, jobs = [], [], [], [], [], []
    for tk in tasks:
        y = tk.targets(); sc = tk.scale(); ev = tk.eval_mask()
        for i, t in enumerate(tk.origins):
            t = int(t)
            index.append((tk.sid, t)); Y.append(y[i]); S.append(sc[i]); EV.append(bool(ev[i]))
            u = np.arange(t - N + 1, t + 1)
            yn = np.full(N, np.nan, np.float32); ok = (u >= 0) & (u < tk.A.shape[0]); yn[ok] = tk.A[u[ok], L]; YN.append(yn)
            jobs.append((tk.A, tk.R, t, L, W, args.warmup, args.samples, 1000 * len(jobs) + 7, args.infer))
    if args.subsample > 1:
        sel = list(range(0, len(index), args.subsample))
        index, Y, S, EV, YN, jobs = [index[i] for i in sel], [Y[i] for i in sel], [S[i] for i in sel], [EV[i] for i in sel], [YN[i] for i in sel], [jobs[i] for i in sel]
    if args.limit:
        index, Y, S, EV, YN, jobs = index[:args.limit], Y[:args.limit], S[:args.limit], EV[:args.limit], YN[:args.limit], jobs[:args.limit]
    Y = np.stack(Y); S = np.asarray(S); EV = np.asarray(EV); YN = np.stack(YN)
    n = len(index)
    os.makedirs(args.out, exist_ok=True)
    fit_path = f"{args.out}/{args.tag}__{args.domain}.fit.npz"
    print(f"domain {args.domain} L {L} N {N} W {W} origins {n} eval {EV.sum()}", flush=True)
    if args.stage in ("fit", "all") and args.refit:
        z = np.load(fit_path, allow_pickle=True); Qn_full = z["Qn_full"]; diags = json.load(open(fit_path.replace(".npz", ".json")))
        sel = [i for i, dg in enumerate(diags) if dg.get("fallback") or dg.get("tau", 0.0) > args.tau_max]
        print(f"re-fitting {len(sel)} of {n} origins (failed or tau > {args.tau_max})", flush=True)
        jobs2 = [tuple(list(jobs[i][:7]) + [jobs[i][7] + 11, jobs[i][8]]) for i in sel]
        t0 = time.time()
        os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["OMP_NUM_THREADS"] = "1"
        ctx = mp.get_context("spawn"); counter = ctx.Value("i", 0); n_cores = os.cpu_count() or 1
        with ctx.Pool(min(args.workers, max(1, len(sel))), initializer=_init_worker, initargs=(counter, max(1, n_cores - 8), 8)) as pool:
            for k, (q, dg) in enumerate(pool.imap(fit_one, jobs2, chunksize=1)):
                i = sel[k]
                if not dg.get("fallback") and dg.get("tau", 0.0) <= args.tau_max:
                    Qn_full[i] = q; diags[i] = dict(dg, refit=True)
                else:
                    diags[i] = dict(diags[i], refit_failed=True)
                if (k + 1) % 50 == 0:
                    print(f"  {k+1}/{len(sel)} refits {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(fit_path, Qn_full=Qn_full, index_sid=np.array([i[0] for i in index]), index_t=np.array([i[1] for i in index]))
        json.dump(diags, open(fit_path.replace(".npz", ".json"), "w"))
        print(f"refits done {time.time()-t0:.0f}s; still divergent/failed: {sum(1 for d_ in diags if d_.get('fallback') or d_.get('tau', 0.0) > args.tau_max)}", flush=True)
    elif args.stage in ("fit", "all"):
        t0 = time.time()
        Qn_full = np.full((n, L, 9), np.nan, np.float32); diags = []
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["OMP_NUM_THREADS"] = "1"
        ctx = mp.get_context("spawn")
        counter = ctx.Value("i", 0)
        n_cores = os.cpu_count() or 1
        with ctx.Pool(min(args.workers, n), initializer=_init_worker, initargs=(counter, max(1, n_cores - 8), 8), maxtasksperchild=40) as pool:
            for i, (q, dg) in enumerate(pool.imap(fit_one, jobs, chunksize=1)):
                Qn_full[i] = q; diags.append(dg)
                if (i + 1) % 100 == 0 or i + 1 == n:
                    el = time.time() - t0
                    print(f"  {i+1}/{n} fits {el:.0f}s ({el/(i+1):.2f}s/fit effective) fallback {sum(d_['fallback'] for d_ in diags)}", flush=True)
                if (i + 1) % 500 == 0:
                    np.savez_compressed(fit_path, Qn_full=Qn_full, index_sid=np.array([i_[0] for i_ in index]), index_t=np.array([i_[1] for i_ in index]), n_done=i + 1)
                    json.dump(diags, open(fit_path.replace(".npz", ".json"), "w"))
        np.savez_compressed(fit_path, Qn_full=Qn_full, index_sid=np.array([i[0] for i in index]), index_t=np.array([i[1] for i in index]))
        json.dump(diags, open(fit_path.replace(".npz", ".json"), "w"))
        print(f"fits done {time.time()-t0:.0f}s, saved {fit_path}", flush=True)
    if args.stage in ("tsfm", "all"):
        import torch
        z = np.load(fit_path, allow_pickle=True); Qn_full = z["Qn_full"].copy()
        diags = json.load(open(fit_path.replace(".npz", ".json")))
        n_div = 0
        for i, dg in enumerate(diags):
            if dg.get("tau", 0.0) > args.tau_max or dg.get("mean_delay", 0.0) > args.delay_max_frac * L:
                Qn_full[i] = np.nan; n_div += 1
        print(f"divergent fits treated as failed (tau > {args.tau_max} or mean delay > {args.delay_max_frac} L): {n_div}", flush=True)

        Qn = np.full((n, N, 9), np.nan, np.float32)
        ctx_med, ctx_mc = [], [[] for _ in range(5)]
        n_fb = 0
        tk_by = {tk.sid: tk for tk in tasks}
        for i, (sid, t) in enumerate(index):
            tk = tk_by[sid]
            x = as_of_series(tk.A, tk.R, t, L, C)
            last = x[np.isfinite(x)][-1] if np.isfinite(x).any() else np.nan
            xn = np.full(L, np.nan, np.float32)
            for k in range(L):
                j = C - L + k
                v = x[j] if 0 <= j < C else np.nan
                xn[k] = v if np.isfinite(v) else (xn[k - 1] if k > 0 and np.isfinite(xn[k - 1]) else last)
            qf = Qn_full[i].copy()
            miss = ~np.isfinite(qf[:, 4])
            n_fb += int(miss.all())
            qf[miss] = xn[miss, None]
            qf = np.sort(qf, axis=-1)
            Qn[i] = qf[L - N:]
            xr = x.copy()
            for k in range(L):
                j = C - L + k
                if 0 <= j < C and np.isfinite(qf[k, 4]) and np.isfinite(xr[j]):
                    xr[j] = qf[k, 4]
            ctx_med.append(xr)
            for kk, tau_i in enumerate([0, 2, 4, 6, 8]):
                xk = x.copy()
                for k in range(L):
                    j = C - L + k
                    if 0 <= j < C and np.isfinite(qf[k, tau_i]) and np.isfinite(xk[j]):
                        xk[j] = qf[k, tau_i]
                ctx_mc[kk].append(xk)
        print(f"origins without a fit (latest report used): {n_fb}", flush=True)
        dev = "cuda"
        if args.arch == "chronos2":
            from recast.models.va_chronos2 import VAChronos2
            model = VAChronos2("models/chronos2", age_channel=False, age_bias=False, shared_norm=False).to(dev).eval()
            QIDX = [2, 4, 6, 8, 10, 12, 14, 16, 18]

            def predict_plain(ctxs, bs=64):
                out = np.full((len(ctxs), H, 9), np.nan, np.float32)
                with torch.no_grad():
                    for s in range(0, len(ctxs), bs):
                        X = np.stack(ctxs[s: s + bs]).astype(np.float32); Bn = X.shape[0]
                        vals = torch.tensor(np.concatenate([X, np.full((Bn, H), np.nan, np.float32)], 1))[:, None, :].to(dev)
                        mask = torch.isfinite(vals).float(); age = torch.zeros_like(vals)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            o = model(vals, mask, age, torch.zeros(Bn, 1, device=dev), torch.full((Bn,), L, device=dev), ctx_len=C, target_row=0)
                        q = o["quantiles"].float().cpu().numpy()[:, QIDX, :].transpose(0, 2, 1)
                        out[s: s + Bn] = np.sort(q, axis=-1)
                return out
        else:
            from recast.plain import Plain
            plain = Plain(args.arch)
            predict_plain = lambda ctxs: plain.predict(ctxs, H)
        Qf_2s = predict_plain(ctx_med)
        Qmc = np.stack([predict_plain(c) for c in ctx_mc])
        samp = Qmc.transpose(1, 2, 0, 3).reshape(n, H, 45)
        Qf_mc = np.quantile(samp, LEVELS, axis=-1).transpose(1, 2, 0).astype(np.float32)
        res = dict(domain=args.domain, freq=freq, L=L, N=N, H=H, W=W, n=n, n_eval=int(EV.sum()), n_fallback=n_fb, n_divergent=n_div, n_error=sum(1 for d_ in diags if "error" in d_), n_refit=sum(1 for d_ in diags if d_.get("refit")),
                   warmup=args.warmup, samples=args.samples, infer=args.infer, subsample=args.subsample, tau_max=args.tau_max, delay_max_frac=args.delay_max_frac)
        res["nowcast_bayes"] = metrics(Qn[EV], YN[EV], S[EV])
        res["forecast_2s_point"] = metrics(Qf_2s[EV], Y[EV], S[EV])
        res["forecast_2s_mc"] = metrics(Qf_mc[EV], Y[EV], S[EV])
        for k, v in res.items():
            if isinstance(v, dict):
                print(k, {kk: round(vv, 4) for kk, vv in v.items() if not isinstance(vv, list)})
        json.dump(res, open(f"{args.out}/{args.tag}__{args.domain}.json", "w"), indent=1)
        np.savez_compressed(f"{args.out}/{args.tag}__{args.domain}.npz", Qn_bayes=Qn, Qn_full=Qn_full, Qf_2s=Qf_2s, Qf_mc=Qf_mc, Y=Y, YN=YN, S=S, EV=EV,
                            index_sid=np.array([i[0] for i in index]), index_t=np.array([i[1] for i in index]))
        print("saved", f"{args.out}/{args.tag}__{args.domain}.npz")
