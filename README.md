# RECAST — Revision-Aware Adaptation of Time-Series Foundation Models

Reference implementation of **RECAST** (*Read the Version Triangle: Revision-Aware
Adaptation of Time-Series Foundation Models*).

Operational series are revised after first release, so the information available at a
forecast origin is a **version triangle**, while time-series foundation models (TSFMs)
are trained on settled histories. RECAST reads the visible reporting history, re-indexes
the triangle into time-aligned **age views**, augments a frozen backbone with
age-encoded inputs, cross-age attention and a zero-initialised gated edge head, and
anchors forecasts to the frozen backbone on nowcast-repaired histories. Adaptation
training combines a compositional prior over reporting processes with settled-context
distillation and is zero-shot: no evaluation domain is seen during training.

The code covers training, all baselines (naive / oracle / Views / chain ladder / B2F /
residual adapter / Bayesian nowcaster), evaluation over the eleven domains, aggregation,
the ablation and sensitivity studies, and the figure / table pipeline.

---

## 1. Repository layout

```
recast/                 core package
  vintage.py            version-matrix / age-view construction and release indices
  protocol.py           evaluation protocol: origins, tasks, metrics, scaling
  datasets.py           raw-archive readers and bundle builders (one per domain)
  views.py              age-view stream construction, as-of / settled contexts
  synth.py              synthetic reporting-process prior (delays, modifiers)
  data.py               training batches sampled from the synthetic prior
  anchor.py             repaired-history anchor and quantile combination
  baselines.py          chain ladder and repaired-context utilities
  outputside.py         output-side baselines (B2F, residual adapters)
  plain.py              frozen backbones without adaptation
  registry.py           backbone registry (paths, quantile indices, model builder)
  models/               one adapted model per backbone
scripts/                training, evaluation, aggregation, tables, export drivers
R/                      figure and paired-bootstrap pipeline (ggplot2)
configs/                training queues and evaluation lists
```

## 2. Environment

```bash
pip install -r requirements.txt
```

The figure pipeline additionally needs R (>= 4.2) with `ggplot2`, `dplyr`, `tidyr`,
`patchwork`, `svglite` and `scales`:

```bash
Rscript -e 'install.packages(c("ggplot2","dplyr","tidyr","patchwork","svglite","scales"))'
```

A CUDA GPU is required for training and for the TSFM baselines.

## 3. Directory setup

All scripts resolve paths relative to the repository root, so run them from there
(`python scripts/<name>.py`). The directories below are created on demand and are
excluded by `.gitignore`:

```
models/       backbone weights (step 4)
data/raw/     downloaded vintage archives (step 5)
data/bundles/ built domain bundles (step 6)
data/         synthetic prior pools (step 7)
checkpoints/  trained adapters
results/      evaluation outputs, aggregates, tables
logs/         training and evaluation logs
```

## 4. Backbone weights -> `models/`

All weights are Apache-2.0. Dump each snapshot into the directory named below, e.g.
`huggingface-cli download <repo> --local-dir models/<dir>`:

| Key | Directory | Source |
|---|---|---|
| `chronos2` (primary) | `models/chronos2` | https://huggingface.co/amazon/chronos-2 |
| `bolt_s` | `models/chronosbolt` | https://huggingface.co/amazon/chronos-bolt-small |
| `bolt_b` | `models/amazon_chronos-bolt-base` | https://huggingface.co/amazon/chronos-bolt-base |
| `toto` | `models/Datadog_Toto-2.0-313m` | https://huggingface.co/Datadog/Toto-2.0-313m |
| `timemoe` | `models/Maple728_TimeMoE-50M` | https://huggingface.co/Maple728/TimeMoE-50M |

## 5. Domain data -> `data/raw/`

```bash
bash scripts/download/download_raw.sh          # KIT, RESPINOW, Philadelphia Fed, EIA-930
python scripts/download/dl_v4_server.py       # Delphi COVIDcast daily (US-CLI / US-Flu / US-Hosp)
python scripts/download/dl_v5_server.py       # Delphi COVIDcast weekly (US-NHSN / US-NSSP)
python scripts/download/dl_alfred.py          # ALFRED vintages (US-UI and monthly macro)
```

The eleven domains and their public archives:

| Domain | Key | Grid | Source / download link |
|---|---|---|---|
| DE-Hosp (German COVID-19 hospitalisations) | `kit` | daily | KIT hospitalization-nowcast-hub, `data-truth/COVID-19/COVID-19_hospitalizations_preprocessed.csv` — https://github.com/KITmetricslab/hospitalization-nowcast-hub |
| US-CLI (claims-based CLI) | `dv` | daily | Delphi COVIDcast `doctor-visits / smoothed_adj_cli` — https://api.delphi.cmu.edu/epidata/covidcast/ |
| US-Flu (outpatient flu) | `chng_flu` | daily | Delphi COVIDcast `chng / smoothed_adj_outpatient_flu` |
| US-Hosp (claims-based COVID admissions) | `hosp_cov` | daily | Delphi COVIDcast `hospital-admissions / smoothed_adj_covid19_from_claims` |
| DE-RESP (German respiratory surveillance) | `respinow` | weekly | RESPINOW-Hub reporting triangles — https://github.com/KITmetricslab/RESPINOW-Hub |
| US-NSSP (% ED visits) | `nssp` | weekly | Delphi COVIDcast weekly `nssp/*_ew` (see `dl_v5_server.py`) |
| US-NHSN (hospital admissions) | `nhsn` | weekly | Delphi COVIDcast weekly `nhsn/*_ew` |
| US-Macro-M (monthly macro vintages) | `macro_m` | monthly | Philadelphia Fed RTDSM `*mvmd.xlsx` — https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/real-time-data-set-for-macroeconomists |
| US-Macro-Q (quarterly macro vintages) | `rtdsm_q` | quarterly | Philadelphia Fed RTDSM `*qvqd.xlsx` (same page) |
| US-UI (unemployment insurance claims) | `alfred_w` | weekly | ALFRED vintages (`ICSA`, `CCSA`) via `https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=<ID>&vintage_date=<YYYY-MM-DD>` |
| US-Grid (hourly grid load, raw -> adjusted) | `eia930` | hourly | EIA-930 Balance files `sixMonthFiles/EIA930_BALANCE_<year>_<Jan_Jun or Jul_Dec>.csv` — https://www.eia.gov/electricity/gridmonitor/ |

File placement inside `data/raw/`:

```
data/raw/kit/        COVID-19_hospitalizations_preprocessed.csv, COVID-19_hospitalizations.csv
data/raw/delphi_v4/  <source>__<signal>__<state>.csv.gz
data/raw/delphi_v5/  <source>__<signal>__state.csv.gz
data/raw/respinow/   reporting_triangle-<source>.csv, target-<source>.csv
data/raw/rtdsm/      <id>mvmd.xlsx, <id>qvqd.xlsx
data/raw/alfred/     <SERIES_ID>.parquet
data/raw/eia930/     EIA930_BALANCE_<year>_<half>.csv
```

## 6. Build domain bundles -> `data/bundles/`

```bash
python scripts/build_bundles.py                 # all domains
python scripts/build_bundles.py kit dv eia930   # a subset
```

Each bundle is written as `data/bundles/<name>.npz` + `.json`, where
`A[u, a]` is the value of reference position `u` as it was known `a` steps after its
release, `R[u]` the release index, plus `time`, `freq` and `L`. `macro_m` merges RTDSM-M
with the ALFRED monthly vintages.

## 7. Synthetic prior pool -> `data/`

Training samples base signals from a synthetic prior (GP / renewal / seasonal ARMA),
optionally grounded in real, non-benchmark series:

```bash
python scripts/build_pool.py          # -> data/pool.npz        (chronos_datasets via ModelScope)
python scripts/build_pool_local.py    # -> data/pool_local.npz  (optional local hourly sources)
```

`scripts/train.py` expects the concatenation at `data/pool_all.npz`; build it by merging
the available pools on the `series` / `freq` / `name` arrays, e.g.

```bash
python - <<'PY'
import numpy as np
parts = [np.load(p, allow_pickle=True) for p in ("data/pool.npz", "data/pool_local.npz")]
np.savez_compressed("data/pool_all.npz",
                    series=np.concatenate([p["series"] for p in parts]),
                    freq=np.concatenate([p["freq"] for p in parts]),
                    name=np.concatenate([p["name"] for p in parts]))
PY
```

## 8. Train RECAST

The full recipe for the primary Chronos-2 model is the `v12_main` line of
`configs/queue_v12.txt`:

```bash
python scripts/train.py --name v12_main \
  --steps 25000 --workers 16 \
  --distill 1.0 --distill_pre 1.0 --now_aux 1.0 --gate_sup 0.5 --w_true 0.0 \
  --lr 2e-5 --lr_new 1e-4 --p_real 0.85 \
  --freeze_old_cols --freeze_output_head \
  --evidence_gate --separate_now_head --forecast_anchor --mult_head
```

Checkpoints are written to `checkpoints/<name>.pt` and logs to
`logs/train_<name>.log`; the run prints `DONE` on success. The other four backbones use
the same objective with `--arch bolt_s|bolt_b|toto|timemoe` and a smaller
`--token_budget` where the backbone is memory heavy (`toto` 8000, `timemoe` 2500).

Queue runners claim lines of a queue file through lock directories, one process per GPU:

```bash
bash scripts/pool.sh configs/queue_v12.txt 0
```

## 9. Evaluate

```bash
python scripts/eval_rrbench.py --domain kit --arch chronos2 \
  --model checkpoints/v12_main.pt --tag v12_final \
  --evidence_gate --separate_now_head --forecast_anchor --mult_head --mc_anchor
```

`--model` also accepts the plain references `zeroshot | naive | oracle`. All domains for
one model:

```bash
python scripts/eval_all.py --arch chronos2 --model checkpoints/v12_main.pt --tag v12_final \
  --extra "--evidence_gate --separate_now_head --forecast_anchor --mult_head --mc_anchor"
```

Results are written to `results/rrbench/<tag>__<domain>.{json,npz}` (quantiles, targets,
scales, evaluation masks and per-origin indices). `configs/eval_abl.txt` and
`configs/eval_bb.txt` list the ablation and per-backbone checkpoints for
`scripts/eval_list.sh`.

## 10. Baselines

```bash
bash scripts/plain.sh                                            # naive / oracle for all five backbones
python scripts/eval_baselines.py --domain kit                    # chain ladder + two-stage CL -> TSFM
python scripts/eval_outputside.py --domain kit --arch chronos2   # B2F and residual adapters
python scripts/eval_bayes.py --domain kit                        # hierarchical Bayesian nowcaster (NUTS)
```

`eval_bayes.py` needs `jax` / `numpyro` and is defined on the count domains (DE-Hosp,
US-NHSN). `scripts/cache_cl.sh` pre-computes the backbone-independent chain-ladder
contexts.

## 11. Aggregation, tables and numbers

```bash
python scripts/aggregate.py --tags v12_final,abl2_light,chronos2_zsviews_psn \
  --arch chronos2 --no_bayes --out results/agg_v12_final.json
```

`aggregate.py` aligns every method on identical scored cells and runs paired t-tests and
Wilcoxon tests against the naive baseline. `bash scripts/final_assemble_v12.sh` runs the
full pipeline: aggregation for main, ablations, sensitivity and backbones, the LaTeX
tables (`make_tables*.py`) and the macro file `results/tables/numbers.tex` consumed by
the manuscript.

## 12. Ablations, sensitivity and mechanism studies

```bash
bash scripts/pool.sh configs/queue_abl.txt 1        # ablation queue (7.5k steps each)
bash scripts/post_v12.sh                            # diagnostics, sensitivity, mechanism, two-stage baselines
python scripts/sens_eval.py --model checkpoints/v12_main.pt --prefix v12 --group all
python scripts/mech_synth.py --model checkpoints/v12_main.pt --n 300 \
  --evidence_gate --separate_now_head --forecast_anchor --mult_head
python scripts/diag_failure.py --domain alfred_w --model checkpoints/v12_main.pt \
  --tag v12_final --out results/diag_failure_alfred_w.json
bash scripts/dataeff.sh                             # context-length study
bash scripts/timing_views.sh 0                      # inference timing over view counts
```

`configs/queue_abl.txt`, `configs/queue_v12.txt` and `configs/eval_abl.txt` hold the exact
ablation recipes; `configs/eval_bb.txt` the four per-backbone checkpoints.

## 13. Figures and resampling analysis

Non-visual export of the archived artefacts to CSV, then R:

```bash
python scripts/server_data_export.py --out exports/figures --stage all
python scripts/export_nowcast_bootstrap.py --dir exports/figures
python scripts/make_bootstrap_inputs.py --dir exports/figures
Rscript R/make_figures.R exports/figures results/figures
Rscript R/paired_bootstrap.R --data-dir=exports/figures --out-dir=results/analysis \
        --source-dir=results/source_data
```

`bash scripts/run_figures.sh` chains all five steps. `make_figures.R` writes PDF, PNG, SVG
and TIFF plus the plotted source CSVs; `paired_bootstrap.R` runs the crossed
series x time-block resampling for the forecast and nowcast comparisons against the
naive baseline, the chain-ladder pipeline and the chain ladder.

## 14. Evaluation protocol

| Grid | `C` | `H` | `N_max` | stride | season |
|---|---|---|---|---|---|
| Daily | 256 | 28 | 28 | 7 | 7 |
| Weekly | 104 | 8 | 12 | 1 | 52 |
| Monthly | 120 | 12 | 6 | 1 | 12 |
| Quarterly | 60 | 4 | 4 | 1 | 4 |
| Hourly | 336 | 48 | 24 | 48 | 24 |

Metrics: MASE of the predictive median, scaled CRPS (2/9 of the pinball sum at levels
0.1 ... 0.9) and central 80% coverage, all scaled by the seasonal-naive absolute error
over the preceding `2C` settled values. Heavy-revision domains: DE-Hosp, US-CLI, US-Flu,
US-Hosp, US-Grid; the other six form the light-revision group. Daily and hourly series use
200 and 120 evenly spaced origins. Forecast gap closure is
`(CRPS_naive - CRPS_method) / (CRPS_naive - CRPS_oracle)`, averaged over the domains whose
naive-to-oracle gap exceeds 2%.

## 15. Notes

* `--mc_anchor` propagates the nowcast uncertainty through the anchor using three
  quantile paths; `--anchor_mode paths|mc|moment` selects the scheme directly.
* Data-efficiency evaluation (`--C 96|128|192`) re-runs a frozen checkpoint with a
  shortened context and does not retrain.
* `scripts/build_bundles.py` also builds the accessory variants (`kit_age7`,
  `kit_L40/L60`, `dv_L40`, `nssp_*`, `nhsn_*`, `alfred_m`) used by the domain table and
  the settlement-age analysis.
