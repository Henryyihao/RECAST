#!/bin/bash
cd "$(dirname "$0")/.."
run() { d=$1; MO=""; case $d in dv|chng_flu|hosp_cov|kit) MO="--max_origins 200";; eia930) MO="--max_origins 120";; esac
  [ -f results/rrbench/clctx__$d.npz ] || CUDA_VISIBLE_DEVICES='' python scripts/eval_baselines.py --domain $d --arch none $MO > logs/cache_cl_$d.log 2>&1; }
for d in nssp nhsn macro_m dv rtdsm_q respinow; do run $d & done; wait
for d in kit chng_flu hosp_cov alfred_w eia930; do run $d & done; wait
echo CACHE_DONE
