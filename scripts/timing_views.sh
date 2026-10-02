#!/bin/bash
cd "$(dirname "$0")/.."
EX="--evidence_gate --separate_now_head --forecast_anchor --mult_head --mc_anchor"
for K in 16 8 4 2; do for d in kit dv chng_flu hosp_cov respinow nssp nhsn macro_m rtdsm_q alfred_w eia930; do
  MO=""; case $d in dv|chng_flu|hosp_cov|kit) MO="--max_origins 200";; eia930) MO="--max_origins 120";; esac
  CUDA_VISIBLE_DEVICES=$1 python scripts/eval_rrbench.py --domain $d --arch chronos2 --model checkpoints/v12_main.pt --tag v12t_views$K $MO $EX --n_views $K 2>&1 | grep -E '^saved' | cut -c1-100
done; done
echo TIMING_DONE
