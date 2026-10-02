#!/bin/bash
cd "$(dirname "$0")/.."
EX="--evidence_gate --separate_now_head --forecast_anchor --mult_head --mc_anchor"
for C in 96 128 192; do
  for d in kit dv; do
    echo "== RECAST C=$C $d"; python scripts/eval_rrbench.py --domain $d --arch chronos2 --model checkpoints/v12_main.pt --tag v12_C$C $EX --max_origins 200 --C $C 2>&1 | grep -E '^forecast|^nowcast' | cut -c1-110
    echo "== CL C=$C $d"; python scripts/eval_baselines.py --domain $d --max_origins 200 --C $C --tag baselines_C$C 2>&1 | grep -E 'nowcast_cl|forecast_2s_point' | cut -c1-110
  done
done
echo DATAEFF_DONE
