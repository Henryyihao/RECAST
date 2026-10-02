#!/bin/bash
cd "$(dirname "$0")/.."
while ! grep -q "END v12_main" logs/pool_gpu0.log; do sleep 30; done
DOMS="kit dv chng_flu hosp_cov respinow nssp nhsn macro_m rtdsm_q alfred_w eia930"
for d in $DOMS; do ln -sf v12_main__$d.json results/rrbench/v12_final__$d.json; ln -sf v12_main__$d.npz results/rrbench/v12_final__$d.npz; done
EX="--evidence_gate --separate_now_head --forecast_anchor --mult_head --mc_anchor"
export CUDA_VISIBLE_DEVICES=0
echo "== failure diagnostics $(date)"
python scripts/diag_failure.py --domain alfred_w --model checkpoints/v12_main.pt --tag v12_final --out results/diag_failure_alfred_w.json > logs/diag_failure.log 2>&1
python scripts/diag_failure.py --domain rtdsm_q --model checkpoints/v12_main.pt --tag v12_final --out results/diag_failure_rtdsm_q.json >> logs/diag_failure.log 2>&1
echo "== sensitivity $(date)"
python scripts/sens_eval.py --model checkpoints/v12_main.pt --prefix v12 --group all --gpu 0 > logs/sens_v12.log 2>&1
echo "== mechanism $(date)"
python scripts/mech_synth.py --model checkpoints/v12_main.pt --n 300 $EX > logs/mech_v12.log 2>&1
echo "== data efficiency $(date)"
for C in 96 128 192; do for d in kit dv; do python scripts/eval_rrbench.py --domain $d --arch chronos2 --model checkpoints/v12_main.pt --tag v12_C$C $EX --max_origins 200 --C $C 2>&1 | grep -E '^forecast|^nowcast' | cut -c1-110; done; done > logs/dataeff_v12.log 2>&1
echo "== two-stage baselines for the other backbones $(date)"
for a in bolt_s bolt_b toto timemoe; do for d in $DOMS; do
  MO=""; case $d in dv|chng_flu|hosp_cov|kit) MO="--max_origins 200";; eia930) MO="--max_origins 120";; esac
  [ -f results/rrbench/baselines_${a}__$d.json ] || python scripts/eval_baselines.py --domain $d --arch $a $MO 2>&1 | grep -E 'forecast_2s|Error|error' | cut -c1-120
done; done > logs/baselines_bb.log 2>&1
echo "POST_V12_DONE $(date)"
