#!/bin/bash
cd "$(dirname "$0")/.."
for a in chronos2 bolt_s bolt_b toto timemoe; do for mode in naive oracle; do python scripts/eval_all.py --arch $a --model $mode --tag ${a}_${mode}; done; done
echo PLAIN_DONE
