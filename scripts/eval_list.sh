#!/bin/bash
cd "$(dirname "$0")/.."
while read -r arch ck tag extra; do
  [ -z "$arch" ] && continue; case "$arch" in \#*) continue;; esac
  if [ -f results/rrbench/${tag}__eia930.json ]; then echo "skip $tag"; continue; fi
  echo "=== eval $tag $(date)"
  python scripts/eval_all.py --arch $arch --model $ck --tag $tag --extra "$extra" > logs/eval_${tag}.log 2>&1
  tail -14 logs/eval_${tag}.log
done < "$1"
echo EVAL_LIST_DONE
