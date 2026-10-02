#!/bin/bash
cd "$(dirname "$0")/.."
Q=$1; GPU=$2
mkdir -p locks
while true; do
  claimed=""
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in \#*) continue;; esac
    name=$(echo "$line" | cut -d'|' -f1 | xargs)
    targs=$(echo "$line" | cut -d'|' -f2)
    eargs=$(echo "$line" | cut -d'|' -f3)
    if mkdir locks/$name 2>/dev/null; then claimed=$name; break; fi
  done < "$Q"
  [ -z "$claimed" ] && break
  echo "=== START $claimed gpu$GPU $(date) ==="
  if ! ( [ -f checkpoints/${claimed}.pt ] && grep -q DONE logs/train_${claimed}.log 2>/dev/null ); then
    CUDA_VISIBLE_DEVICES=$GPU python scripts/train.py --name $claimed $targs > logs/${claimed}.out 2>&1
  fi
  if grep -q DONE logs/train_${claimed}.log 2>/dev/null; then
    CUDA_VISIBLE_DEVICES=$GPU python scripts/eval_all.py --arch chronos2 --model checkpoints/${claimed}.pt --tag $claimed --extra "$eargs" > logs/eval_${claimed}.log 2>&1
  else
    echo "TRAIN FAILED $claimed"
  fi
  echo "=== END $claimed gpu$GPU $(date) ==="
done
echo "POOL DONE $Q gpu$GPU $(date)"
