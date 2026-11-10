#!/bin/zsh
# usage: batch_xa.sh <mode> <file with lines: tag<TAB>specjson>
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
while IFS=$'\t' read -r tag spec; do
  [ -z "$tag" ] && continue
  [ -f "reports/phase15/xa_$1_$tag.json" ] && continue
  PYTHONPATH=. data/ibl-venv/bin/python scripts/phase15/run_xanimal.py $1 "$spec" $tag 2>&1 | grep -v -i warn
done < $2
