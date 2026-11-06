#!/usr/bin/env bash
set -euo pipefail
root=/mnt/main/justin/mime-jc121-open-judges
export HF_HOME="$root/cache/huggingface"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export USE_FLAX=0
cd "$root"
for model in verdict laya kev pplx; do
  interpreter=/mnt/local/mime-jc121-open-judges/bin/python
  if [ "$model" = pplx ]; then interpreter=/mnt/local/mime-jc121-pplx/bin/python; fi
  "$interpreter" -u judge_open.py "$model" --root "$root" --limit 4 > "$model-pilot.log" 2>&1
  "$interpreter" -u judge_open.py "$model" --root "$root" > "$model-run.log" 2>&1
done
