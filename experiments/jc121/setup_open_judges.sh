#!/usr/bin/env bash
set -euo pipefail
root=/mnt/main/justin/mime-jc121-open-judges
export UV_CACHE_DIR="$root/cache/uv"
export PIP_CACHE_DIR="$root/cache/pip"
export HF_HOME="$root/cache/huggingface"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export USE_FLAX=0
cd "$root"
mkdir -p cache
uv venv --python 3.12 /mnt/local/mime-jc121-open-judges
uv pip install --python /mnt/local/mime-jc121-open-judges/bin/python --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0' 'torchvision==0.23.0'
uv pip install --python /mnt/local/mime-jc121-open-judges/bin/python 'transformers==5.17.0' 'laya==0.4.27' 'gliclass==0.1.22' 'peft==0.21.0' 'accelerate==1.15.0' pydantic scikit-learn numpy datasets
/mnt/local/mime-jc121-open-judges/bin/python -m pip --version || true
uv pip freeze --python /mnt/local/mime-jc121-open-judges/bin/python > environment-small.txt
echo 'SMALL_ENV_READY'
uv venv --python 3.12 /mnt/local/mime-jc121-pplx
uv pip install --python /mnt/local/mime-jc121-pplx/bin/python --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.0' 'torchvision==0.29.0'
uv pip install --python /mnt/local/mime-jc121-pplx/bin/python 'transformers==5.17.0' 'safetensors==0.8.0' 'pillow==12.3.0' 'huggingface-hub==1.31.0' accelerate
uv pip freeze --python /mnt/local/mime-jc121-pplx/bin/python > environment-pplx.txt
echo 'ALL_ENVS_READY'
