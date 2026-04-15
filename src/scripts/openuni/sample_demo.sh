#!/bin/bash

# pip install xtuner absl-py omegaconf piq
# pip install -U transformers
# pip install /dllab/share/user/deyuan/test/diffusers

# https://huggingface.co/OpenGVLab/InternVL3-2B
export INTERNVL3_PATH="path/to/InternVL3-2B"
# https://huggingface.co/Efficient-Large-Model/Sana_1600M_512px_diffusers
export SANA_1600M_512PX_PATH="path/to/Sana_1600M_512px_diffusers"
# https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers
export DCAE_PATH="path/to/dc-ae-f32c32-sana-1.1-diffusers"

export PYTHONPATH=./
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800  # 30min

CONFIG_PATH=$1
python steerers/openuni/sample_demo.py $CONFIG_PATH