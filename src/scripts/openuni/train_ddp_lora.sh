#!/bin/bash

# pip install xtuner absl-py omegaconf piq
# pip install -U transformers
# pip install -U diffusers

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
NNODES=${WORLD_SIZE:=1}
NPROC_PER_NODE=${NPROC_PER_NODE:=8}
NODE_RANK=${RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12347}

TORCH_DISTRIBUTED_DEBUG=DETAIL

torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT \
  steerers/openuni/sft_ddp_lora.py $CONFIG_PATH