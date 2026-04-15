import torch
from networks.openuni.internvl3_sana_hf import OpenUniInternVL3SANAHF
from networks.openuni.internvl3.modeling_internvl_chat import InternVLChatModel
from diffusers import (
    AutoencoderDC,
    SanaTransformer2DModel,
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
)

from mmengine.config import read_base

from networks.openuni.transformer_sana import SanaTransformer2DModelWrapper

with read_base():
    from .internvl3_2b_512_processor import (
        prompt_template,
        tokenizer,
        internvl3_model_name_or_path,
        image_size,
    )

import os
sana_model_name_or_path = (
    os.environ.get("SANA_1600M_512PX_PATH")
)

model = dict(
    type=OpenUniInternVL3SANAHF,
    num_queries=256,
    connector=dict(
        hidden_size=1536,
        intermediate_size=8960,
        num_hidden_layers=6,
        _attn_implementation="flash_attention_2",
        num_attention_heads=24,
    ),
    lmm=dict(
        type=InternVLChatModel.from_pretrained,
        pretrained_model_name_or_path=internvl3_model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
    ),
    vae=dict(
        type=AutoencoderDC.from_pretrained,
        pretrained_model_name_or_path=os.getenv("DCAE_PATH"),
        torch_dtype=torch.bfloat16,
    ),
    transformer=dict(
        # type=SanaTransformer2DModel.from_pretrained,
        type=SanaTransformer2DModelWrapper.from_pretrained,
        pretrained_model_name_or_path=sana_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ),
    train_scheduler=dict(
        type=FlowMatchEulerDiscreteScheduler.from_pretrained,
        pretrained_model_name_or_path=sana_model_name_or_path,
        subfolder="scheduler",
    ),
    test_scheduler=dict(
        type=DPMSolverMultistepScheduler.from_pretrained,
        pretrained_model_name_or_path=sana_model_name_or_path,
        subfolder="scheduler",
    ),
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    lora_modules=None,
    freeze_lmm=True,
    freeze_transformer=True,
)
