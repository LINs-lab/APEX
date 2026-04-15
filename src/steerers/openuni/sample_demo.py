import sys
import os
import torch
from torch.amp import autocast as torch_autocast
from torchvision.utils import save_image
from functools import partial
from omegaconf import OmegaConf

from services.tools import create_logger

# from networks import MODELS
from methodes import METHODES

from xtuner.registry import BUILDER
from mmengine.config import Config
from xtuner.model.utils import guess_load_checkpoint

config_path = sys.argv[1]
config = OmegaConf.load(config_path)
config = OmegaConf.to_container(config, resolve=True)

parent_path = config_path.split("/")
exp_name = os.path.join(parent_path[-2], parent_path[-1].split(".")[0])
config["train"]["output_dir"] = os.path.join(config["train"]["output_dir"], exp_name)
config["train"]["save_checkpoint_path"] = os.path.join(
    config["train"]["output_dir"], "checkpoints"
)

method_type = config["method"].pop("method_type")
method = METHODES[method_type](**config["method"])

device = torch.device(f"cuda")

model_config = Config.fromfile(config["model"]["type"])
wrapped_model = BUILDER.build(model_config.model)
state_dict = guess_load_checkpoint(config["model"]["path"])
new_state_dict = {}
for key, value in state_dict.items():
    if (
        key.startswith("transformer.")
        or key.startswith("connector.")
        or key.startswith("projector.")
    ):
        new_key = "transformer." + key
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value
state_dict = new_state_dict
missing, unexpected = wrapped_model.load_state_dict(state_dict, strict=False)
# logger.info(f"Unexpected parameters: {unexpected}, Missing parameters: {missing}")
wrapped_model = wrapped_model.to(wrapped_model.dtype)
wrapped_model.eval().requires_grad_(False)
wrapped_model.transformer.connector.eval().requires_grad_(False).to(torch.float32)
wrapped_model.transformer.projector.eval().requires_grad_(False).to(torch.float32)
wrapped_model.transformer.transformer.train().requires_grad_(False).to(torch.float32)

wrapped_model.transformer.transformer.time_embed_2 = (
    wrapped_model.transformer.transformer.time_embed_2.to_empty(device=device)
)
wrapped_model.transformer.transformer.init_time_embed_2_weights()
wrapped_model = wrapped_model.to(device)

load_path = os.path.join(
    config["train"]["save_checkpoint_path"],
    f"global_step_{config['sample'].pop('ckpt')}",
)

wrapped_model.transformer.transformer = (
    wrapped_model.transformer.transformer.from_pretrained(
        os.path.join(load_path, "model"), torch_dtype=torch.float32
    ).to(device)
)

cfg_scale = config["sample"].pop("cfg_scale")
cfg_interval = config["sample"].pop("cfg_interval")

# please see the config to swith between different sampling style
sampler = partial(method.sampling_loop, **config["sample"])

demo_c = [
    "a photo of a bench",
    "a photo of a cow",
]

with (
    torch.no_grad(),
    torch_autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"),
):
    demox = wrapped_model.sample(
        demo_c,
        sampler=sampler,
        height=512,
        width=512,
        seed=42,
        cfg_scale=cfg_scale,
        return_traj=False,
    )
    demox = (
        demox.view(-1, len(demo_c), *demox.shape[-3:])
        .permute(1, 0, 2, 3, 4)
        .reshape(-1, *demox.shape[-3:])
    )
    save_image(
        (demox + 1) / 2,
        f"./test.png",
        nrow=len(demox) // len(demo_c),
    )
