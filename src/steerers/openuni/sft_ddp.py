import os
import sys
import time
from absl import app, flags
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
import random
from torch.utils.data import DataLoader
from torch.amp import autocast as torch_autocast

from omegaconf import OmegaConf
from copy import deepcopy
from torchvision.utils import save_image

from data import Text2ImageDataset, Text2ImageParquetDataset
from services.tools import create_logger

# from networks import MODELS
from methodes import METHODES

from xtuner.registry import BUILDER
from mmengine.config import Config
from xtuner.model.utils import guess_load_checkpoint


def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def save_ckpt(
    ckpt_root_dir,
    model_to_save,
    optimizer,
    global_step,
):
    model_dir = os.path.join(ckpt_root_dir, f"global_step_{global_step}", "model")
    os.makedirs(model_dir, exist_ok=True)
    ema_dir = os.path.join(ckpt_root_dir, f"global_step_{global_step}", "ema")
    os.makedirs(ema_dir, exist_ok=True)
    opt_dir = os.path.join(ckpt_root_dir, f"global_step_{global_step}", "optimizer")
    os.makedirs(opt_dir, exist_ok=True)

    model_to_save.transformer.module.transformer.save_pretrained(model_dir)
    model_to_save.ema_transformer.transformer.save_pretrained(ema_dir)
    torch.save(optimizer.state_dict(), os.path.join(opt_dir, "optimizer.pt"))


def main(_):
    config_path = sys.argv[1]
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    method_type = config["method"].pop("method_type")
    method = METHODES[method_type](**config["method"])

    # --- Distributed Setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    parent_path = config_path.split("/")
    exp_name = os.path.join(parent_path[-2], parent_path[-1].split(".")[0])
    config["train"]["output_dir"] = os.path.join(
        config["train"]["output_dir"], exp_name
    )
    config["train"]["save_checkpoint_path"] = os.path.join(
        config["train"]["output_dir"], "checkpoints"
    )
    if is_main_process(rank):
        os.makedirs(config["train"]["output_dir"], exist_ok=True)
        os.makedirs(config["train"]["save_checkpoint_path"], exist_ok=True)

    logger = create_logger(__name__, config["train"]["output_dir"])

    demoimages_dir = os.path.join(config["train"]["output_dir"], "demoimages")
    if is_main_process(rank):
        os.makedirs(demoimages_dir, exist_ok=True)
    demo_c = [
        "a photo of a bench",
        "a photo of a cow",
    ]

    set_seed(config["train"]["seed"], rank)  # Pass rank for different seeds per process

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = torch.bfloat16
    enable_amp = mixed_precision_dtype is not None

    # --- Load pipeline and models ---
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
    wrapped_model.transformer.transformer.train().requires_grad_(True).to(torch.float32)

    wrapped_model.transformer.transformer.time_embed_2 = (
        wrapped_model.transformer.transformer.time_embed_2.to_empty(device=device)
    )
    wrapped_model.transformer.transformer.init_time_embed_2_weights()
    wrapped_model = wrapped_model.to(device)

    # --- Set up EMA model ---
    wrapped_model.add_module("ema_transformer", deepcopy(wrapped_model.transformer))
    wrapped_model.ema_transformer.requires_grad_(False)

    wrapped_model.transformer = DDP(
        wrapped_model.transformer,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    transformer_ddp = wrapped_model.transformer
    transformer_trainable_parameters = list(
        filter(lambda p: p.requires_grad, transformer_ddp.module.parameters())
    )
    transformer_ddp.train()

    # --- Optimizer ---
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        # transformer_ddp.module.parameters(),
        lr=config["train"]["lr"],
        betas=config["train"]["betas"],
        weight_decay=config["train"]["weight_decay"],
        foreach=True,
    )

    # --- Datasets and Dataloaders ---
    train_dataset = Text2ImageParquetDataset(
        data_dirs=config["data"]["train_dirs"],
        height=config["data"]["height"],
        width=config["data"]["width"],
        center_crop=True,
        random_flip=False,
    )
    train_sampler = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=4,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
        batch_size=config["train"]["micro_batch_size"],
    )

    total_train_batch_size = (
        config["train"]["micro_batch_size"]
        * world_size
        * config["train"]["grad_accumulation_steps"]
    )
    effective_grad_accum_steps = config["train"]["grad_accumulation_steps"]
    total_train_steps = len(train_dataset) // total_train_batch_size

    # --- Resume from checkpoint ---
    first_epoch = 0
    global_step = 0
    if config["train"]["load_checkpoint_path"] != "":
        load_path = config["train"]["load_checkpoint_path"]
        logger.info(f"Resuming from checkpoint: {load_path}")

        wrapped_model.transformer.module.transformer = (
            wrapped_model.transformer.module.transformer.from_pretrained(
                os.path.join(load_path, "model"), torch_dtype=torch.float32
            ).to(device)
        )
        wrapped_model.ema_transformer.transformer = (
            wrapped_model.ema_transformer.transformer.from_pretrained(
                os.path.join(load_path, "ema"), torch_dtype=torch.float32
            ).to(device)
        )
        opt_state = torch.load(
            os.path.join(load_path, "optimizer", "optimizer.pt"), map_location=device
        )
        optimizer.load_state_dict(opt_state)

        global_step_str = os.path.basename(os.path.normpath(load_path))
        if global_step_str.startswith("global_step_"):
            global_step = int(global_step_str.replace("global_step_", ""))
            first_epoch = global_step // total_train_steps  # approximate
        else:
            global_step = 0
            first_epoch = 0

    if is_main_process(rank):
        with (
            torch.no_grad(),
            torch_autocast(
                enabled=enable_amp, dtype=mixed_precision_dtype, device_type="cuda"
            ),
        ):
            demox = wrapped_model.sample(
                demo_c,
                sampler=method.sampling_loop,
                height=config["data"]["height"],
                width=config["data"]["width"],
                seed=42,
                cfg_scale=0.0,
                return_traj=True,
            )
            demoimages_path = f"{demoimages_dir}/{global_step:07d}.png"
            demox = (
                demox.view(-1, len(demo_c), *demox.shape[-3:])
                .permute(1, 0, 2, 3, 4)
                .reshape(-1, *demox.shape[-3:])
            )
            save_image(
                (demox + 1) / 2,
                os.path.join(demoimages_path),
                nrow=len(demox) // len(demo_c),
            )
            logger.info(f"Saved demoimages to {demoimages_path}")
            del demox

    logger.info("***** Running training *****")
    optimizer.zero_grad()

    for epoch in range(first_epoch, config["train"]["num_train_epochs"]):
        if hasattr(train_dataloader, "set_epoch"):
            train_sampler.set_epoch(epoch)

        train_iter = iter(train_dataloader)

        optimizer_step = 0
        for _ in range(total_train_steps):
            global_step += 1
            is_sync_step = global_step % effective_grad_accum_steps == 0
            if not is_sync_step:
                sync_context = transformer_ddp.no_sync()
            else:
                from contextlib import nullcontext

                sync_context = nullcontext()

            torch.cuda.synchronize()
            batch = next(train_iter)[0]
            text, image, z = batch["text"], batch["image"].cuda(), batch["z"]

            # logger.info(text[:2])
            start_time = time.time()
            if z.numel() != 0:
                z = z.cuda()
            else:
                z = None

            with sync_context:
                with torch.no_grad():
                    (
                        prompt_embeds,
                        prompt_embeds_mask,
                        uncond_prompt_embeds,
                        uncond_prompt_embeds_mask,
                    ) = wrapped_model.encode_prompt(text, do_cfg=True)
                    prompt_embeds = prompt_embeds.to(torch.float32)
                    uncond_prompt_embeds = uncond_prompt_embeds.to(torch.float32)

                    latents = wrapped_model.pixels_to_latents(image.to(torch.bfloat16))
                    latents = latents.to(torch.float32)

                # if is_main_process(rank):
                #     logger.info(f"\033[92mcond size: {prompt_embeds.shape}\033[0m")
                #     logger.info(f"\033[92muncond size: {uncond_prompt_embeds.shape}\033[0m")
                #     logger.info(f"\033[92mlatents size: {latents.shape}\033[0m")

                with torch_autocast(
                    enabled=enable_amp,
                    dtype=mixed_precision_dtype,
                    device_type="cuda",
                    cache_enabled=False,
                ):
                    loss = method.training_step(
                        wrapped_model,
                        latents,
                        c=[prompt_embeds, prompt_embeds_mask],
                        e=[uncond_prompt_embeds, uncond_prompt_embeds_mask],
                        step=(global_step - 1),
                        v=z,
                    )

                scaled_loss = loss / effective_grad_accum_steps
                scaled_loss.backward()
                
            for param in wrapped_model.parameters():
                if param.grad is not None:
                    torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

            grad_norm = 0.0
            if is_sync_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    transformer_ddp.module.parameters(),
                    config["train"]["max_grad_norm"],
                )
                optimizer.step()
                optimizer.zero_grad()
                optimizer_step += 1

            actual_loss = scaled_loss * effective_grad_accum_steps
            loss_tensor = torch.tensor(actual_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / world_size

            delta_time = time.time() - start_time

            if is_main_process(rank):
                logger.info(
                    f"step=({global_step:08d}): "
                    f"loss: {avg_loss:.4f}, "
                    f"grad_norm: {grad_norm:.2f}, "
                    # f"lr: {lr:.2e}, "
                    f"step_time: {delta_time:.2f}s"
                )

            if global_step % 100 == 0 and is_main_process(rank):
                with (
                    torch.no_grad(),
                    torch_autocast(
                        enabled=enable_amp,
                        dtype=mixed_precision_dtype,
                        device_type="cuda",
                    ),
                ):
                    demox = wrapped_model.sample(
                        demo_c,
                        sampler=method.sampling_loop,
                        height=config["data"]["height"],
                        width=config["data"]["width"],
                        seed=42,
                        cfg_scale=0.0,
                        return_traj=True,
                    )
                    demoimages_path = f"{demoimages_dir}/{global_step:07d}.png"
                    demox = (
                        demox.view(-1, len(demo_c), *demox.shape[-3:])
                        .permute(1, 0, 2, 3, 4)
                        .reshape(-1, *demox.shape[-3:])
                    )
                    save_image(
                        (demox + 1) / 2,
                        os.path.join(demoimages_path),
                        nrow=len(demox) // len(demo_c),
                    )
                    logger.info(f"Saved demoimages to {demoimages_path}")
                    del demox

                    save_ckpt(
                        config["train"]["save_checkpoint_path"],
                        wrapped_model,
                        optimizer,
                        global_step,
                    )
                    logger.info(
                        f"Saved ckpt to {config['train']['save_checkpoint_path']}"
                    )

            del batch

    if is_main_process(rank):
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
