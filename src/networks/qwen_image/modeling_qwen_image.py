import torch
from torch import nn
import torch.nn.functional as F
from PIL.Image import Image
from typing import Union, Callable, List

from diffusers import (
    QwenImagePipeline,
    QwenImageEditPlusPipeline,
    QwenImageTransformer2DModel,
)
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    CONDITION_IMAGE_SIZE,
    VAE_IMAGE_SIZE,
    calculate_dimensions,
)

from .transformer_qwenimage import QwenImageTransformer2DModelWrapper

from services.tools import create_logger

logger = create_logger(__name__)


class GenTransformer(torch.nn.Module):
    def __init__(self, transformer, vae_scale_factor, aux_time_embed) -> None:
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.in_channels = transformer.config.in_channels // 4
        self.vae_scale_factor = vae_scale_factor
        self.aux_time_embed = aux_time_embed

    def enable_gradient_checkpointing(self):
        self.transformer.enable_gradient_checkpointing()

    def gradient_checkpointing_enable(self, *args, **kwargs):
        def _gradient_checkpointing_func(module, *args):
            return torch.utils.checkpoint.checkpoint(
                module.__call__,
                *args,
                **kwargs["gradient_checkpointing_kwargs"],
            )

        self.transformer.enable_gradient_checkpointing(_gradient_checkpointing_func)

    def init_weights(self):
        pass

    def add_adapter(self, *args, **kwargs):
        self.transformer.add_adapter(*args, **kwargs)

    def set_adapter(self, *args, **kwargs):
        self.transformer.set_adapter(*args, **kwargs)

    def disable_adapter(self, *args, **kwargs):
        self.transformer.disable_adapter(*args, **kwargs)

    def disable_lora(self):
        self.transformer.disable_lora()

    def enable_lora(self):
        self.transformer.enable_lora()

    def forward(self, x_t, t, c, tt=None, **kwargs):
        if kwargs.get("cfg_scale", 0) == 0:
            x_t_ = x_t.unsqueeze(1)
            if len(c) == 3:
                x_ref = c[-1].unsqueeze(1)
            else:
                x_ref = None

            packed_x_t = QwenImagePipeline._pack_latents(
                x_t_,
                x_t_.shape[0],
                x_t_.shape[2],
                x_t_.shape[3],
                x_t_.shape[4],
            )
            if x_ref is not None:
                packed_x_ref = QwenImageEditPlusPipeline._pack_latents(
                    x_ref,
                    x_ref.shape[0],
                    x_ref.shape[2],
                    x_ref.shape[3],
                    x_ref.shape[4],
                )
            else:
                packed_x_ref = None

            if x_ref is not None:
                img_shapes = [
                    [
                        (1, x_t.shape[-2] // 2, x_t.shape[-1] // 2),
                        (1, c[-1].shape[-2] // 2, c[-1].shape[-1] // 2),
                    ]
                ] * len(x_t)
            else:
                img_shapes = [[(1, x_t.shape[-2] // 2, x_t.shape[-1] // 2)]] * len(x_t)
            txt_seq_lens = c[1].int().sum(dim=1).tolist() if c[1] is not None else None
            txt_seq_lens = [int(i) for i in txt_seq_lens]

            encoder_hs = c[0][:, : max(txt_seq_lens)]
            encoder_hs_mask = c[1][:, : max(txt_seq_lens)]

            if x_ref is not None:
                packed_latents = torch.cat([packed_x_t, packed_x_ref], dim=1)
            else:
                packed_latents = packed_x_t

            attn_mask = torch.cat(
                (
                    torch.where(
                        encoder_hs_mask == 1,
                        torch.tensor(
                            0.0, device=packed_latents.device, dtype=torch.float32
                        ),
                        torch.tensor(
                            float("-inf"),
                            device=packed_latents.device,
                            dtype=torch.float32,
                        ),
                    ),
                    torch.zeros(
                        encoder_hs_mask.shape[0],
                        packed_latents.shape[1],
                        device=packed_latents.device,
                        dtype=torch.float32,
                    ),
                ),
                dim=1,
            )  # B * S

            attn_mask = (
                attn_mask[:, None, None, :]
                .expand(attn_mask.shape[0], 1, attn_mask.shape[1], attn_mask.shape[1])
                .contiguous()
            )

            transformer_kwargs = {
                "hidden_states": packed_latents,
                "timestep": t,
                "guidance": None,
                "encoder_hidden_states_mask": encoder_hs_mask,  # this is god damn not unused
                "encoder_hidden_states": encoder_hs,
                "img_shapes": img_shapes,
                "txt_seq_lens": txt_seq_lens,
                "attention_kwargs": {
                    "attention_mask": attn_mask
                },  # this is the real god damn mask
                "return_dict": False,
            }
            if self.aux_time_embed:
                assert tt is not None, "tt must be provided when aux_time_embed is True"
                transformer_kwargs["target_timestep"] = tt

            prediction = self.transformer(**transformer_kwargs)[0]
            prediction = prediction[:, : packed_x_t.size(1)]

            prediction = QwenImagePipeline._unpack_latents(
                prediction,
                height=x_t.shape[-2] * self.vae_scale_factor,
                width=x_t.shape[-1] * self.vae_scale_factor,
                vae_scale_factor=self.vae_scale_factor,
            )
            return prediction.squeeze(2)
        else:
            return self.forward_with_cfg(x=x_t, t=t, c=c, tt=tt, **kwargs)

    def forward_with_cfg(self, x, t, c, cfg_scale, cfg_interval=None, tt=None):
        """
        Forward pass for classifier-free guidance with interval, impl by UCGM.
        """

        t = t.flatten()
        if t[0] >= cfg_interval[0] and t[0] <= cfg_interval[1]:
            half = x[: len(x) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.forward(combined, t, c, tt)

            eps, rest = (
                model_out[:, : self.in_channels],
                model_out[:, self.in_channels :],
            )
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

            eps = torch.cat([half_eps, half_eps], dim=0)
            eps = torch.cat([eps, rest], dim=1)
        else:
            half = x[: len(x) // 2]
            t = t[: len(t) // 2]
            c = [c_[: len(c_) // 2] for c_ in c]
            half_eps = self.forward(half, t, c, tt)
            eps = torch.cat([half_eps, half_eps], dim=0)

        return eps


class QwenImage(torch.nn.Module):
    def __init__(
        self,
        model_id: str,
        model_type: str = "t2i",
        aux_time_embed: bool = False,
        text_dtype: torch.dtype = torch.bfloat16,
        imgs_dtype: torch.dtype = torch.bfloat16,
        max_sequence_length: int = 1024,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        self.aux_time_embed = aux_time_embed
        if aux_time_embed:
            transformer_cls = QwenImageTransformer2DModelWrapper
        else:
            transformer_cls = QwenImageTransformer2DModel

        qwen_transformer = transformer_cls.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=imgs_dtype,
            low_cpu_mem_usage=False,
        )

        if aux_time_embed:
            qwen_transformer.init_time_embed_2_weights()

        self.model_type = model_type
        if model_type == "t2i":
            self.model = QwenImagePipeline.from_pretrained(
                model_id, torch_dtype=imgs_dtype, transformer=qwen_transformer
            )
        elif model_type == "edit":
            self.model = QwenImageEditPlusPipeline.from_pretrained(
                model_id, torch_dtype=imgs_dtype, transformer=qwen_transformer
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.transformer = GenTransformer(
            self.model.transformer, self.model.vae_scale_factor, self.aux_time_embed
        )

        self.device = device
        self.max_sequence_length = max_sequence_length

        self.imgs_dtype = imgs_dtype
        self.text_dtype = text_dtype

        self.model.vae = (
            self.model.vae.to(dtype=self.imgs_dtype)
            .requires_grad_(False)
            .eval()
            .to(device)
        )
        # self.model.vae.enable_slicing()
        self.model.text_encoder = (
            self.model.text_encoder.to(dtype=self.text_dtype)
            .requires_grad_(False)
            .eval()
            .to(device)
        )

    def forward(self, x_t, t, c, tt=None):
        return self.transformer(x_t, t, c, tt)

    def get_no_split_modules(self):
        text_encoder_no_split_modules = [
            m for m in self.model.text_encoder._no_split_modules
        ]
        transformer_no_split_modules = [
            m for m in self.model.transformer._no_split_modules
        ]
        return text_encoder_no_split_modules + transformer_no_split_modules

    def train(self, mode: bool = True):
        self.transformer.train()
        return self

    def eval(self, mode: bool = True):
        self.transformer.eval()
        return self

    def requires_grad_(self, requires_grad: bool = True):
        self.transformer.requires_grad_(requires_grad)
        return self

    def encode_prompt(
        self,
        prompt: List[str],
        image: Union[list[Image], list[torch.Tensor], None] = None,
        do_cfg: bool = True,
    ):
        if do_cfg:
            input_args = {
                "prompt": tuple(prompt) + tuple(len(prompt) * ["Generate an image."]),
                "prompt_embeds": None,
                "prompt_embeds_mask": None,
                "device": self.device,
                "num_images_per_prompt": 1,
                "max_sequence_length": self.max_sequence_length,
            }

            if self.model_type == "t2i":
                prompt_embeds_, prompt_attention_mask_ = self.model.encode_prompt(
                    **input_args
                )
                prompt_embeds, neg_prompt_embeds = prompt_embeds_.chunk(2)
                prompt_attention_mask, neg_prompt_attention_mask = (
                    prompt_attention_mask_.chunk(2)
                )
                # logger.info_rank0(f"prompt_attention_mask: {prompt_attention_mask.shape}, neg_prompt_attention_mask: {neg_prompt_attention_mask.shape}")
                # logger.info_rank0(f"prompt_attention_mask: {prompt_attention_mask}")
                # logger.info_rank0(f"neg_prompt_attention_mask: {neg_prompt_attention_mask}")
                del prompt_embeds_, prompt_attention_mask_
            elif self.model_type == "edit":
                input_args["prompt"] = prompt
                cond_width, cond_height = calculate_dimensions(
                    CONDITION_IMAGE_SIZE, image.shape[-1] / image.shape[-2]
                )
                cond_image = F.interpolate(
                    image,
                    (cond_height, cond_width),
                    mode="bilinear",
                    align_corners=False,
                )
                input_args["image"] = cond_image
                prompt_embeds, prompt_attention_mask = self.model.encode_prompt(
                    **input_args
                )
                input_args["prompt"] = len(prompt) * ["Edit this image."]
                input_args["image"] = cond_image
                neg_prompt_embeds, neg_prompt_attention_mask = self.model.encode_prompt(
                    **input_args
                )
                print(
                    "prompt_embeds",
                    prompt_embeds.shape,
                    prompt_attention_mask.sum(dim=1),
                    "neg_prompt_embeds",
                    neg_prompt_embeds.shape,
                    neg_prompt_attention_mask.sum(dim=1),
                )
                max_len = max(prompt_embeds.shape[1], neg_prompt_embeds.shape[1])
                if prompt_embeds.shape[1] < max_len:
                    pad_diff = max_len - prompt_embeds.shape[1]
                    prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, pad_diff), value=0.0)
                    prompt_attention_mask = F.pad(
                        prompt_attention_mask, (0, pad_diff), value=0
                    )

                if neg_prompt_embeds.shape[1] < max_len:
                    pad_diff = max_len - neg_prompt_embeds.shape[1]
                    neg_prompt_embeds = F.pad(
                        neg_prompt_embeds, (0, 0, 0, pad_diff), value=0.0
                    )
                    neg_prompt_attention_mask = F.pad(
                        neg_prompt_attention_mask, (0, pad_diff), value=0
                    )

            return (
                prompt_embeds,
                prompt_attention_mask,
                neg_prompt_embeds,
                neg_prompt_attention_mask,
            )
        else:
            input_args = {
                "prompt": prompt,
                "prompt_embeds": None,
                "prompt_embeds_mask": None,
                "device": self.device,
                "num_images_per_prompt": 1,
                "max_sequence_length": self.max_sequence_length,
            }
            if self.model_type == "t2i":
                prompt_embeds, prompt_attention_mask = self.model.encode_prompt(
                    **input_args
                )
            elif self.model_type == "edit":
                input_args["image"] = image
                prompt_embeds, prompt_attention_mask = self.model.encode_prompt(
                    **input_args
                )

            return (
                prompt_embeds,
                prompt_attention_mask,
                None,
                None,
            )

    @torch.no_grad()
    def pixels_to_latents(self, pixels):
        pixel_values = pixels.to(self.model.vae.dtype).unsqueeze(2)

        # pixel_latents = self.model.vae.encode(pixel_values).latent_dist.sample()
        pixel_latents = self.model.vae.encode(pixel_values).latent_dist.mean
        pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)

        latents_mean = (
            torch.tensor(self.model.vae.config.latents_mean)
            .view(1, 1, self.model.vae.config.z_dim, 1, 1)
            .to(pixel_latents.device, pixel_latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.model.vae.config.latents_std).view(
            1, 1, self.model.vae.config.z_dim, 1, 1
        ).to(pixel_latents.device, pixel_latents.dtype)
        pixel_latents = (pixel_latents - latents_mean) * latents_std
        return pixel_latents.squeeze(1)

    # @torch.no_grad()
    def latents_to_pixels(self, latents):
        x_cur = latents.to(self.model.vae.dtype).unsqueeze(2)
        x_cur_mean = (
            torch.tensor(self.model.vae.config.latents_mean)
            .view(1, self.model.vae.config.z_dim, 1, 1, 1)
            .to(x_cur.device, x_cur.dtype)
        )
        x_cur_std = 1.0 / torch.tensor(self.model.vae.config.latents_std).view(
            1, self.model.vae.config.z_dim, 1, 1, 1
        ).to(x_cur.device, x_cur.dtype)
        latents = x_cur / x_cur_std + x_cur_mean
        pixels = self.model.vae.decode(latents, return_dict=False)[0][:, :, 0]
        return pixels

    @torch.no_grad()
    def sample(
        self,
        prompts: List[str],
        images: Union[list[Image], list[torch.Tensor], None] = None,
        cfg_scale: float = 4.5,
        seed: Union[int, List[int]] = 42,
        height: int = 512,
        width: int = 512,
        times: int = 1,
        return_traj: bool = False,
        sampler: Union[nn.Module, Callable, None] = None,
        use_ema: bool = False,
    ):
        do_cfg = cfg_scale > 0.0
        (
            prompt_embeds,
            prompt_attention_mask,
            neg_prompt_embeds,
            neg_prompt_attention_mask,
        ) = self.encode_prompt(prompts, images, do_cfg)

        if isinstance(seed, list):
            assert len(seed) == len(prompts) * times, (
                f"Length of seed list ({len(seed)}) must match total number of samples ({len(prompts) * times})"
            )
            noise = torch.cat(
                [
                    torch.randn(
                        [
                            1,
                            self.transformer.in_channels,
                            height // self.model.vae_scale_factor,
                            width // self.model.vae_scale_factor,
                        ],
                        dtype=self.imgs_dtype,
                        generator=torch.Generator(device="cpu").manual_seed(s),
                    )
                    for s in seed
                ],
                dim=0,
            ).cuda()
        else:
            noise = torch.randn(
                [
                    len(prompts) * times,
                    self.transformer.in_channels,
                    height // self.model.vae_scale_factor,
                    width // self.model.vae_scale_factor,
                ],
                dtype=self.imgs_dtype,
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).cuda()

        if do_cfg:
            prompt_embeds = torch.cat(
                (times * [prompt_embeds] + times * [neg_prompt_embeds]), dim=0
            )
            pooled_prompt_embeds = torch.cat(
                (times * [prompt_attention_mask] + times * [neg_prompt_attention_mask]),
                dim=0,
            )
            latents = torch.cat([noise] * 2)
            if use_ema:
                assert hasattr(self, "ema_transformer"), (
                    "`use_ema` is set True but `ema_transformer` is not initialized"
                )
                model_fn = self.ema_transformer
            else:
                model_fn = self.transformer
        else:
            latents = noise
            prompt_embeds = torch.cat(times * [prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(times * [prompt_attention_mask], dim=0)
            if use_ema:
                assert hasattr(self, "ema_transformer"), (
                    "`use_ema` is set True but `ema_transformer` is not initialized"
                )
                model_fn = self.ema_transformer
            else:
                model_fn = self.transformer

        if do_cfg:
            model_kwargs = dict(
                c=[prompt_embeds, pooled_prompt_embeds],
                cfg_scale=cfg_scale,
                cfg_interval=[0.0, 1.0],
            )
            if images is not None and self.model_type == "edit":
                latents_ref = self.pixels_to_latents(images).to(prompt_embeds.dtype)
                model_kwargs["c"].append(torch.cat([latents_ref, latents_ref], dim=0))
        else:
            model_kwargs = dict(c=[prompt_embeds, pooled_prompt_embeds])
            if images is not None and self.model_type == "edit":
                model_kwargs["c"].append(
                    self.pixels_to_latents(images).to(prompt_embeds.dtype)
                )

        latents = sampler(latents, model_fn, **model_kwargs)

        if do_cfg:
            latents, _ = latents.chunk(2, dim=1)

        if latents.shape[0] > 10:
            latents = latents[1::2].reshape(-1, *latents.shape[2:])
        else:
            latents = latents.reshape(-1, *latents.shape[2:])

        latents = latents if return_traj else latents[-len(prompts) * times :]

        if return_traj:
            images = []
            for i in range(len(latents)):
                latent = latents[i : i + 1].cuda()
                image = self.latents_to_pixels(latent)
                images.append(image)
            images = torch.cat(images, dim=0)
            return images
        else:
            CHUNK_SIZE = 8
            if latents.shape[0] <= CHUNK_SIZE:
                images = self.latents_to_pixels(latents.cuda())
            else:
                images = torch.cat(
                    [
                        self.latents_to_pixels(chunk.cuda())
                        for chunk in latents.split(CHUNK_SIZE, dim=0)
                    ],
                    dim=0,
                )

        return images

    @torch.no_grad()
    def prepare_data(
        self,
        prompt,
        images,
        times=1,
    ):
        do_cfg = True
        (
            prompt_embeds,
            prompt_attention_mask,
            neg_prompt_embeds,
            neg_prompt_attention_mask,
        ) = self.encode_prompt(prompt, do_cfg)

        if do_cfg:
            prompt_embeds = torch.cat(
                (times * [prompt_embeds] + times * [neg_prompt_embeds]), dim=0
            )
            pooled_prompt_embeds = torch.cat(
                (times * [prompt_attention_mask] + times * [neg_prompt_attention_mask]),
                dim=0,
            )
        latents = self.pixels_to_latents(images.to(self.device))
        c = (
            prompt_embeds[: times * len(prompt)],
            pooled_prompt_embeds[: times * len(prompt)],
            prompt_embeds[times * len(prompt) :],
            pooled_prompt_embeds[times * len(prompt) :],
        )
        return latents, c
