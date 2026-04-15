import torch

from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel

from .transformer_sd_3 import SD3Transformer2DModelWrapper

from services.tools import create_logger

logger = create_logger(__name__)

import logging
logging.getLogger("diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3").setLevel(logging.ERROR)


class GenTransformer(torch.nn.Module):
    def __init__(self, transformer, vae_scale_factor, aux_time_embed) -> None:
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.in_channels = transformer.config.in_channels
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

    def disable_adapters(self):
        self.transformer.disable_adapters()

    def enable_adapters(self):
        self.transformer.enable_adapters()

    def disable_lora(self):
        self.transformer.disable_lora()

    def enable_lora(self):
        self.transformer.enable_lora()

    def forward(self, x_t, t, c, tt=None, **kwargs):
        if kwargs.get("cfg_scale", 0) == 0:
            encoder_hs = c[0]
            encoder_hs_mask = (c[0] != 0).any(dim=-1).to(torch.long)
            pooled_encoder_hs = c[1]
            attn_mask = torch.cat(
                (
                    torch.zeros(
                        encoder_hs_mask.shape[0],
                        x_t.shape[1],
                        device=x_t.device,
                        dtype=torch.float32,
                    ),
                    torch.where(
                        encoder_hs_mask == 1,
                        torch.tensor(0.0, device=x_t.device, dtype=torch.float32),
                        torch.tensor(
                            float("-inf"), device=x_t.device, dtype=torch.float32
                        ),
                    ),
                ),
                dim=1,
            )  # B * S

            # credit to @xj
            attn_mask = (
                attn_mask[:, None, None, :]
                .expand(attn_mask.shape[0], 1, attn_mask.shape[1], attn_mask.shape[1])
                .contiguous()
            )

            transformer_kwargs = {
                "hidden_states": x_t,
                "timestep": t * 1000,
                "encoder_hidden_states": encoder_hs,
                "pooled_projections": pooled_encoder_hs,
                # "joint_attention_kwargs": {
                #     "attention_mask": attn_mask
                # },  # this is the real god damn mask
                "return_dict": False,
            }
            if self.aux_time_embed:
                assert tt is not None, "tt must be provided when aux_time_embed is True"
                transformer_kwargs["target_timestep"] = tt * 1000

            # print(x_t.shape)
            # print('-'*10)
            # print(encoder_hs.shape)
            # print('='*10)
            # print(pooled_encoder_hs.shape)
            # print('*'*10)

            prediction = self.transformer(**transformer_kwargs)[0]

            return prediction
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


class StableDiffusion3(torch.nn.Module):
    def __init__(
        self,
        model_id,
        model_type="t2i",
        aux_time_embed=False,
        text_dtype=torch.bfloat16,
        imgs_dtype=torch.bfloat16,
        max_sequence_length=256,
        device="cuda",
        lora_id=None,
    ) -> None:
        super().__init__()

        self.aux_time_embed = aux_time_embed
        if aux_time_embed:
            transformer_cls = SD3Transformer2DModelWrapper
        else:
            transformer_cls = SD3Transformer2DModel
        
        sd_transformer = transformer_cls.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=imgs_dtype,
            low_cpu_mem_usage=False,
        )

        if aux_time_embed:
            sd_transformer.init_time_embed_2_weights()

        self.model_type = model_type
        self.model = StableDiffusion3Pipeline.from_pretrained(
            model_id, torch_dtype=imgs_dtype, transformer=sd_transformer
        )

        self.transformer = GenTransformer(
            self.model.transformer, self.model.vae_scale_factor, self.aux_time_embed
        )

        self.device = device
        self.max_sequence_length = max_sequence_length

        self.imgs_dtype = imgs_dtype
        self.text_dtype = text_dtype

        self.model.vae = (
            self.model.vae.to(dtype=torch.float32)
            .requires_grad_(False)
            .eval()
            .to(device)
        )
        self.model.vae.enable_slicing()
        self.model.text_encoder = (
            self.model.text_encoder.to(dtype=self.text_dtype)
            .requires_grad_(False)
            .eval()
            .to(device)
        )
        self.model.text_encoder_2 = (
            self.model.text_encoder_2.to(dtype=self.text_dtype)
            .requires_grad_(False)
            .eval()
            .to(device)
        )
        self.model.text_encoder_3 = (
            self.model.text_encoder_3.to(dtype=self.text_dtype)
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
        text_encoder_2_no_split_modules = [
            m for m in self.model.text_encoder_2._no_split_modules
        ]
        text_encoder_3_no_split_modules = [
            m for m in self.model.text_encoder_2._no_split_modules
        ]
        transformer_no_split_modules = [
            m for m in self.model.transformer._no_split_modules
        ]
        return (
            text_encoder_no_split_modules
            + transformer_no_split_modules
            + text_encoder_2_no_split_modules
            + text_encoder_3_no_split_modules
        )

    def train(self, mode: bool = True):
        self.transformer.train()
        return self

    def eval(self, mode: bool = True):
        self.transformer.eval()
        return self

    def requires_grad_(self, requires_grad: bool = True):
        self.transformer.requires_grad_(requires_grad)
        return self

    def encode_prompt(self, prompt, prompt_2=None, prompt_3=None, do_cfg=True):
        if do_cfg:
            input_args = {
                "prompt": tuple(prompt),
                "prompt_2": prompt_2,
                "prompt_3": prompt_3,
                "negative_prompt": tuple(len(prompt) * ["Generate an image."]),
                "negative_prompt_2": None,
                "negative_prompt_3": None,
                "prompt_embeds": None,
                "pooled_prompt_embeds": None,
                "device": self.device,
                "num_images_per_prompt": 1,
                "max_sequence_length": self.max_sequence_length,
            }

            (
                prompt_embeds,
                neg_prompt_embeds,
                pooled_prompt_embeds,
                neg_pooled_prompt_embeds,
            ) = self.model.encode_prompt(**input_args)

            return (
                prompt_embeds,
                pooled_prompt_embeds,
                neg_prompt_embeds,
                neg_pooled_prompt_embeds,
            )
        else:
            input_args = {
                "prompt": prompt,
                "prompt_2": prompt_2,
                "prompt_3": prompt_3,
                "prompt_embeds": None,
                "pooled_prompt_embeds": None,
                "device": self.device,
                "num_images_per_prompt": 1,
                "max_sequence_length": self.max_sequence_length,
            }
            prompt_embeds, _, pooled_prompt_embeds, _ = self.model.encode_prompt(
                **input_args
            )

            return (
                prompt_embeds,
                pooled_prompt_embeds,
                None,
                None,
            )

    @torch.no_grad()
    def pixels_to_latents(self, pixels):
        pixel_values = pixels.to(self.model.vae.dtype)
        # pixel_latents = self.model.vae.encode(pixel_values).latent_dist.sample()
        pixel_latents = self.model.vae.encode(pixel_values).latent_dist.mean
        pixel_latents = (
            pixel_latents - self.model.vae.config.shift_factor
        ) * self.model.vae.config.scaling_factor
        return pixel_latents

    # @torch.no_grad()
    def latents_to_pixels(self, latents):
        x_cur = latents.to(self.model.vae.dtype)
        latents = (
            x_cur / self.model.vae.config.scaling_factor
        ) + self.model.vae.config.shift_factor
        pixels = self.model.vae.decode(latents, return_dict=False)[0]
        return pixels

    @torch.no_grad()
    def sample(
        self,
        prompts,
        images=None,
        cfg_scale=4.5,
        seed=42,
        height=512,
        width=512,
        times=1,
        return_traj=False,
        sampler=None,
        use_ema=False,
    ):
        do_cfg = cfg_scale > 0.0
        (
            prompt_embeds,
            pooled_prompt_embeds,
            neg_prompt_embeds,
            neg_pooled_prompt_embeds,
        ) = self.encode_prompt(prompts, do_cfg=do_cfg)

        if isinstance(seed, list):
            assert len(seed) == len(prompts) * times, (
                f"Length of seed list ({len(seed)}) must match total number of samples ({len(prompts) * times})"
            )
            noise = torch.cat(
                [
                    torch.randn(
                        [
                            1,
                            self.transformer.module.in_channels if hasattr(self.transformer, "module") else self.transformer.in_channels,
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
                    self.transformer.module.in_channels if hasattr(self.transformer, "module") else self.transformer.in_channels,
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
                (times * [pooled_prompt_embeds] + times * [neg_pooled_prompt_embeds]),
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
            pooled_prompt_embeds = torch.cat(times * [pooled_prompt_embeds], dim=0)
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
        else:
            model_kwargs = dict(c=[prompt_embeds, pooled_prompt_embeds])

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
