import torch

from diffusers import (
    ZImagePipeline,
    ZImageTransformer2DModel
)
from .transformer_z_image import ZImageTransformer2DModelWrapper

from services.tools import create_logger

logger = create_logger(__name__)

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

    def disable_lora(self):
        self.transformer.disable_lora()

    def enable_lora(self):
        self.transformer.enable_lora()

    def forward(self, x_t, t, c=None, tt=None, **kwargs):
        if c is None:
            c = kwargs.get('c', None)
        if c is None:
            raise ValueError("Condition 'c' must be provided either as positional or keyword argument")
        
        batch_size = x_t.shape[0]
        
        # Z-Image expects: List[Tensor(C, F, H, W)] where F=1 for images
        x_t_ = x_t.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
        x_list = list(x_t_.unbind(dim=0))  # List of B tensors, each (C, 1, H, W)
        
        encoder_hs = c[0]  # (B, L, D)
        encoder_hs_mask = c[1]  # (B, L)
        
        # Get actual sequence lengths from mask
        txt_seq_lens = encoder_hs_mask.int().sum(dim=1).tolist()
        txt_seq_lens = [int(i) for i in txt_seq_lens]
        max_txt_len = max(txt_seq_lens)
        
        # Truncate to max actual length to save computation
        encoder_hs = encoder_hs[:, :max_txt_len]  # (B, max_L, D)
        encoder_hs_mask = encoder_hs_mask[:, :max_txt_len]  # (B, max_L)
        

        # Z-Image expects: List[Tensor(L_i, D)] with actual lengths (no padding)
        cap_feats_list = []
        for i in range(batch_size):
            actual_len = txt_seq_lens[i]
            cap_feats_list.append(encoder_hs[i, :actual_len])  # (L_i, D)
        
        t_sign = t.sign()
        t_abs = t.abs()
        t = t_sign * (1.0 - t_abs)
        transformer_kwargs = {
            "x": x_list,
            "t": t,
            "cap_feats": cap_feats_list,
            "patch_size": 2,
            "f_patch_size": 1,
        }
        
        if self.aux_time_embed:
            tt_sign = tt.sign()
            tt_abs = tt.abs()
            tt = tt_sign * (1.0 - tt_abs)
            transformer_kwargs["target_timestep"] = tt
        
        output_list, _ = self.transformer(**transformer_kwargs)
        
        # output_list: List[Tensor(C, 1, H, W)] x B
        output_tensor = torch.stack(output_list, dim=0)  # (B, C, 1, H, W)
        prediction = output_tensor.squeeze(2)  # (B, C, H, W)
        
        return -prediction

    def forward_with_cfg(self, x, t, c=None, cfg_scale=None, cfg_interval=None, tt=None, **kwargs):
        if c is None:
            c = kwargs.get('c', None)
        if cfg_scale is None:
            cfg_scale = kwargs.get('cfg_scale', 1.0)
        if c is None:
            raise ValueError("Condition 'c' must be provided")
        
        if cfg_interval is None:
            cfg_interval = [0.0, 1.0]  

        t = t.flatten()
        if t[0] >= cfg_interval[0] and t[0] <= cfg_interval[1]:
            half = x[: len(x) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.forward(combined, t, c=c, tt=tt)

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
            half_eps = self.forward(half, t, c=c, tt=tt)
            eps = torch.cat([half_eps, half_eps], dim=0)

        return eps


class ZImage(torch.nn.Module):
    def __init__(
        self,
        model_id,
        model_type='t2i',
        aux_time_embed=False,
        text_dtype=torch.bfloat16,
        imgs_dtype=torch.bfloat16,
        max_sequence_length=1024,
        device="cuda",
    ) -> None:
        super().__init__()

        self.aux_time_embed = aux_time_embed
        if aux_time_embed:
            transformer_cls = ZImageTransformer2DModelWrapper
        else:
            transformer_cls = ZImageTransformer2DModel
        
        z_image_transformer = transformer_cls.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=imgs_dtype,
            low_cpu_mem_usage=False,
        )
        
        self.model_type = model_type
        if model_type == 't2i':
            self.model = ZImagePipeline.from_pretrained(model_id, torch_dtype=imgs_dtype, transformer=z_image_transformer)
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
            self.model.vae.to(dtype=self.imgs_dtype).requires_grad_(False).eval().to(device)
        )
        self.model.text_encoder = (
            self.model.text_encoder.to(dtype=self.text_dtype)
            .requires_grad_(False)
            .eval().to(device)
        )

    def forward(self, x_t, t, c=None, tt=None, **kwargs):
        return self.transformer(x_t, t, c=c, tt=tt, **kwargs)

    def get_no_split_modules(self):
        text_encoder_no_split_modules = [m for m in self.model.text_encoder._no_split_modules]
        transformer_no_split_modules = [m for m in self.model.transformer._no_split_modules]
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

    def encode_prompt(self, prompt, image=None, do_cfg=True):
        if do_cfg:
            if self.model_type == 't2i':
                prompt_embeds_list, neg_prompt_embeds_list = self.model.encode_prompt(
                    prompt=prompt,
                    negative_prompt=None, 
                    do_classifier_free_guidance=True,
                    device=self.device,
                    max_sequence_length=self.max_sequence_length,
                )
                
                max_len_pos = max(len(emb) for emb in prompt_embeds_list)
                max_len_neg = max(len(emb) for emb in neg_prompt_embeds_list)
                max_len = max(max_len_pos, max_len_neg)
                
                batch_size = len(prompt_embeds_list)
                embed_dim = prompt_embeds_list[0].shape[-1]
                device = prompt_embeds_list[0].device
                dtype = prompt_embeds_list[0].dtype
                
                prompt_embeds = torch.zeros(
                    (batch_size, max_len, embed_dim), device=device, dtype=dtype
                )
                prompt_attention_mask = torch.zeros(
                    (batch_size, max_len), device=device, dtype=dtype
                )
                
                for i, emb in enumerate(prompt_embeds_list):
                    length = len(emb)
                    prompt_embeds[i, :length] = emb
                    prompt_attention_mask[i, :length] = 1.0
                
                # Same for negative prompts
                neg_prompt_embeds = torch.zeros(
                    (batch_size, max_len, embed_dim), device=device, dtype=dtype
                )
                neg_prompt_attention_mask = torch.zeros(
                    (batch_size, max_len), device=device, dtype=dtype
                )
                
                for i, emb in enumerate(neg_prompt_embeds_list):
                    length = len(emb)
                    neg_prompt_embeds[i, :length] = emb
                    neg_prompt_attention_mask[i, :length] = 1.0
                
            elif self.model_type == 'edit':
                pass

            return (
                prompt_embeds.to(self.imgs_dtype),
                prompt_attention_mask.to(self.imgs_dtype),
                neg_prompt_embeds.to(self.imgs_dtype),
                neg_prompt_attention_mask.to(self.imgs_dtype),
            )
        else:
            if self.model_type == 't2i':
                prompt_embeds_list, _ = self.model.encode_prompt(
                    prompt=prompt,
                    negative_prompt=None,
                    do_classifier_free_guidance=False,
                    device=self.device,
                    max_sequence_length=self.max_sequence_length,
                )
                
                max_len = max(len(emb) for emb in prompt_embeds_list)
                batch_size = len(prompt_embeds_list)
                embed_dim = prompt_embeds_list[0].shape[-1]
                device = prompt_embeds_list[0].device
                dtype = prompt_embeds_list[0].dtype
                
                prompt_embeds = torch.zeros(
                    (batch_size, max_len, embed_dim), device=device, dtype=dtype
                )
                prompt_attention_mask = torch.zeros(
                    (batch_size, max_len), device=device, dtype=dtype
                )
                
                for i, emb in enumerate(prompt_embeds_list):
                    length = len(emb)
                    prompt_embeds[i, :length] = emb
                    prompt_attention_mask[i, :length] = 1.0
                
            elif self.model_type == 'edit':
                raise NotImplementedError("Edit mode is not yet fully implemented for Z-Image")

            return (
                prompt_embeds.to(self.imgs_dtype),
                prompt_attention_mask.to(self.imgs_dtype),
                None,
                None,
            )

    @torch.no_grad()
    def pixels_to_latents(self, pixels):
        pixel_values = pixels.to(self.model.vae.dtype)
        pixel_latents = self.model.vae.encode(pixel_values).latent_dist.mean  
        pixel_latents = (pixel_latents - self.model.vae.config.shift_factor) * self.model.vae.config.scaling_factor
        
        return pixel_latents

    # @torch.no_grad()
    def latents_to_pixels(self, latents):
        latents = latents.to(self.model.vae.dtype)
        latents = (latents / self.model.vae.config.scaling_factor) + self.model.vae.config.shift_factor
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
        use_ema=False
    ):
        do_cfg = cfg_scale > 0.0
        (
            prompt_embeds,
            prompt_attention_mask,
            neg_prompt_embeds,
            neg_prompt_attention_mask,
        ) = self.encode_prompt(prompts, images, do_cfg)

        noise = torch.randn(
            [
                len(prompts) * times,
                self.transformer.in_channels,
                height // self.model.vae_scale_factor,
                width // self.model.vae_scale_factor,
            ],
            dtype=self.imgs_dtype,
            generator=torch.Generator(device='cpu').manual_seed(seed),
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
            model_fn = self.transformer.forward_with_cfg
        else:
            latents = noise
            prompt_embeds = torch.cat(times * [prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(times * [prompt_attention_mask], dim=0)
            if use_ema:
                assert hasattr(self, 'ema_transformer'), "`use_ema` is set True but `ema_transformer` is not initialized"
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
            images = self.latents_to_pixels(latents.cuda())

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
