import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
from xtuner.registry import BUILDER
from mmengine.model import BaseModel
from mmengine.logging import print_log
from torch.nn.utils.rnn import pad_sequence
from xtuner.model.utils import guess_load_checkpoint
from diffusers.pipelines.sana.pipeline_sana import SanaPipeline
from peft import LoraConfig
from networks.openuni.connector import ConnectorConfig, ConnectorEncoder

IMAGENET_MEAN = (0.485, 0.456, 0.406)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            names = name.split(".")
            lora_module = names[0] if len(names) == 1 else names[-1]
            if lora_module == "0":
                lora_module = "to_out.0"
            lora_module_names.add(lora_module)

    return list(lora_module_names)


class GenTransformer(torch.nn.Module):
    def __init__(self, transformer, proj_type) -> None:
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.in_channels = transformer.config.in_channels
        self.enable_gradient_checkpointing = transformer.enable_gradient_checkpointing
        self.disable_gradient_checkpointing = transformer.disable_gradient_checkpointing
        self.proj_type = proj_type

    def forward(self, z, t, c, tt=None, **kwargs):
        # c_dtype = c[0].dtype
        # c = (self.llm2dit(c[0].to(torch.bfloat16)).to(c_dtype), c[1])
        c = (self.llm2dit(c[0]), c[1])
        timestep = t.expand(z.shape[0]) * 1000
        target_timestep = tt.expand(z.shape[0]) * 1000
        prediction = self.transformer(
            hidden_states=z,
            timestep=timestep,
            target_timestep=target_timestep,
            encoder_hidden_states=c[0],
            encoder_attention_mask=c[1],
            attention_kwargs=None,
            return_dict=False,
        )[0]
        return prediction

    def llm2dit(self, x):
        if self.proj_type == "proj_enc":
            return self.connector(self.projector(x))
        elif self.proj_type == "enc_proj":
            return self.projector(self.connector(x))
        elif self.proj_type == "proj_enc_proj":
            return self.projector[1](self.connector(self.projector[0](x)))
        else:
            raise ValueError(f"Unknown proj type: {self.proj_type}")

    def forward_with_cfg(self, x, t, c, cfg_scale, cfg_interval=None):
        """
        Forward pass for classifier-free guidance with interval, impl by UCGM.
        """

        t = t.flatten()
        if t[0] >= cfg_interval[0] and t[0] <= cfg_interval[1]:
            half = x[: len(x) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.forward(combined, t, c)

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
            half_eps = self.forward(half, t, c)
            eps = torch.cat([half_eps, half_eps], dim=0)

        return eps


class OpenUniInternVL3SANAHF(BaseModel):
    def __init__(
        self,
        lmm,
        transformer,
        train_scheduler,
        test_scheduler,
        vae,
        tokenizer,
        prompt_template,
        connector,
        num_queries=256,
        pretrained_pth=None,
        use_activation_checkpointing=True,
        lora_modules=None,  # ["to_k", "to_q", "to_v"],
        lora_rank=8,
        lora_alpha=8,
        freeze_lmm=True,
        freeze_transformer=True,
        vit_input_size=448,
        max_length=2048,
        proj_type="enc_proj",
    ):
        super().__init__()
        self.use_activation_checkpointing = use_activation_checkpointing

        self.lmm = BUILDER.build(lmm)
        if freeze_lmm:
            self.lmm.requires_grad_(False)
        self.freeze_lmm = freeze_lmm

        self.train_scheduler = BUILDER.build(train_scheduler)
        self.test_scheduler = BUILDER.build(test_scheduler)

        self.transformer = BUILDER.build(transformer)
        if freeze_transformer:
            self.transformer.requires_grad_(False)
        self.freeze_transformer = freeze_transformer
        if lora_modules is not None:
            if lora_modules == "auto":
                lora_modules = find_all_linear_names(self.transformer)
            # import pdb; pdb.set_trace()
            transformer_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights="gaussian",
                target_modules=lora_modules,
            )
            self.transformer.add_adapter(transformer_lora_config)
        self.transformer = GenTransformer(self.transformer, proj_type)

        self.vae = BUILDER.build(vae)
        self.vae.requires_grad_(False)

        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = prompt_template
        self.vit_input_size = vit_input_size
        self.max_length = max_length
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(
            prompt_template["IMG_CONTEXT_TOKEN"]
        )
        self.register_buffer("vit_mean", torch.tensor(IMAGENET_MEAN), persistent=False)
        self.register_buffer("vit_std", torch.tensor(IMAGENET_STD), persistent=False)

        self.num_queries = num_queries
        self.transformer.connector = ConnectorEncoder(ConnectorConfig(**connector))

        self.proj_type = proj_type
        if self.proj_type == "proj_enc":
            assert (
                self.transformer.connector.config.hidden_size
                == self.transformer.config.caption_channels
            )
            self.transformer.projector = nn.Linear(
                self.llm.config.hidden_size,
                self.transformer.connector.config.hidden_size,
            )
        elif self.proj_type == "enc_proj":
            assert (
                self.transformer.connector.config.hidden_size
                == self.llm.config.hidden_size
            )
            self.transformer.projector = nn.Linear(
                self.transformer.connector.config.hidden_size,
                self.transformer.config.caption_channels,
            )
        elif self.proj_type == "proj_enc_proj":
            self.transformer.projector = nn.ModuleList(
                [
                    nn.Linear(
                        self.llm.config.hidden_size,
                        self.transformer.connector.config.hidden_size,
                    ),
                    nn.Linear(
                        self.transformer.connector.config.hidden_size,
                        self.transformer.config.caption_channels,
                    ),
                ]
            )
        else:
            raise ValueError(f"Unknown proj type: {self.proj_type}")

        self.meta_queries = nn.Parameter(
            torch.zeros(num_queries, self.llm.config.hidden_size)
        )
        nn.init.normal_(
            self.meta_queries, std=1 / math.sqrt(self.llm.config.hidden_size)
        )

        if use_activation_checkpointing:
            self.llm.enable_input_require_grads()
            self.gradient_checkpointing_enable()

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            info = self.load_state_dict(pretrained_state_dict, strict=False)
            print_log(f"Load pretrained weight from {pretrained_pth}")

        self.imgs_dtype = torch.float32

    @property
    def llm(self):
        return self.lmm.language_model

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.transformer.enable_gradient_checkpointing()
        self.transformer.connector.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.transformer.disable_gradient_checkpointing()
        self.transformer.connector.gradient_checkpointing = False

    @property
    def device(self):
        return self.llm.device

    @property
    def dtype(self):
        return self.llm.dtype

    def train(self: T, mode: bool = True) -> T:
        super().train(mode=mode)
        if self.vae is not None:
            self.vae.train(mode=False)
        if not mode:
            self.gradient_checkpointing_disable()

        return self

    @torch.no_grad()
    def pixels_to_latents(self, x):
        scaling_factor = self.vae.config.scaling_factor
        z = self.vae.encode(x)[0] * scaling_factor
        return z

    @torch.no_grad()
    def latents_to_pixels(self, z):
        scaling_factor = self.vae.config.scaling_factor
        x_rec = self.vae.decode(z / scaling_factor)[0]
        return x_rec

    def prepare_forward_input(
        self,
        x,
        inputs_embeds=None,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
    ):
        b, l, _ = x.shape
        assert l > 0
        attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones(b, l)], dim=1
        )
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids[position_ids < 0] = 0

        # prepare context
        if past_key_values is not None:
            inputs_embeds = x
            position_ids = position_ids[:, -l:]
        else:
            if inputs_embeds is None:
                input_ids = input_ids.to(self.device)
                inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([inputs_embeds, x], dim=1)

        inputs = dict(
            inputs_embeds=inputs_embeds.to(torch.bfloat16),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        return inputs

    # def forward(self, data, data_samples=None, mode="loss"):
    #     if mode == "loss":
    #         return self.compute_loss(data_dict=data)
    #     else:
    #         raise NotImplementedError

    def forward(self, *args, use_encode=False, **kwargs):
        if use_encode is False:
            prediction = self.transformer(*args, **kwargs)
            return prediction
        else:
            return self.forward_encode(*args, **kwargs)

    def forward_encode(self, prompts, refer_images=None, do_cfg=True):
        if refer_images is None:
            return self.encode_prompt(prompts, do_cfg=do_cfg)
        else:
            refer_images = self.latents_to_pixels(refer_images.to(self.device))
            refer_images = refer_images.clamp(-1, 1)
            return self.encode_prompt_and_image(prompts, refer_images, do_cfg=do_cfg)

    def compute_loss(self, data_dict):
        losses = {}
        for data_type in ["text2image", "image2image"]:
            if data_type in data_dict:
                losses[f"loss_{data_type}"] = getattr(self, f"{data_type}_loss")(
                    data_dict[data_type]
                )
        if len(losses) == 0:
            if "pixel_values_src" in data_dict:
                losses[f"loss_image2image"] = self.image2image_loss(data_dict)
            else:
                losses[f"loss_text2image"] = self.text2image_loss(data_dict)

        return losses

    @torch.no_grad()
    def get_semantic_features(self, pixel_values):
        # pixel_values: [-1, 1]
        pixel_values = (pixel_values + 1.0) / 2  # [0, 1]
        pixel_values = pixel_values - self.vit_mean.view(1, 3, 1, 1)
        pixel_values = pixel_values / self.vit_std.view(1, 3, 1, 1)

        pixel_values = F.interpolate(
            pixel_values,
            size=(self.vit_input_size, self.vit_input_size),
            mode="bilinear",
        )
        vit_embeds = self.lmm.extract_feature(pixel_values)

        return vit_embeds

    # @torch.no_grad()
    # def prepare_text_conditions(self, prompt, cfg_prompt=None):
    #     if cfg_prompt is None:
    #         cfg_prompt = self.prompt_template["CFG"]
    #     else:
    #         cfg_prompt = self.prompt_template["GENERATION"].format(
    #             input=cfg_prompt.strip()
    #         )
    #     prompt = self.prompt_template["GENERATION"].format(input=prompt.strip())

    #     all_prompts = [
    #         self.prompt_template["INSTRUCTION"].format(input=prompt)
    #         + self.prompt_template["IMG_START_TOKEN"],
    #         self.prompt_template["INSTRUCTION"].format(input=cfg_prompt)
    #         + self.prompt_template["IMG_START_TOKEN"],
    #     ]

    #     input_ids = [
    #         self.tokenizer.encode(p, add_special_tokens=True, return_tensors="pt")[0]
    #         for p in all_prompts
    #     ]
    #     valid_lens = [len(input_ids_) for input_ids_ in input_ids]
    #     input_ids = pad_sequence(
    #         input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
    #     )
    #     attention_mask = torch.zeros_like(input_ids).bool()
    #     for i in range(len(input_ids)):
    #         attention_mask[i, : valid_lens[i]] = True

    #     return dict(
    #         input_ids=input_ids.to(self.device),
    #         attention_mask=attention_mask.to(self.device),
    #     )

    def text2image_loss(self, data_dict):

        # obtain image latents
        if "image_latents" in data_dict:
            image_latents = data_dict["image_latents"].to(
                dtype=self.dtype, device=self.device
            )
        else:
            pixel_values = data_dict["pixel_values"].to(
                dtype=self.dtype, device=self.device
            )
            image_latents = self.pixels_to_latents(pixel_values)

        b, _, height, weight = image_latents.shape

        input_ids = data_dict["input_ids"].to(self.device)
        attention_mask = data_dict["attention_mask"].to(self.device)
        hidden_states = self.meta_queries[None].expand(b, self.num_queries, -1)

        inputs = self.prepare_forward_input(
            x=hidden_states, input_ids=input_ids, attention_mask=attention_mask
        )

        output = self.llm.model(**inputs, return_dict=True)
        hidden_states = output.last_hidden_state[:, -self.num_queries :]
        self.connector.to(torch.bfloat16)
        self.projector.to(torch.bfloat16)
        hidden_states = self.llm2dit(hidden_states.to(torch.bfloat16))

        loss_diff = self.diff_loss(
            model_input=image_latents,
            prompt_embeds=hidden_states,
            prompt_attention_mask=None,
        )

        return loss_diff

    def image2image_loss(self, data_dict):

        pixel_values_src = data_dict["pixel_values_src"].to(
            dtype=self.dtype, device=self.device
        )
        vit_embeds = self.get_semantic_features(pixel_values_src)
        vit_embeds.requires_grad = True

        pixel_values = data_dict["pixel_values"].to(
            dtype=self.dtype, device=self.device
        )
        image_latents = self.pixels_to_latents(pixel_values)

        b, _, height, weight = image_latents.shape

        input_ids = data_dict["input_ids"].to(self.device)
        attention_mask = data_dict["attention_mask"].to(self.device)

        inputs_embeds = vit_embeds.new_zeros(
            *input_ids.shape, self.llm.config.hidden_size
        )
        inputs_embeds[input_ids == self.image_token_id] = vit_embeds.flatten(0, 1)
        inputs_embeds[input_ids != self.image_token_id] = (
            self.llm.get_input_embeddings()(input_ids[input_ids != self.image_token_id])
        )

        max_length = self.max_length
        if inputs_embeds.shape[1] > max_length:
            inputs_embeds = inputs_embeds[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]

        hidden_states = self.meta_queries[None].expand(b, self.num_queries, -1)

        inputs = self.prepare_forward_input(
            x=hidden_states, inputs_embeds=inputs_embeds, attention_mask=attention_mask
        )

        output = self.llm.model(**inputs, return_dict=True)
        hidden_states = output.last_hidden_state[:, -self.num_queries :]
        hidden_states = self.llm2dit(hidden_states)

        loss_diff = self.diff_loss(
            model_input=image_latents,
            prompt_embeds=hidden_states,
            prompt_attention_mask=None,
        )

        return loss_diff

    @torch.no_grad()
    def sample(
        self,
        prompts,
        images=None,
        cfg_scale=4.5,
        seed=None,
        height=512,
        width=512,
        times=1,
        return_traj=False,
        sampler=None,
    ):
        torch.manual_seed(seed)
        do_cfg = cfg_scale > 0.0
        if images == None:
            (
                prompt_embeds,
                prompt_attention_mask,
                neg_prompt_embeds,
                neg_prompt_attention_mask,
            ) = self.encode_prompt(prompts, do_cfg=do_cfg)
        else:
            (
                prompt_embeds,
                prompt_attention_mask,
                neg_prompt_embeds,
                neg_prompt_attention_mask,
            ) = self.encode_prompt_and_image(prompts, images, do_cfg=do_cfg)

        noise = torch.randn(
            [prompt_embeds.shape[0] * times, 32, height // 32, width // 32],
            # generator=generator,
            device=self.device,
            dtype=prompt_embeds.dtype,
        )
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
            model_fn = self.transformer

        prompt_embeds = prompt_embeds.to(torch.bfloat16)
        pooled_prompt_embeds = pooled_prompt_embeds.to(torch.bfloat16)
        latents = latents.to(torch.bfloat16)

        if do_cfg:
            model_kwargs = dict(
                c=(prompt_embeds, pooled_prompt_embeds),
                cfg_scale=cfg_scale,
                cfg_interval=[0.0, 1.0],
            )
        else:
            model_kwargs = dict(c=(prompt_embeds, pooled_prompt_embeds))

        latents = sampler(latents, model_fn, **model_kwargs)
        if do_cfg:
            latents, _ = latents.chunk(2, dim=1)

        if latents.shape[0] > 10:
            latents = latents[1::2].reshape(-1, *latents.shape[2:])
        else:
            latents = latents.reshape(-1, *latents.shape[2:])

        # print(latents.shape)
        latents = latents if return_traj else latents[-len(prompts) * times :]

        if return_traj:
            images = []
            for i in range(len(latents)):
                latent = latents[i : i + 1].to(self.device)
                image = self.latents_to_pixels(latent)
                images.append(image)
            images = torch.cat(images, dim=0)
            return images
        else:
            images = self.latents_to_pixels(latents.to(self.device))

        return images

    def prepare_text_conditions(
        self, input_prompts, do_cfg=False, img_tokens=None, cfg_prompt=None
    ):

        if cfg_prompt is None:
            cfg_prompt = self.prompt_template["CFG"]
        else:
            cfg_prompt = self.prompt_template["GENERATION"].format(
                input=cfg_prompt.strip()
            )
        cfg_prompt = self.prompt_template["INSTRUCTION"].format(input=cfg_prompt)
        if self.prompt_template.get("IMG_START_TOKEN_FOR_GENERATION", True):
            cfg_prompt += self.prompt_template["IMG_START_TOKEN"]

        prompts = []
        for prompt in input_prompts:
            if img_tokens is not None:
                prompt = (
                    prompt
                    + "\n Reference image:"
                    + self.prompt_template["IMG_START_TOKEN"]
                    + self.prompt_template["IMG_CONTEXT_TOKEN"] * img_tokens
                    + self.prompt_template["IMG_END_TOKEN"]
                )
            prompt = self.prompt_template["GENERATION"].format(input=prompt)
            prompt = self.prompt_template["INSTRUCTION"].format(input=prompt)
            if self.prompt_template.get("IMG_START_TOKEN_FOR_GENERATION", True):
                prompt += self.prompt_template["IMG_START_TOKEN"]
            prompts.append(prompt)

        if do_cfg:
            prompts = prompts + len(prompts) * [cfg_prompt]

        inputs = self.tokenizer(
            prompts, add_special_tokens=True, return_tensors="pt", padding=True
        ).to(self.device)

        input_ids = inputs.get("input_ids", None)
        inputs_embeds = inputs.get("inputs_embeds", None)
        attention_mask = inputs.get("attention_mask", None)

        return input_ids, inputs_embeds, attention_mask

    def encode_prompt(self, input_prompts, do_cfg=True):

        input_ids, inputs_embeds, attention_mask = self.prepare_text_conditions(
            input_prompts, do_cfg=do_cfg
        )

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        bsz = attention_mask.shape[0]
        if do_cfg:
            assert bsz % 2 == 0

        hidden_states = self.meta_queries[None].expand(bsz, self.num_queries, -1)
        inputs = self.prepare_forward_input(
            x=hidden_states, inputs_embeds=inputs_embeds, attention_mask=attention_mask
        )

        output = self.llm.model(**inputs, return_dict=True)
        hidden_states = output.last_hidden_state[:, -self.num_queries :]

        # hidden_states = self.llm2dit(hidden_states)
        attention_mask = torch.ones(
            bsz, self.num_queries, device=self.device, dtype=torch.bool
        )

        if do_cfg:
            prompt_embeds = hidden_states[: bsz // 2]
            prompt_attention_mask = attention_mask[: bsz // 2]
            neg_prompt_embeds = hidden_states[bsz // 2 :]
            neg_prompt_attention_mask = attention_mask[bsz // 2 :]
        else:
            prompt_embeds = hidden_states
            prompt_attention_mask = attention_mask

        if do_cfg:
            return (
                prompt_embeds.to(self.imgs_dtype),
                prompt_attention_mask.to(self.imgs_dtype),
                neg_prompt_embeds.to(self.imgs_dtype),
                neg_prompt_attention_mask.to(self.imgs_dtype),
            )
        else:
            return (
                prompt_embeds.to(self.imgs_dtype),
                prompt_attention_mask.to(self.imgs_dtype),
                None,
                None,
            )

    def encode_prompt_and_image(self, input_prompts, images, do_cfg=True):

        pixel_values_src = images.to(dtype=self.dtype, device=self.device)
        vit_embeds = self.get_semantic_features(pixel_values_src)
        # vit_embeds.requires_grad = True

        bsz = vit_embeds.shape[0]
        bsz = bsz * 2 if do_cfg else bsz

        input_ids, _, attention_mask = self.prepare_text_conditions(
            input_prompts, do_cfg=do_cfg, img_tokens=vit_embeds.shape[1]
        )

        inputs_embeds = vit_embeds.new_zeros(
            *input_ids.shape, self.llm.config.hidden_size
        )

        inputs_embeds[input_ids == self.image_token_id] = vit_embeds.flatten(0, 1)
        inputs_embeds[input_ids != self.image_token_id] = (
            self.llm.get_input_embeddings()(input_ids[input_ids != self.image_token_id])
        )

        max_length = self.max_length
        if inputs_embeds.shape[1] > max_length:
            inputs_embeds = inputs_embeds[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]

        hidden_states = self.meta_queries[None].expand(bsz, self.num_queries, -1)

        inputs = self.prepare_forward_input(
            x=hidden_states, inputs_embeds=inputs_embeds, attention_mask=attention_mask
        )

        output = self.llm.model(**inputs, return_dict=True)
        hidden_states = output.last_hidden_state[:, -self.num_queries :]

        attention_mask = torch.ones(
            bsz, self.num_queries, device=self.device, dtype=torch.bool
        )

        if do_cfg:
            prompt_embeds = hidden_states[: bsz // 2]
            prompt_attention_mask = attention_mask[: bsz // 2]
            neg_prompt_embeds = hidden_states[bsz // 2 :]
            neg_prompt_attention_mask = attention_mask[bsz // 2 :]
        else:
            prompt_embeds = hidden_states
            prompt_attention_mask = attention_mask

        if do_cfg:
            return (
                prompt_embeds.to(self.imgs_dtype),
                prompt_attention_mask.to(self.imgs_dtype),
                neg_prompt_embeds.to(self.imgs_dtype),
                neg_prompt_attention_mask.to(self.imgs_dtype),
            )
        else:
            return (
                prompt_embeds.to(self.imgs_dtype),
                prompt_attention_mask.to(self.imgs_dtype),
                None,
                None,
            )

    @torch.no_grad()
    def encode(
        self,
        prompt,
        images,
        refer_images=None,
        times=1,
    ):
        do_cfg = True
        if refer_images is None:
            (
                prompt_embeds,
                prompt_attention_mask,
                neg_prompt_embeds,
                neg_prompt_attention_mask,
            ) = self.encode_prompt(prompt, do_cfg=do_cfg)
        else:
            (
                prompt_embeds,
                prompt_attention_mask,
                neg_prompt_embeds,
                neg_prompt_attention_mask,
            ) = self.encode_prompt_and_image(prompt, refer_images, do_cfg=do_cfg)

        if do_cfg:
            prompt_embeds = torch.cat(
                (times * [prompt_embeds] + times * [neg_prompt_embeds]), dim=0
            )
            pooled_prompt_embeds = torch.cat(
                (times * [prompt_attention_mask] + times * [neg_prompt_attention_mask]),
                dim=0,
            )

        latents = self.vae.encode(images.to(self.device).to(self.imgs_dtype)).latent
        # print(latents)

        c = (
            prompt_embeds[: times * len(prompt)],
            pooled_prompt_embeds[: times * len(prompt)],
            prompt_embeds[times * len(prompt) :],
            pooled_prompt_embeds[times * len(prompt) :],
        )
        return latents, c

    def prepare_text_generation(self, input_prompts, img_tokens=None):

        prompts = []
        for prompt in input_prompts:
            if img_tokens is not None:
                prompt = (
                    prompt
                    + "\nReference image:"
                    + self.prompt_template["IMG_START_TOKEN"]
                    + self.prompt_template["IMG_CONTEXT_TOKEN"] * img_tokens
                    + self.prompt_template["IMG_END_TOKEN"]
                )

            prompt = self.prompt_template["INSTRUCTION"].format(input=prompt)
            system = self.prompt_template["SYSTEM"].format(
                system="You are a very helpful AI assistant."
            )
            prompts.append(system + prompt)

        inputs = self.tokenizer(
            prompts, add_special_tokens=True, return_tensors="pt", padding=True
        ).to(self.device)

        input_ids = inputs.get("input_ids", None)
        inputs_embeds = inputs.get("inputs_embeds", None)
        attention_mask = inputs.get("attention_mask", None)

        return input_ids, inputs_embeds, attention_mask

    def generate(self, prompts, images):
        if images is not None:
            pixel_values_src = images.to(dtype=self.dtype, device=self.device)
            # vit_embeds = self.lmm.extract_feature(pixel_values_src)
            vit_embeds = self.get_semantic_features(pixel_values_src)
            img_tokens = vit_embeds.shape[1]
        else:
            pixel_values_src = None
            vit_embeds = None
            img_tokens = None

        input_ids, _, attention_mask = self.prepare_text_generation(
            prompts, img_tokens=img_tokens
        )

        self.lmm.img_context_token_id = self.image_token_id

        generate_kwargs = dict(
            max_new_tokens=20,
            do_sample=False,
            # temperature=0.7,
            # top_p=0.9,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            pad_token_id=self.tokenizer.pad_token_id,
        )

        output_ids = self.lmm.generate(
            pixel_values=pixel_values_src,
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=vit_embeds,
            **generate_kwargs,
        )
        texts = []
        for output_id in output_ids:
            texts.append(self.tokenizer.decode(output_id, skip_special_tokens=True))
        return texts
