from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import register_to_config
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

from diffusers.models.modeling_outputs import Transformer2DModelOutput

from diffusers.models.transformers.sana_transformer import (
    SanaCombinedTimestepGuidanceEmbeddings,
    AdaLayerNormSingle,
    SanaTransformer2DModel,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SanaTransformer2DModelWrapper(SanaTransformer2DModel):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: Optional[int] = 32,
        num_attention_heads: int = 70,
        attention_head_dim: int = 32,
        num_layers: int = 20,
        num_cross_attention_heads: Optional[int] = 20,
        cross_attention_head_dim: Optional[int] = 112,
        cross_attention_dim: Optional[int] = 2240,
        caption_channels: int = 2304,
        mlp_ratio: float = 2.5,
        dropout: float = 0.0,
        attention_bias: bool = False,
        sample_size: int = 32,
        patch_size: int = 1,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: Optional[int] = None,
        guidance_embeds: bool = False,
        guidance_embeds_scale: float = 0.1,
        qk_norm: Optional[str] = None,
        timestep_scale: float = 1.0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            num_cross_attention_heads=num_cross_attention_heads,
            cross_attention_head_dim=cross_attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            caption_channels=caption_channels,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_bias=attention_bias,
            sample_size=sample_size,
            patch_size=patch_size,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            interpolation_scale=interpolation_scale,
            guidance_embeds=guidance_embeds,
            guidance_embeds_scale=guidance_embeds_scale,
            qk_norm=qk_norm,
            timestep_scale=timestep_scale,
        )

        inner_dim = num_attention_heads * attention_head_dim
        self.time_embed_2 = AdaLayerNormSingle(inner_dim)

    def init_time_embed_2_weights(self, mean=0.0, std=0.02):        
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=mean, std=std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.time_embed_2.apply(_init_weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples: Optional[Tuple[torch.Tensor]] = None,
        return_dict: bool = True,
        target_timestep: torch.Tensor = None,
    ) -> Union[Tuple[torch.Tensor, ...], Transformer2DModelOutput]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size, num_channels, height, width = hidden_states.shape
        p = self.config.patch_size
        post_patch_height, post_patch_width = height // p, width // p

        hidden_states = self.patch_embed(hidden_states)

        t_scale = (timestep - target_timestep).unsqueeze(1) / 1000
        if guidance is not None:
            timestep, embedded_timestep = self.time_embed(
                timestep, guidance=guidance, hidden_dtype=hidden_states.dtype
            )
        else:
            timestep, embedded_timestep = self.time_embed(
                timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )

        if guidance is not None:
            target_timestep, embedded_target_timestep = self.time_embed_2(
                target_timestep, guidance=guidance, hidden_dtype=hidden_states.dtype
            )
        else:
            target_timestep, embedded_target_timestep = self.time_embed_2(
                target_timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )

        timestep = timestep + target_timestep * t_scale
        embedded_timestep = embedded_timestep + embedded_target_timestep * t_scale

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(
            batch_size, -1, hidden_states.shape[-1]
        )

        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        # 2. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for index_block, block in enumerate(self.transformer_blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(
                    controlnet_block_samples
                ):
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block - 1]
                    )

        else:
            for index_block, block in enumerate(self.transformer_blocks):
                hidden_states = block(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(
                    controlnet_block_samples
                ):
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block - 1]
                    )

        # 3. Normalization
        hidden_states = self.norm_out(
            hidden_states, embedded_timestep, self.scale_shift_table
        )

        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_height,
            post_patch_width,
            self.config.patch_size,
            self.config.patch_size,
            -1,
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(
            batch_size, -1, post_patch_height * p, post_patch_width * p
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
