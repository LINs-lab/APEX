from .qwen_image import QwenImage
from .stable_diffusion_3 import StableDiffusion3

from functools import partial

MODELS = {
    "SD3_5": StableDiffusion3,
    "QwenImage": QwenImage,
    "QwenImageEdit": partial(QwenImage, model_type='edit'),
}
