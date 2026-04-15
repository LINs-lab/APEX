from .qwen_image import QwenImage

from functools import partial

MODELS = {
    "QwenImage": QwenImage,
    "QwenImageEdit": partial(QwenImage, model_type='edit'),
}
