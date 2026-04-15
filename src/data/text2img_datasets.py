import os
import glob
from pathlib import Path

import pandas as pd
import torch
import torchvision
from PIL import Image
from torchvision import transforms

from services.tools import create_logger

logger = create_logger(__name__)

class Text2ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dirs,  # List[str]: list of directories containing image and txt files
        height=1024,
        width=1024,
        center_crop=True,
        random_flip=False,
        datasets_repeat=1,
        image_extensions=("jpg", "jpeg", "png", "webp", "bmp"),
    ):
        """
        Args:
            data_dirs: List of directory paths. Each dir contains image files (e.g., aaa.png)
                       and corresponding text files (e.g., aaa.txt) with same base name.
            height, width: target resolution
            center_crop: if True, center crop; else random crop
            random_flip: if True, apply random horizontal flip
            datasets_repeat: repeat dataset N times
            image_extensions: supported image file extensions (lowercase)
        """
        self.height = height
        self.width = width
        self.datasets_repeat = datasets_repeat

        # Build list of (image_path, text_path) pairs
        self.image_paths = []
        self.text_paths = []

        image_extensions = tuple(f".{ext.lower()}" for ext in image_extensions)

        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            if not data_dir.exists():
                logger.warning(f"Data directory {data_dir} does not exist. Skipping.")
                continue

            # Find all image files
            for img_path in data_dir.iterdir():
                if img_path.suffix.lower() not in image_extensions:
                    continue
                txt_path = img_path.with_suffix(".txt")
                if not txt_path.exists():
                    logger.warning(f"Corresponding text file not found: {txt_path}. Skipping {img_path}.")
                    continue

                self.image_paths.append(str(img_path))
                self.text_paths.append(str(txt_path))

        assert len(self.image_paths) > 0, "No valid (image, text) pairs found!"
        logger.info_rank0(f"{len(self.image_paths)} (image, text) pairs loaded from {len(data_dirs)} directories.")

        # Build image processor (same as original)
        self.image_processor = transforms.Compose(
            [
                transforms.Resize(min(height, width), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop((height, width)) if center_crop else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __getitem__(self, index):
        # Use deterministic index for reproducibility (same as original logic)
        data_id = (index) % len(self.image_paths)

        # Load image
        image = Image.open(self.image_paths[data_id]).convert("RGB")

        # Load prompts: read all lines from .txt, strip whitespace, ignore empty lines
        with open(self.text_paths[data_id], "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        if not prompts:
            # Fallback: use empty string if no prompts
            logger.warning(f"No prompts found in {self.text_paths[data_id]}. Using empty string.")
            prompts = [""]

        # Randomly choose one prompt if multiple exist (common practice in T2I training)
        text = prompts[torch.randint(0, len(prompts), ()).item()]

        # # Image preprocessing (same as original)
        # target_height, target_width = self.height, self.width
        # width, height = image.size
        # scale = max(target_width / width, target_height / height)
        # shape = [round(height * scale), round(width * scale)]
        # image = torchvision.transforms.functional.resize(
        #     image, shape, interpolation=transforms.InterpolationMode.BILINEAR
        # )
        image = self.image_processor(image)

        if torch.any(image < -1) or torch.any(image > 1):
            logger.warning("Image values are outside the expected range [-1, 1].")

        return [{"text": text, "image": image, "z": torch.empty(0)}]

    def __len__(self):
        return len(self.image_paths) * self.datasets_repeat