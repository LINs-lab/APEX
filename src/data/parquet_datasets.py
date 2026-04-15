
import os
from glob import glob
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HF_Dataset
from PIL import Image, ImageFile
from io import BytesIO
import pyarrow.parquet as pq
from torchvision import transforms

from services.tools import create_logger

logger = create_logger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Text2ImageParquetDataset(Dataset):
    def __init__(
        self,
        data_dirs,
        height=1024,
        width=1024,
        center_crop=True,
        random_flip=False,
        datasets_repeat=1,
        cache_dir=None,
    ):
        self.height = height
        self.width = width
        self.datasets_repeat = datasets_repeat
        self.image_processor = transforms.Compose(
            [
                transforms.Resize(min(height, width), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop((height, width)) if center_crop else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        if isinstance(data_dirs, list):
            data_pq = []
            for dir in data_dirs:
                data_pq.extend(glob(os.path.join(dir, "*.parquet")))
        else:
            data_pq = glob(os.path.join(data_dirs, "*.parquet"))

        self.length = 0
        for file in data_pq:
            self.length += pq.read_metadata(file).num_rows
        self.ds = HF_Dataset.from_parquet(data_pq, num_proc=32, cache_dir=cache_dir)

        logger.info(f"{self.length} (image, text) pairs loaded from {len(data_dirs)} directories.")

    def __len__(self):
        return self.length * self.datasets_repeat

    def __getitem__(self, index):
        idx = index % self.length
        item = self.ds[idx]
        text = item["text"]
        image_bytes = item["image_bytes"]
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = self.image_processor(image)
        z_bytes = item.get("z_bytes", None)
        if z_bytes is not None:
            z = torch.load(BytesIO(z_bytes))
        else:
            z = torch.empty(0)

        return [{"text": text, "image": image, "z": z}]