import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchgeo.datasets import LoveDA, LandCoverAI, DeepGlobeLandCover
from config import Config

# Statistiche ImageNet (i backbone pretrained le richiedono)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SatelliteSegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None, split="train"):
        """
        Wrapper su torchgeo. Sceglie il dataset in base a Config.DATASET_NAME.
        split: "train", "val" o "test".
        """
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.dataset_name = Config.DATASET_NAME.lower()

        os.makedirs(data_dir, exist_ok=True)
        print(f"Initializing TorchGeo {self.dataset_name.upper()} ({split} split)...")

        if self.dataset_name == "loveda":
            self.geo_dataset = LoveDA(root=self.data_dir, split=self.split, download=True, checksum=False)
        elif self.dataset_name == "landcoverai":
            self.geo_dataset = LandCoverAI(root=os.path.join(self.data_dir, "landcoverai"),
                                           split=self.split, download=True, checksum=False)
        elif self.dataset_name == "deepglobe":
            dg_split = "valid" if self.split == "val" else self.split
            self.geo_dataset = DeepGlobeLandCover(root=os.path.join(self.data_dir, "deepglobe"), split=dg_split)
        else:
            raise ValueError(f"Dataset {self.dataset_name} non implementato nel wrapper.")

    def __len__(self):
        return len(self.geo_dataset)

    def __getitem__(self, idx):
        sample = self.geo_dataset[idx]
        image = sample["image"]  # (C, H, W) tensor
        mask = sample["mask"]    # (H, W) o (1, H, W) tensor

        # numpy (H, W, C) per Albumentations
        image_np = image.numpy().transpose(1, 2, 0)
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

        mask_np = mask.numpy().squeeze().astype(np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image_np, mask=mask_np)
            image_np = augmented['image']
            mask_np = augmented['mask']

        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0

        # Normalizzazione ImageNet per i backbone pretrained
        if getattr(Config, "USE_IMAGENET_NORM", False):
            image_tensor = (image_tensor - _IMAGENET_MEAN) / _IMAGENET_STD

        mask_tensor = torch.from_numpy(mask_np).long()
        return image_tensor, mask_tensor
