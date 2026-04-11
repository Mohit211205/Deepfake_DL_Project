from torch.utils.data import Dataset
from PIL import Image
import os

from utils.augmentations import augment


class BYOLDataset(Dataset):
    def __init__(self, folder):
        self.paths = [
            os.path.join(folder, img)
            for img in os.listdir(folder)
            if img.endswith(".jpg")
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")

        x1 = augment(img)
        x2 = augment(img)

        return x1, x2
