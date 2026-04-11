from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms


class DeepfakeDataset(Dataset):
    def __init__(self, real_folder, fake_folder):
        self.paths = []
        self.labels = []

        for img in os.listdir(real_folder):
            self.paths.append(os.path.join(real_folder, img))
            self.labels.append(0)

        for img in os.listdir(fake_folder):
            self.paths.append(os.path.join(fake_folder, img))
            self.labels.append(1)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(self.labels[idx])

        return img, label
