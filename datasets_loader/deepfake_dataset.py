from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms


class DeepfakeDataset(Dataset):
    def __init__(self, real_folder, fake_folder):
        self.paths = []
        self.labels = []

        # Load image names
        real_images = os.listdir(real_folder)
        fake_images = os.listdir(fake_folder)

        # Balance dataset
        min_len = min(len(real_images), len(fake_images))
        real_images = real_images[:min_len]
        fake_images = fake_images[:min_len]

        # Add real images
        for img in real_images:
            self.paths.append(os.path.join(real_folder, img))
            self.labels.append(0)

        # Add fake images
        for img in fake_images:
            self.paths.append(os.path.join(fake_folder, img))
            self.labels.append(1)

        # Image transforms
        self.transform = transforms.Compose([
   	    transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.ToTensor()
        ])

        # Debug prints
        print("Total real:", self.labels.count(0))
        print("Total fake:", self.labels.count(1))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(self.labels[idx])

        return img, label
