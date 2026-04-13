import torch
from PIL import Image
from torchvision import transforms
import os

from models.deepfake_classifier import DeepfakeClassifier  # adjust if needed

# Load model
model = DeepfakeClassifier()
model.load_state_dict(torch.load("outputs/checkpoints/classifier.pth", map_location="cpu"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Class names
classes = ["REAL", "FAKE"]

# Test folder
folder = "test_images"

for img_name in os.listdir(folder):
    img_path = os.path.join(folder, img_name)

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    output = model(img)
    pred = torch.argmax(output, dim=1).item()

    print(f"{img_name} → {classes[pred]}")
